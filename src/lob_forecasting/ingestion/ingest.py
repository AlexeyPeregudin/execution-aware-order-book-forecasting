"""Fetch (or find) the raw files and write the manifest.

A few things we're careful about here:
  - never overwrite a raw file that already exists (unless overwrite=True)
  - always store a checksum
  - write to a temp file first so a half-written file never gets registered
  - if a download's checksum doesn't match what was expected, raise
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from ..config import ExperimentConfig
from ..utils.hashing import sha256_bytes, sha256_file
from .manifest import ManifestEntry, RawDataManifest, write_checksums
from .sources import DataSourceAdapter, IngestionError, SourceSpec, build_adapter


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_source_id(spec: SourceSpec) -> str:
    """Stable id for a (venue, symbol, date). Used to dedup manifest entries."""
    return f"{spec.venue}_{spec.symbol}_{spec.date:%Y%m%d}"


def _dest_path(raw_root: Path, spec: SourceSpec) -> Path:
    return raw_root / spec.venue / spec.symbol / f"{spec.date.isoformat()}.{spec.file_format}"


def _count_csv_rows(data: bytes) -> int | None:
    """Number of data rows in a CSV (minus the header). None for non-CSV."""
    text = data.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0
    return max(len(lines) - 1, 0)


def _atomic_write(dest: Path, data: bytes) -> None:
    """Write to a .tmp file then rename, so dest is never half-written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def _entry_from_bytes(
    spec: SourceSpec,
    rel_path: str,
    checksum: str,
    data: bytes,
    schema_version: str,
    created_at: str,
) -> ManifestEntry:
    row_count = _count_csv_rows(data) if spec.file_format == "csv" else None
    return ManifestEntry(
        source_id=make_source_id(spec),
        venue=spec.venue,
        symbol=spec.symbol,
        date=spec.date,
        source_url_or_path=spec.source_url_or_path,
        file_path=rel_path,
        file_format=spec.file_format,
        schema_version=schema_version,
        created_at_utc=created_at,
        checksum_sha256=checksum,
        row_count_or_null=row_count,
        notes=spec.notes,
    )


def ingest_raw_data(
    config: ExperimentConfig,
    project_root: str | Path | None = None,
    adapter: DataSourceAdapter | None = None,
) -> RawDataManifest:
    """Get all the raw files for this config and return the manifest.

    The adapter argument is mostly for tests; normally we build it from
    config.ingestion. The manifest and checksum files are written out too.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    ing = config.ingestion
    raw_root = root / config.data.raw_dir
    manifest_path = root / ing.manifest_path
    checksums_path = root / ing.checksums_path

    src = adapter if adapter is not None else build_adapter(ing)
    specs = src.enumerate(config, root)

    prior = RawDataManifest.load(manifest_path).by_id()
    result: dict[str, ManifestEntry] = {}

    for spec in specs:
        sid = make_source_id(spec)
        dest = _dest_path(raw_root, spec)
        rel_path = dest.relative_to(root).as_posix()
        prior_entry = prior.get(sid)

        # file already there and we're not overwriting -> reuse it
        if dest.exists() and not ing.overwrite:
            checksum = sha256_file(dest)
            if prior_entry is not None and prior_entry.checksum_sha256 == checksum:
                # nothing changed, keep the old record as-is
                result[sid] = prior_entry
                continue
            # file is there but not recorded (or changed); register it without
            # refetching, since we don't overwrite existing raw files
            data = dest.read_bytes()
            created = prior_entry.created_at_utc if prior_entry else _now_utc()
            result[sid] = _entry_from_bytes(
                spec, rel_path, checksum, data, ing.schema_version, created
            )
            continue

        # otherwise fetch it
        data = src.read_bytes(spec)
        checksum = sha256_bytes(data)
        if spec.expected_checksum and spec.expected_checksum != checksum:
            raise IngestionError(
                f"Corrupted download for {spec.source_url_or_path}: "
                f"expected sha256 {spec.expected_checksum}, got {checksum}"
            )
        _atomic_write(dest, data)
        # double-check what actually landed on disk
        if sha256_file(dest) != checksum:
            raise IngestionError(f"Post-write checksum mismatch for {rel_path}; aborting.")
        result[sid] = _entry_from_bytes(
            spec, rel_path, checksum, data, ing.schema_version, _now_utc()
        )

    manifest = RawDataManifest(entries=list(result.values()))
    manifest.save(manifest_path)
    write_checksums(checksums_path, manifest)
    return manifest


def verify_raw_files(
    manifest: RawDataManifest, project_root: str | Path | None = None
) -> None:
    """Check every file in the manifest exists and still matches its checksum."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    missing: list[str] = []
    mismatched: list[str] = []
    for entry in manifest.entries:
        path = root / entry.file_path
        if not path.exists():
            missing.append(entry.file_path)
            continue
        if sha256_file(path) != entry.checksum_sha256:
            mismatched.append(entry.file_path)
    if missing:
        raise FileNotFoundError(
            "Missing raw files referenced by manifest: " + ", ".join(missing)
        )
    if mismatched:
        raise IngestionError("Checksum mismatch for raw files: " + ", ".join(mismatched))
