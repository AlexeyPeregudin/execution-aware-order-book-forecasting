"""The raw-data manifest: one record per raw file, plus load/save helpers.

The manifest is what lets later steps find the raw files and check they haven't
changed (via the sha256 checksum).
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ManifestEntry(BaseModel):
    """One raw file and everything we know about it.

    checksum_sha256 is required, so an entry without a checksum won't validate.
    """

    source_id: str
    venue: str
    symbol: str
    date: date_type
    source_url_or_path: str
    file_path: str  # relative to the project root, forward slashes
    file_format: str
    schema_version: str
    created_at_utc: str
    checksum_sha256: str
    row_count_or_null: int | None = None
    notes: str = ""
    # only set when data was collected from a live stream
    collection_start_utc: str | None = None
    collection_end_utc: str | None = None


class RawDataManifest(BaseModel):
    """A list of ManifestEntry records, keyed by source_id."""

    entries: list[ManifestEntry] = Field(default_factory=list)

    def by_id(self) -> dict[str, ManifestEntry]:
        return {e.source_id: e for e in self.entries}

    def get(self, source_id: str) -> ManifestEntry | None:
        return self.by_id().get(source_id)

    def has(self, source_id: str) -> bool:
        return source_id in self.by_id()

    def __len__(self) -> int:
        return len(self.entries)

    def save(self, path: str | Path) -> Path:
        """Write the manifest to YAML. Entries are sorted so diffs stay small."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self.entries, key=lambda e: (e.venue, e.symbol, e.date, e.source_id))
        payload: dict[str, Any] = {
            "_meta": {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "entry_count": len(ordered),
            },
            "entries": [e.model_dump(mode="json") for e in ordered],
        }
        with out.open("w", encoding="utf-8") as fh:
            yaml.dump(payload, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "RawDataManifest":
        """Load a manifest from YAML, or return an empty one if the file is missing."""
        p = Path(path)
        if not p.exists():
            return cls(entries=[])
        with p.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        entries = raw.get("entries") or []
        return cls(entries=[ManifestEntry.model_validate(e) for e in entries])


def write_checksums(path: str | Path, manifest: RawDataManifest) -> Path:
    """Write a simple file_path -> checksum map to checksums.yml."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mapping = {
        e.file_path: e.checksum_sha256
        for e in sorted(manifest.entries, key=lambda e: e.file_path)
    }
    with out.open("w", encoding="utf-8") as fh:
        yaml.dump(mapping, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return out
