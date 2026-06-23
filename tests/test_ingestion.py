"""Tests for the ingestion module."""

from __future__ import annotations

import copy

import pytest

from lob_forecasting.config import ExperimentConfig
from lob_forecasting.ingestion import (
    IngestionError,
    ManifestEntry,
    RawDataManifest,
    SourceSpec,
    build_adapter,
    ingest_raw_data,
    make_source_id,
    verify_raw_files,
)
from lob_forecasting.ingestion.sources import DataSourceAdapter, SyntheticSampleAdapter

# Config fixtures

BASE_CONFIG: dict = {
    "data": {
        "venue": "binance",
        "symbols": ["BTCUSDT"],
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "top_k": 5,
        "raw_dir": "data/raw",
    },
    "sampling": {"horizons_events": [10, 50, 200]},
    "labels": {},
    "splits": {
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "embargo_events": 200,
    },
    "features": {},
    "models": {"run": ["no_change"]},
    "backtest": {"threshold_grid": [0.0, 0.0001]},
    "ingestion": {
        "mode": "synthetic",
        "synthetic": {"num_days": 2, "rows_per_day": 50, "seed": 7},
    },
}


def make_config(**ingestion_overrides) -> ExperimentConfig:
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["ingestion"].update(ingestion_overrides)
    return ExperimentConfig.model_validate(cfg)


# Synthetic ingestion: files + manifest


def test_synthetic_creates_files_and_manifest(tmp_path):
    config = make_config()
    manifest = ingest_raw_data(config, project_root=tmp_path)

    # 1 symbol * 2 days = 2 files.
    assert len(manifest) == 2
    for entry in manifest.entries:
        assert (tmp_path / entry.file_path).exists()
        assert entry.checksum_sha256  # non-empty
        assert entry.venue == "binance"
        assert entry.symbol == "BTCUSDT"
        assert entry.row_count_or_null == 50

    # Manifest + checksum files were written.
    assert (tmp_path / config.ingestion.manifest_path).exists()
    assert (tmp_path / config.ingestion.checksums_path).exists()


def test_manifest_round_trips_from_disk(tmp_path):
    config = make_config()
    ingest_raw_data(config, project_root=tmp_path)
    reloaded = RawDataManifest.load(tmp_path / config.ingestion.manifest_path)
    assert len(reloaded) == 2
    assert all(e.checksum_sha256 for e in reloaded.entries)




def test_checksum_stable_across_regeneration(tmp_path):
    config = make_config()
    m1 = ingest_raw_data(config, project_root=tmp_path)

    # Independently regenerate the bytes via the adapter; checksum must match.
    adapter = build_adapter(config.ingestion)
    specs = adapter.enumerate(config, tmp_path)
    from lob_forecasting.utils.hashing import sha256_bytes

    by_id = {make_source_id(s): adapter.read_bytes(s) for s in specs}
    for entry in m1.entries:
        assert sha256_bytes(by_id[entry.source_id]) == entry.checksum_sha256


def test_synthetic_bytes_are_deterministic():
    ing = make_config().ingestion
    adapter = SyntheticSampleAdapter(ing)
    spec = SourceSpec(
        venue="binance",
        symbol="BTCUSDT",
        date=__import__("datetime").date(2024, 1, 1),
        file_format="csv",
        source_url_or_path="synthetic://x",
        extra={"top_k": 5},
    )
    assert adapter.read_bytes(spec) == adapter.read_bytes(spec)




def test_duplicate_ingestion_is_idempotent(tmp_path):
    config = make_config()
    m1 = ingest_raw_data(config, project_root=tmp_path)
    m2 = ingest_raw_data(config, project_root=tmp_path)

    assert len(m1) == len(m2) == 2
    ids1 = {e.source_id for e in m1.entries}
    ids2 = {e.source_id for e in m2.entries}
    assert ids1 == ids2

    # created_at is preserved across the skip path (no spurious rewrites).
    created1 = {e.source_id: e.created_at_utc for e in m1.entries}
    created2 = {e.source_id: e.created_at_utc for e in m2.entries}
    assert created1 == created2


def test_raw_files_immutable_on_second_run(tmp_path):
    config = make_config()
    ingest_raw_data(config, project_root=tmp_path)
    raw_file = next((tmp_path / "data/raw").rglob("*.csv"))
    original_bytes = raw_file.read_bytes()
    mtime_before = raw_file.stat().st_mtime_ns

    ingest_raw_data(config, project_root=tmp_path)  # second run skips
    assert raw_file.read_bytes() == original_bytes
    assert raw_file.stat().st_mtime_ns == mtime_before




def test_verify_missing_raw_file_raises(tmp_path):
    manifest = RawDataManifest(
        entries=[
            ManifestEntry(
                source_id="binance_BTCUSDT_20240101",
                venue="binance",
                symbol="BTCUSDT",
                date=__import__("datetime").date(2024, 1, 1),
                source_url_or_path="x",
                file_path="data/raw/binance/BTCUSDT/2024-01-01.csv",
                file_format="csv",
                schema_version="1",
                created_at_utc="2024-01-01T00:00:00+00:00",
                checksum_sha256="deadbeef",
            )
        ]
    )
    with pytest.raises(FileNotFoundError, match="Missing raw files"):
        verify_raw_files(manifest, project_root=tmp_path)


def test_verify_passes_after_ingestion(tmp_path):
    config = make_config()
    manifest = ingest_raw_data(config, project_root=tmp_path)
    verify_raw_files(manifest, project_root=tmp_path)  # must not raise


def test_verify_detects_tampered_file(tmp_path):
    config = make_config()
    manifest = ingest_raw_data(config, project_root=tmp_path)
    tampered = next((tmp_path / "data/raw").rglob("*.csv"))
    tampered.write_bytes(b"corrupted contents\n")
    with pytest.raises(IngestionError, match="Checksum mismatch"):
        verify_raw_files(manifest, project_root=tmp_path)




class _BadChecksumAdapter(DataSourceAdapter):
    """Adapter that advertises a wrong expected checksum to simulate corruption."""

    mode = "synthetic"

    def enumerate(self, config, project_root):
        return [
            SourceSpec(
                venue="binance",
                symbol="BTCUSDT",
                date=__import__("datetime").date(2024, 1, 1),
                file_format="csv",
                source_url_or_path="bad://download",
                expected_checksum="0" * 64,  # will never match
            )
        ]

    def read_bytes(self, spec):
        return b"hello world\n"


def test_corrupted_download_fails_loudly(tmp_path):
    config = make_config()
    with pytest.raises(IngestionError, match="Corrupted download"):
        ingest_raw_data(config, project_root=tmp_path, adapter=_BadChecksumAdapter())


# Local archive & fixture modes


def test_local_archive_mode(tmp_path):
    # Stage a fake archive tree.
    src_root = tmp_path / "external"
    f = src_root / "binance" / "BTCUSDT" / "2024-01-01.csv"
    f.parent.mkdir(parents=True)
    f.write_text("timestamp_ms,bid_px_1\n1,100.0\n", encoding="utf-8")

    config = make_config(mode="local_archive", source_root=str(src_root))
    manifest = ingest_raw_data(config, project_root=tmp_path)

    assert len(manifest) == 1
    entry = manifest.entries[0]
    assert (tmp_path / entry.file_path).read_text(encoding="utf-8").startswith("timestamp_ms")
    assert entry.row_count_or_null == 1


def test_make_source_id_is_deterministic():
    spec = SourceSpec(
        venue="binance",
        symbol="BTCUSDT",
        date=__import__("datetime").date(2024, 1, 1),
        file_format="csv",
        source_url_or_path="x",
    )
    assert make_source_id(spec) == "binance_BTCUSDT_20240101"
