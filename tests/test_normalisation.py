"""Tests for the normalisation module."""

from __future__ import annotations

import copy
from datetime import date

import pandas as pd
import pytest

from lob_forecasting.config import ExperimentConfig
from lob_forecasting.ingestion import ManifestEntry, RawDataManifest, ingest_raw_data
from lob_forecasting.normalisation import (
    EVENT_COLUMNS,
    EventTableIndex,
    SnapshotCsvAdapter,
    build_venue_adapter,
    finalise_events,
    normalise_events,
)

# Config + helpers

BASE_CONFIG: dict = {
    "data": {
        "venue": "binance",
        "symbols": ["BTCUSDT"],
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "top_k": 5,
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
        "synthetic": {"num_days": 1, "rows_per_day": 20, "seed": 7},
    },
}


def make_config(**overrides) -> ExperimentConfig:
    cfg = copy.deepcopy(BASE_CONFIG)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return ExperimentConfig.model_validate(cfg)


def _entry() -> ManifestEntry:
    return ManifestEntry(
        source_id="binance_BTCUSDT_20240101",
        venue="binance",
        symbol="BTCUSDT",
        date=date(2024, 1, 1),
        source_url_or_path="x",
        file_path="data/raw/binance/BTCUSDT/2024-01-01.csv",
        file_format="csv",
        schema_version="1",
        created_at_utc="2024-01-01T00:00:00+00:00",
        checksum_sha256="abc",
    )


def _write_snapshot_csv(path, rows: list[str], header_k: int = 2) -> None:
    cols = ["timestamp_ms"]
    for k in range(1, header_k + 1):
        cols += [f"bid_px_{k}", f"bid_qty_{k}", f"ask_px_{k}", f"ask_qty_{k}"]
    path.write_text("\n".join([",".join(cols), *rows]) + "\n", encoding="utf-8")




def test_known_sample_converts_to_canonical_rows(tmp_path):
    raw = tmp_path / "snap.csv"
    # One snapshot, K=1: ts=1ms, bid 100@2, ask 101@3.
    _write_snapshot_csv(raw, ["1,100.0,2.0,101.0,3.0"], header_k=1)

    result = SnapshotCsvAdapter().parse(raw, _entry())
    events = finalise_events(result.events)

    assert list(events.columns) == list(EVENT_COLUMNS)
    assert len(events) == 2  # 1 bid + 1 ask

    bid = events[events["side"] == "bid"].iloc[0]
    ask = events[events["side"] == "ask"].iloc[0]
    assert bid["timestamp_exchange_ns"] == 1_000_000  # 1 ms -> ns
    assert bid["event_type"] == "snapshot"
    assert bool(bid["is_snapshot"]) is True
    assert bid["price"] == 100.0 and bid["quantity"] == 2.0
    assert ask["price"] == 101.0 and ask["quantity"] == 3.0
    assert bid["raw_file_id"] == "binance_BTCUSDT_20240101"
    assert (events["quality_flags"] == "").all()
    # Monotone event ids starting at 0.
    assert list(events["event_id"]) == [0, 1]




def test_unsorted_messages_sorted_deterministically():
    # Provisional trade events deliberately out of timestamp order.
    df = pd.DataFrame(
        {
            "timestamp_exchange_ns": [300, 100, 200, 100],
            "venue": "binance",
            "symbol": "BTCUSDT",
            "event_type": "trade",
            "side": "buy",
            "price": [10.0, 11.0, 12.0, 13.0],
            "quantity": [1.0, 1.0, 1.0, 1.0],
            "update_id": [4, 1, 3, 2],
            "trade_id": pd.NA,
            "is_snapshot": False,
            "raw_file_id": "f",
            "quality_flags": "",
        }
    )
    out = finalise_events(df)

    # Sorted by (timestamp, update_id): ts100/uid1, ts100/uid2, ts200/uid3, ts300/uid4.
    assert list(out["timestamp_exchange_ns"]) == [100, 100, 200, 300]
    assert list(out["update_id"]) == [1, 2, 3, 4]
    assert list(out["event_id"]) == [0, 1, 2, 3]

    # Deterministic: same input -> identical output.
    out2 = finalise_events(df)
    pd.testing.assert_frame_equal(out, out2)




def test_zero_or_negative_prices_flagged():
    df = pd.DataFrame(
        {
            "timestamp_exchange_ns": [1, 2, 3],
            "venue": "binance",
            "symbol": "BTCUSDT",
            "event_type": "snapshot",
            "side": "bid",
            "price": [100.0, 0.0, -5.0],
            "quantity": [1.0, 1.0, 1.0],
            "update_id": [1, 2, 3],
            "trade_id": pd.NA,
            "is_snapshot": True,
            "raw_file_id": "f",
            "quality_flags": "",
        }
    )
    out = finalise_events(df)

    assert out.iloc[0]["quality_flags"] == ""
    assert "nonpositive_price" in out.iloc[1]["quality_flags"]
    assert "nonpositive_price" in out.iloc[2]["quality_flags"]
    # Invariant: price > 0 whenever price is not null -> bad prices nulled.
    assert pd.isna(out.iloc[1]["price"])
    assert pd.isna(out.iloc[2]["price"])
    not_null = out["price"].notna()
    assert (out.loc[not_null, "price"] > 0).all()


def test_negative_quantity_flagged():
    df = pd.DataFrame(
        {
            "timestamp_exchange_ns": [1, 2],
            "venue": "binance",
            "symbol": "BTCUSDT",
            "event_type": "snapshot",
            "side": "bid",
            "price": [100.0, 100.0],
            "quantity": [0.0, -1.0],  # zero is allowed (>= 0)
            "update_id": [1, 2],
            "trade_id": pd.NA,
            "is_snapshot": True,
            "raw_file_id": "f",
            "quality_flags": "",
        }
    )
    out = finalise_events(df)
    assert out.iloc[0]["quality_flags"] == ""  # qty 0 is clean
    assert "negative_quantity" in out.iloc[1]["quality_flags"]
    assert pd.isna(out.iloc[1]["quantity"])




def test_duplicate_update_ids_flagged():
    df = pd.DataFrame(
        {
            "timestamp_exchange_ns": [1, 2, 3],
            "venue": "binance",
            "symbol": "BTCUSDT",
            "event_type": "depth_update",
            "side": "bid",
            "price": [100.0, 101.0, 102.0],
            "quantity": [1.0, 1.0, 1.0],
            "update_id": [5, 5, 6],  # first two collide
            "trade_id": pd.NA,
            "is_snapshot": False,
            "raw_file_id": "f",
            "quality_flags": "",
        }
    )
    out = finalise_events(df).sort_values("update_id").reset_index(drop=True)
    dup_rows = out[out["update_id"] == 5]
    assert (dup_rows["quality_flags"].str.contains("duplicate_update_id")).all()
    assert "duplicate_update_id" not in out[out["update_id"] == 6].iloc[0]["quality_flags"]


def test_idless_rows_not_falsely_flagged_as_duplicates():
    # Many id-less rows must not all be flagged duplicates of one another.
    df = pd.DataFrame(
        {
            "timestamp_exchange_ns": [1, 2, 3],
            "venue": "binance",
            "symbol": "BTCUSDT",
            "event_type": "trade",
            "side": "buy",
            "price": [100.0, 100.0, 100.0],
            "quantity": [1.0, 1.0, 1.0],
            "update_id": pd.NA,
            "trade_id": pd.NA,
            "is_snapshot": False,
            "raw_file_id": "f",
            "quality_flags": "",
        }
    )
    out = finalise_events(df)
    assert (out["quality_flags"] == "").all()


# Unknown event-type policy


def _unknown_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_exchange_ns": [1, 2],
            "venue": "binance",
            "symbol": "BTCUSDT",
            "event_type": ["trade", "weird_type"],
            "side": pd.NA,
            "price": [1.0, 1.0],
            "quantity": [1.0, 1.0],
            "update_id": [1, 2],
            "trade_id": pd.NA,
            "is_snapshot": False,
            "raw_file_id": "f",
            "quality_flags": "",
        }
    )


def test_unknown_event_type_flag_policy():
    out = finalise_events(_unknown_df(), unknown_event_type_policy="flag")
    assert len(out) == 2
    weird = out[out["event_type"] == "weird_type"].iloc[0]
    assert "unknown_event_type" in weird["quality_flags"]


def test_unknown_event_type_reject_policy():
    out = finalise_events(_unknown_df(), unknown_event_type_policy="reject")
    assert len(out) == 1
    assert set(out["event_type"]) == {"trade"}
    assert list(out["event_id"]) == [0]  # ids reassigned after rejection


# Orchestration: end-to-end with ingestion + Parquet round-trip


def test_normalise_events_end_to_end(tmp_path):
    config = make_config()
    manifest = ingest_raw_data(config, project_root=tmp_path)
    index = normalise_events(config, manifest, project_root=tmp_path)

    assert len(index) == 1  # 1 symbol * 1 day
    part = index.partitions[0]
    # 20 snapshots * top_k=5 * 2 sides = 200 events.
    assert part.row_count == 200

    parquet_path = tmp_path / part.file_path
    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    assert list(df.columns) == list(EVENT_COLUMNS)

    # Output invariants (5.3).
    assert df["event_id"].is_monotonic_increasing
    assert list(df["event_id"]) == list(range(len(df)))
    assert df["timestamp_exchange_ns"].is_monotonic_increasing
    assert (df["price"].dropna() > 0).all()
    assert (df["quantity"].dropna() >= 0).all()
    assert set(df["event_type"].unique()) == {"snapshot"}
    assert df["is_snapshot"].all()
    assert df["quality_flags"].notna().all()


def test_event_index_round_trips(tmp_path):
    config = make_config()
    manifest = ingest_raw_data(config, project_root=tmp_path)
    normalise_events(config, manifest, project_root=tmp_path)

    idx_path = tmp_path / config.data.processed_dir / "events" / "events_index.yaml"
    assert idx_path.exists()
    reloaded = EventTableIndex.load(idx_path)
    assert reloaded.total_rows() == 200


def test_missing_raw_file_raises(tmp_path):
    config = make_config()
    manifest = RawDataManifest(entries=[_entry()])  # points to a non-existent file
    with pytest.raises(FileNotFoundError, match="missing"):
        normalise_events(config, manifest, project_root=tmp_path)


def test_adapter_selection():
    assert isinstance(build_venue_adapter("binance"), SnapshotCsvAdapter)
    assert isinstance(build_venue_adapter("anything", override="snapshot_csv"), SnapshotCsvAdapter)
