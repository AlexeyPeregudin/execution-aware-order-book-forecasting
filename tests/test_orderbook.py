"""Tests for the order-book reconstruction module."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from lob_forecasting.config import ExperimentConfig
from lob_forecasting.ingestion import ingest_raw_data
from lob_forecasting.normalisation import normalise_events
from lob_forecasting.orderbook import (
    BookState,
    build_order_books,
    compute_derived,
    reconstruct_book,
)
from lob_forecasting.orderbook.book_table import book_columns

# Event-frame builder

_EV_COLS = [
    "event_id", "timestamp_exchange_ns", "venue", "symbol", "event_type",
    "side", "price", "quantity", "update_id", "is_snapshot",
]


def _events(rows: list[dict]) -> pd.DataFrame:
    """Build a normalised-event DataFrame from a list of partial row dicts."""
    out = []
    for i, r in enumerate(rows):
        base = {
            "event_id": r.get("event_id", i),
            "timestamp_exchange_ns": r["ts"],
            "venue": "binance",
            "symbol": "BTCUSDT",
            "event_type": r.get("event_type", "snapshot"),
            "side": r.get("side", pd.NA),
            "price": r.get("price", pd.NA),
            "quantity": r.get("quantity", pd.NA),
            "update_id": r.get("update_id", pd.NA),
            "is_snapshot": r.get("is_snapshot", r.get("event_type", "snapshot") == "snapshot"),
        }
        out.append(base)
    return pd.DataFrame(out, columns=_EV_COLS)


def _snapshot_group(ts: int, bids: list[tuple], asks: list[tuple], start_id: int = 0) -> list[dict]:
    """Emit snapshot level events for one timestamp."""
    rows = []
    eid = start_id
    for px, qty in bids:
        rows.append({"event_id": eid, "ts": ts, "event_type": "snapshot", "side": "bid", "price": px, "quantity": qty})
        eid += 1
    for px, qty in asks:
        rows.append({"event_id": eid, "ts": ts, "event_type": "snapshot", "side": "ask", "price": px, "quantity": qty})
        eid += 1
    return rows


# BookState unit behaviour


class TestBookState:
    def test_adding_bid_liquidity_updates_best_bid(self):
        state = BookState()
        state.apply("bid", 100.0, 1.0)
        assert state.best_bid() == 100.0
        # A higher bid becomes the new best bid.
        state.apply("bid", 101.0, 2.0)
        assert state.best_bid() == 101.0

    def test_removing_price_level_deletes_it(self):
        state = BookState()
        state.apply("bid", 100.0, 1.0)
        state.apply("bid", 101.0, 2.0)
        assert state.best_bid() == 101.0
        # Zero quantity removes the level entirely.
        state.apply("bid", 101.0, 0.0)
        assert 101.0 not in state.bids
        assert state.best_bid() == 100.0

    def test_best_ask_is_lowest(self):
        state = BookState()
        state.apply("ask", 105.0, 1.0)
        state.apply("ask", 103.0, 1.0)
        assert state.best_ask() == 103.0

    def test_top_k_truncates_and_sorts(self):
        state = BookState()
        for px in (100.0, 98.0, 99.0, 97.0):
            state.apply("bid", px, 1.0)
        for px in (101.0, 103.0, 102.0):
            state.apply("ask", px, 1.0)
        bids, asks = state.top_k(2)
        assert [p for p, _ in bids] == [100.0, 99.0]
        assert [p for p, _ in asks] == [101.0, 102.0]


def test_microprice_matches_formula():
    # bid 100 @ 2, ask 102 @ 8  ->  micro = (102*2 + 100*8)/(8+2) = 1004/10 = 100.4
    row = compute_derived(bids=[(100.0, 2.0)], asks=[(102.0, 8.0)], k=1)
    assert row["mid"] == 101.0
    assert row["spread"] == 2.0
    assert row["microprice"] == pytest.approx(100.4)
    assert row["relative_spread"] == pytest.approx(2.0 / 101.0)


def test_derived_null_when_side_missing():
    row = compute_derived(bids=[(100.0, 1.0)], asks=[], k=1)
    assert pd.isna(row["mid"])
    assert pd.isna(row["spread"])
    assert pd.isna(row["microprice"])
    assert "no_best_ask" in row["quality_flags"]




def test_crossed_book_detected():
    # best bid 105 > best ask 103  -> crossed
    row = compute_derived(bids=[(105.0, 1.0)], asks=[(103.0, 1.0)], k=1)
    assert row["book_is_crossed"] is True
    assert "crossed_book" in row["quality_flags"]
    assert row["spread"] < 0  # still priced, just flagged


def test_crossed_book_detected_in_reconstruction():
    events = _events(_snapshot_group(1, bids=[(105.0, 1.0)], asks=[(103.0, 1.0)]))
    book = reconstruct_book(events, top_k=1)
    assert len(book) == 1
    assert bool(book.iloc[0]["book_is_crossed"]) is True




def test_topk_output_is_sorted():
    bids = [(100.0, 1.0), (99.0, 1.0), (98.0, 1.0)]
    asks = [(101.0, 1.0), (102.0, 1.0), (103.0, 1.0)]
    events = _events(_snapshot_group(1, bids=bids, asks=asks))
    book = reconstruct_book(events, top_k=3)
    r = book.iloc[0]
    assert r["bid_px_1"] > r["bid_px_2"] > r["bid_px_3"]
    assert r["ask_px_1"] < r["ask_px_2"] < r["ask_px_3"]
    # Best levels are the inside market.
    assert r["bid_px_1"] == 100.0
    assert r["ask_px_1"] == 101.0




def test_sequence_gap_flagged_on_update_id_jump():
    rows = [
        # seed the book with a snapshot (resets the sequence)
        *_snapshot_group(1, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)], start_id=0),
        # incremental depth updates with a jump 2 -> 5
        {"event_id": 2, "ts": 2, "event_type": "depth_update", "is_snapshot": False, "side": "bid", "price": 100.0, "quantity": 2.0, "update_id": 1},
        {"event_id": 3, "ts": 3, "event_type": "depth_update", "is_snapshot": False, "side": "bid", "price": 99.5, "quantity": 1.0, "update_id": 2},
        {"event_id": 4, "ts": 4, "event_type": "depth_update", "is_snapshot": False, "side": "bid", "price": 99.0, "quantity": 1.0, "update_id": 5},
    ]
    book = reconstruct_book(_events(rows), top_k=2, mode="replay").sort_values("timestamp_exchange_ns")
    gap_flags = dict(zip(book["timestamp_exchange_ns"], book["book_update_gap_detected"]))
    assert gap_flags[3] == False  # 1 -> 2 contiguous
    assert gap_flags[4] == True   # 2 -> 5 is a gap


# Replay engine: incremental updates change the book over time


def test_replay_incremental_updates_and_removal():
    rows = [
        *_snapshot_group(1, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)], start_id=0),
        # add a better bid at t=2
        {"event_id": 2, "ts": 2, "event_type": "depth_update", "is_snapshot": False, "side": "bid", "price": 100.5, "quantity": 3.0, "update_id": 1},
        # remove it at t=3 -> best bid reverts to 100
        {"event_id": 3, "ts": 3, "event_type": "depth_update", "is_snapshot": False, "side": "bid", "price": 100.5, "quantity": 0.0, "update_id": 2},
    ]
    book = reconstruct_book(_events(rows), top_k=1, mode="replay").sort_values("timestamp_exchange_ns").reset_index(drop=True)
    assert list(book["timestamp_exchange_ns"]) == [1, 2, 3]
    assert book.loc[0, "bid_px_1"] == 100.0
    assert book.loc[1, "bid_px_1"] == 100.5  # better bid took over
    assert book.loc[2, "bid_px_1"] == 100.0  # removed -> reverted


# Schema, missing levels, monotonicity


def test_missing_levels_flagged():
    # Only 1 bid level but K=2 requested -> missing levels.
    events = _events(_snapshot_group(1, bids=[(100.0, 1.0)], asks=[(101.0, 1.0), (102.0, 1.0)]))
    book = reconstruct_book(events, top_k=2)
    r = book.iloc[0]
    assert bool(r["book_has_missing_levels"]) is True
    assert pd.isna(r["bid_px_2"])  # absent level is null


def test_output_schema_matches_spec():
    events = _events(_snapshot_group(1, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]))
    book = reconstruct_book(events, top_k=10)
    assert list(book.columns) == book_columns(10)


def test_snapshot_mode_rejects_non_snapshot_events():
    rows = [
        {"event_id": 0, "ts": 1, "event_type": "depth_update", "is_snapshot": False, "side": "bid", "price": 100.0, "quantity": 1.0, "update_id": 1},
    ]
    from lob_forecasting.orderbook import OrderBookError
    with pytest.raises(OrderBookError, match="non-snapshot"):
        reconstruct_book(_events(rows), top_k=1, mode="snapshot")


# End-to-end: ingest -> normalise -> build_order_books

BASE_CONFIG: dict = {
    "data": {"venue": "binance", "symbols": ["BTCUSDT"], "start_date": "2024-01-01", "end_date": "2024-03-31", "top_k": 10},
    "sampling": {"horizons_events": [10, 50, 200]},
    "labels": {},
    "splits": {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "embargo_events": 200},
    "features": {},
    "models": {"run": ["no_change"]},
    "backtest": {"threshold_grid": [0.0, 0.0001]},
    "ingestion": {"mode": "synthetic", "synthetic": {"num_days": 1, "rows_per_day": 30, "seed": 7}},
}


def test_build_order_books_end_to_end(tmp_path):
    config = ExperimentConfig.model_validate(copy.deepcopy(BASE_CONFIG))
    manifest = ingest_raw_data(config, project_root=tmp_path)
    events = normalise_events(config, manifest, project_root=tmp_path)
    books = build_order_books(config, events, project_root=tmp_path)

    assert len(books) == 1
    part = books.partitions[0]
    assert part.row_count == 30  # 30 snapshots -> 30 book rows
    assert part.top_k == 10

    df = pd.read_parquet(tmp_path / part.file_path)
    assert list(df.columns) == book_columns(10)
    # Output invariants (5.4).
    assert df["timestamp_exchange_ns"].is_monotonic_increasing
    assert (df["bid_px_1"] > df["bid_px_2"]).all()   # strictly decreasing bids
    assert (df["ask_px_1"] < df["ask_px_2"]).all()   # strictly increasing asks
    clean = df["quality_flags"] == ""
    assert (df.loc[clean, "spread"] >= 0).all()       # clean rows: spread >= 0
    assert df.loc[clean, "mid"].notna().all()         # clean rows: best bid/ask present
    # Synthetic data is well-formed: no crossed books, full depth.
    assert part.n_crossed == 0
    assert part.n_missing_levels == 0


def test_books_index_round_trips(tmp_path):
    config = ExperimentConfig.model_validate(copy.deepcopy(BASE_CONFIG))
    manifest = ingest_raw_data(config, project_root=tmp_path)
    events = normalise_events(config, manifest, project_root=tmp_path)
    build_order_books(config, events, project_root=tmp_path)

    from lob_forecasting.orderbook import BookTableIndex
    idx_path = tmp_path / config.data.processed_dir / "books" / "books_index.yaml"
    assert idx_path.exists()
    reloaded = BookTableIndex.load(idx_path)
    assert reloaded.total_rows() == 30
