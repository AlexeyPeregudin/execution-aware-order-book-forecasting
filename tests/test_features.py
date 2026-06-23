"""Tests for the feature generation module."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from lob_forecasting.config import ExperimentConfig
from lob_forecasting.features import (
    build_features,
    compute_features,
    feature_columns,
    imbalance_l1,
    ofi_event_series,
    realised_vol,
    return_lag,
    rolling_ofi,
)
from lob_forecasting.ingestion import ingest_raw_data
from lob_forecasting.normalisation import normalise_events
from lob_forecasting.orderbook import build_order_books, reconstruct_book

# Config helpers


def make_config(**over) -> ExperimentConfig:
    cfg: dict = {
        "data": {"venue": "binance", "symbols": ["BTCUSDT"], "start_date": "2024-01-01", "end_date": "2024-03-31", "top_k": 2},
        "sampling": {"horizons_events": [1, 2], "feature_lookbacks_events": [1, 2]},
        "labels": {},
        "splits": {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "embargo_events": 2},
        "features": {"realised_vol_lookbacks_events": [2]},
        "models": {"run": ["no_change"]},
        "backtest": {"threshold_grid": [0.0, 0.0001]},
        "ingestion": {"mode": "synthetic", "synthetic": {"num_days": 1, "rows_per_day": 30, "seed": 7}},
    }
    for k, v in over.items():
        if isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return ExperimentConfig.model_validate(cfg)


def _book_df(rows: list[dict], k: int = 1) -> pd.DataFrame:
    """Build a minimal book table (top-k) from a list of best-level dicts."""
    out = []
    for i, r in enumerate(rows):
        row = {
            "event_id": r.get("event_id", i),
            "timestamp_exchange_ns": r["ts"],
            "venue": "binance",
            "symbol": "BTCUSDT",
        }
        for lvl in range(1, k + 1):
            row[f"bid_px_{lvl}"] = r.get(f"bid_px_{lvl}", np.nan)
            row[f"bid_qty_{lvl}"] = r.get(f"bid_qty_{lvl}", np.nan)
            row[f"ask_px_{lvl}"] = r.get(f"ask_px_{lvl}", np.nan)
            row[f"ask_qty_{lvl}"] = r.get(f"ask_qty_{lvl}", np.nan)
        bid1, ask1 = row["bid_px_1"], row["ask_px_1"]
        row["mid"] = r.get("mid", (bid1 + ask1) / 2)
        row["spread"] = ask1 - bid1
        row["relative_spread"] = row["spread"] / row["mid"]
        row["microprice"] = np.nan
        row["book_is_crossed"] = False
        row["book_has_missing_levels"] = False
        row["book_update_gap_detected"] = False
        row["quality_flags"] = r.get("quality_flags", "")
        out.append(row)
    return pd.DataFrame(out)


# A tiny, fully hand-computable book sequence (top-1).
KNOWN_ROWS = [
    {"ts": 10, "bid_px_1": 100.0, "bid_qty_1": 2.0, "ask_px_1": 101.0, "ask_qty_1": 3.0},
    {"ts": 20, "bid_px_1": 100.0, "bid_qty_1": 4.0, "ask_px_1": 101.0, "ask_qty_1": 3.0},
    {"ts": 30, "bid_px_1": 101.0, "bid_qty_1": 1.0, "ask_px_1": 102.0, "ask_qty_1": 5.0},
]




def test_known_sequence_matches_expected_values():
    config = make_config()
    book_df = _book_df(KNOWN_ROWS, k=2)  # k=2 but only level 1 present -> level 2 NaN
    feats = compute_features(book_df, config)

    # mid = (bid1 + ask1) / 2
    assert list(feats["mid"]) == [100.5, 100.5, 101.5]

    # imbalance_l1 = (q_b - q_a)/(q_b + q_a)
    assert feats["imbalance_l1"].iloc[0] == pytest.approx((2 - 3) / (2 + 3))
    assert feats["imbalance_l1"].iloc[1] == pytest.approx((4 - 3) / (4 + 3))
    assert feats["imbalance_l1"].iloc[2] == pytest.approx((1 - 5) / (1 + 5))

    # OFI events: e0=0, e1=+2, e2=+4  ->  ofi_2 = [NaN, 2, 6]
    assert pd.isna(feats["ofi_2"].iloc[0])
    assert feats["ofi_2"].iloc[1] == pytest.approx(2.0)
    assert feats["ofi_2"].iloc[2] == pytest.approx(6.0)

    # return_lag_1 = (m_t - m_{t-1}) / m_{t-1}
    assert pd.isna(feats["return_lag_1"].iloc[0])
    assert feats["return_lag_1"].iloc[1] == pytest.approx(0.0)
    assert feats["return_lag_1"].iloc[2] == pytest.approx((101.5 - 100.5) / 100.5)


def test_ofi_event_formula_known_values():
    bp = pd.Series([100.0, 100.0, 101.0])
    bq = pd.Series([2.0, 4.0, 1.0])
    ap = pd.Series([101.0, 101.0, 102.0])
    aq = pd.Series([3.0, 3.0, 5.0])
    e = ofi_event_series(bp, bq, ap, aq)
    assert list(e) == [0.0, 2.0, 4.0]


def test_multilevel_imbalance_known_values():
    config = make_config()
    rows = [{"ts": 1, "bid_px_1": 100.0, "bid_qty_1": 4.0, "ask_px_1": 101.0, "ask_qty_1": 2.0,
             "bid_px_2": 99.0, "bid_qty_2": 6.0, "ask_px_2": 102.0, "ask_qty_2": 8.0}]
    feats = compute_features(_book_df(rows, k=2), config)
    # weights w1=1, w2=1/2.  WB = 4 + 6/2 = 7 ;  WA = 2 + 8/2 = 6
    # I^(K) = (7 - 6)/(7 + 6) = 1/13
    assert feats["imbalance_lK"].iloc[0] == pytest.approx(1.0 / 13.0)




def test_return_lag_uses_only_current_and_lagged_mid():
    mid = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    out = return_lag(mid, 2)
    # Exact formula at t=2: (102 - 100)/100
    assert out.iloc[2] == pytest.approx((102.0 - 100.0) / 100.0)
    # First L rows are undefined.
    assert pd.isna(out.iloc[0]) and pd.isna(out.iloc[1])


def test_return_lag_is_causal_under_future_perturbation():
    config = make_config()
    book_df = _book_df(KNOWN_ROWS, k=2)
    base = compute_features(book_df, config)

    # Perturb the LAST row's mid; earlier feature values must not change.
    perturbed = _book_df(
        [*KNOWN_ROWS[:-1], {**KNOWN_ROWS[-1], "bid_px_1": 999.0, "ask_px_1": 1001.0}], k=2
    )
    after = compute_features(perturbed, config)

    for col in ("return_lag_1", "return_lag_2", "ofi_2", "realised_vol_2"):
        pd.testing.assert_series_equal(
            base[col].iloc[:-1], after[col].iloc[:-1], check_names=False
        )




def test_realised_vol_uses_history_only():
    mid = pd.Series([100.0, 101.0, 100.0, 102.0, 101.0])
    rv = realised_vol(mid, 2)
    # First row undefined (no return); row1 also NaN (needs 2 returns).
    assert pd.isna(rv.iloc[0]) and pd.isna(rv.iloc[1])
    # At t=2: returns at t1 and t2.
    r1 = np.log(101.0 / 100.0)
    r2 = np.log(100.0 / 101.0)
    assert rv.iloc[2] == pytest.approx(np.sqrt(r1**2 + r2**2))


def test_realised_vol_causal_under_future_perturbation():
    mid = pd.Series([100.0, 101.0, 102.0, 103.0])
    base = realised_vol(mid, 2)
    mid2 = mid.copy()
    mid2.iloc[-1] = 999.0  # change the future only
    after = realised_vol(mid2, 2)
    pd.testing.assert_series_equal(base.iloc[:-1], after.iloc[:-1], check_names=False)




def test_ofi_window_uses_rows_up_to_t():
    e = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    ofi3 = rolling_ofi(e, 3)
    # At t=4 the window is e[2:5] = 2+3+4 = 9 (rows up to t only).
    assert ofi3.iloc[4] == pytest.approx(9.0)
    # Not enough history before t=2.
    assert pd.isna(ofi3.iloc[0]) and pd.isna(ofi3.iloc[1])


def test_ofi_causal_under_future_perturbation():
    bp = pd.Series([100.0, 100.0, 101.0, 101.0, 102.0])
    bq = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    ap = pd.Series([101.0, 101.0, 102.0, 102.0, 103.0])
    aq = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    base = rolling_ofi(ofi_event_series(bp, bq, ap, aq), 2)

    bp2, bq2 = bp.copy(), bq.copy()
    bp2.iloc[-1] = 999.0  # perturb future row only
    bq2.iloc[-1] = 99.0
    after = rolling_ofi(ofi_event_series(bp2, bq2, ap, aq), 2)
    pd.testing.assert_series_equal(base.iloc[:-1], after.iloc[:-1], check_names=False)


def test_imbalance_l1_denominator_zero_is_nan():
    out = imbalance_l1(pd.Series([0.0, 2.0]), pd.Series([0.0, 3.0]))
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx((2 - 3) / 5)


# Schema + end-to-end


def test_feature_schema_matches_mvp_columns():
    # Default MVP-style config produces the 5.5 feature column names.
    cfg: dict = {
        "data": {"venue": "binance", "symbols": ["BTCUSDT"], "start_date": "2024-01-01", "end_date": "2024-03-31", "top_k": 10},
        "sampling": {"horizons_events": [10, 50, 200], "feature_lookbacks_events": [10, 50, 200]},
        "labels": {},
        "splits": {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "embargo_events": 200},
        "features": {"realised_vol_lookbacks_events": [50, 200]},
        "models": {"run": ["no_change"]},
        "backtest": {"threshold_grid": [0.0]},
    }
    config = ExperimentConfig.model_validate(cfg)
    cols = feature_columns(config)
    for expected in [
        "mid", "spread", "relative_spread", "microprice", "imbalance_l1", "imbalance_lK",
        "ofi_10", "ofi_50", "ofi_200", "return_lag_10", "return_lag_50", "return_lag_200",
        "realised_vol_50", "realised_vol_200", "quality_flags",
    ]:
        assert expected in cols


def test_build_features_end_to_end(tmp_path):
    config = make_config(ingestion={"mode": "synthetic", "synthetic": {"num_days": 2, "rows_per_day": 40, "seed": 7}})
    manifest = ingest_raw_data(config, project_root=tmp_path)
    events = normalise_events(config, manifest, project_root=tmp_path)
    books = build_order_books(config, events, project_root=tmp_path)
    feats = build_features(config, books, project_root=tmp_path)

    assert len(feats) == 1  # one venue/symbol, two dates merged
    part = feats.partitions[0]
    assert part.n_dates == 2
    assert part.row_count == 80  # 2 days * 40 snapshots

    df = pd.read_parquet(tmp_path / part.file_path)
    assert list(df.columns) == feature_columns(config)
    assert df["timestamp_exchange_ns"].is_monotonic_increasing
    # Features become available once enough history accumulates.
    assert df["ofi_2"].notna().any()
    assert df["return_lag_2"].notna().any()


def test_build_features_index_round_trips(tmp_path):
    config = make_config()
    manifest = ingest_raw_data(config, project_root=tmp_path)
    events = normalise_events(config, manifest, project_root=tmp_path)
    books = build_order_books(config, events, project_root=tmp_path)
    build_features(config, books, project_root=tmp_path)

    from lob_forecasting.features import FeatureTableIndex
    idx_path = tmp_path / "data" / "features" / "features_index.yaml"
    assert idx_path.exists()
    assert FeatureTableIndex.load(idx_path).total_rows() == 30
