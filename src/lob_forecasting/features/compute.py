"""The actual feature math.

Everything here is causal: a value at event t only uses rows up to t. We only
ever shift forward in time (shift with a positive lag) or use trailing rolling
windows, never a negative shift. Each function is small so it's easy to check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from .feature_table import enforce_feature_schema, ofi_lookbacks, return_lags, vol_lookbacks


def imbalance_l1(bid_qty_1: pd.Series, ask_qty_1: pd.Series) -> pd.Series:
    """Best-level imbalance: (bid_size - ask_size) / (bid_size + ask_size)."""
    denom = bid_qty_1 + ask_qty_1
    out = (bid_qty_1 - ask_qty_1) / denom
    return out.where(denom > 0)


def imbalance_lk(book_df: pd.DataFrame, k: int) -> pd.Series:
    """Multi-level imbalance, weighting level i by 1/i.

    Missing levels count as 0 (that's what nansum does).
    """
    weights = np.array([1.0 / i for i in range(1, k + 1)])
    bid_cols = [f"bid_qty_{i}" for i in range(1, k + 1)]
    ask_cols = [f"ask_qty_{i}" for i in range(1, k + 1)]
    bid_q = book_df[bid_cols].to_numpy(dtype="float64")
    ask_q = book_df[ask_cols].to_numpy(dtype="float64")
    wb = np.nansum(bid_q * weights, axis=1)
    wa = np.nansum(ask_q * weights, axis=1)
    denom = wb + wa
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, (wb - wa) / denom, np.nan)
    return pd.Series(out, index=book_df.index)


def ofi_event_series(
    bid_px_1: pd.Series,
    bid_qty_1: pd.Series,
    ask_px_1: pd.Series,
    ask_qty_1: pd.Series,
) -> pd.Series:
    """Order-flow imbalance per event (e_i).

    Compares the best bid/ask to the previous row. Roughly: if the bid price
    went up or held, the bid size adds; if it went down or held, the previous
    bid size subtracts; ask is the mirror image. The first row has nothing to
    compare to, so e is 0 there.
    """
    bp = bid_px_1.to_numpy(dtype="float64")
    bq = bid_qty_1.to_numpy(dtype="float64")
    ap = ask_px_1.to_numpy(dtype="float64")
    aq = ask_qty_1.to_numpy(dtype="float64")

    prev_bp = bid_px_1.shift(1).to_numpy(dtype="float64")
    prev_ap = ask_px_1.shift(1).to_numpy(dtype="float64")
    # a NaN previous price makes the comparisons False (so 0). Fill the previous
    # sizes with 0 too, otherwise 0 * NaN would give NaN on the first row.
    prev_bq = np.nan_to_num(bid_qty_1.shift(1).to_numpy(dtype="float64"), nan=0.0)
    prev_aq = np.nan_to_num(ask_qty_1.shift(1).to_numpy(dtype="float64"), nan=0.0)

    with np.errstate(invalid="ignore"):
        ind_bp_ge = (bp >= prev_bp).astype("float64")
        ind_bp_le = (bp <= prev_bp).astype("float64")
        ind_ap_le = (ap <= prev_ap).astype("float64")
        ind_ap_ge = (ap >= prev_ap).astype("float64")

    e = ind_bp_ge * bq - ind_bp_le * prev_bq - ind_ap_le * aq + ind_ap_ge * prev_aq
    if len(e):
        e[0] = 0.0  # first row has no previous row
    return pd.Series(e, index=bid_px_1.index)


def rolling_ofi(e: pd.Series, lookback: int) -> pd.Series:
    """Sum the per-event OFI over the last `lookback` events."""
    return e.rolling(window=lookback, min_periods=lookback).sum()


def return_lag(mid: pd.Series, lookback: int) -> pd.Series:
    """Past return over L events: (mid_t - mid_{t-L}) / mid_{t-L}."""
    prev = mid.shift(lookback)
    return (mid - prev) / prev


def realised_vol(mid: pd.Series, window: int) -> pd.Series:
    """Realised vol: sqrt of the sum of squared 1-step log returns over a window."""
    log_ret = np.log(mid / mid.shift(1))
    return np.sqrt((log_ret**2).rolling(window=window, min_periods=window).sum())


def compute_features(book_df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Build the whole feature table for one symbol's book sequence."""
    df = book_df.sort_values("timestamp_exchange_ns", kind="mergesort").reset_index(drop=True)
    feat = config.features

    mid = df["mid"].astype("float64")
    bp1 = df["bid_px_1"].astype("float64")
    bq1 = df["bid_qty_1"].astype("float64")
    ap1 = df["ask_px_1"].astype("float64")
    aq1 = df["ask_qty_1"].astype("float64")

    out = pd.DataFrame(
        {
            "event_id": df["event_id"].astype("int64"),
            "timestamp_exchange_ns": df["timestamp_exchange_ns"].astype("int64"),
            "venue": df["venue"],
            "symbol": df["symbol"],
            "mid": mid,
            "spread": df["spread"].astype("float64"),
            "relative_spread": df["relative_spread"].astype("float64"),
            "microprice": df["microprice"].astype("float64"),
        }
    )

    if feat.include_basic_microstructure:
        out["imbalance_l1"] = imbalance_l1(bq1, aq1)
    if feat.include_multilevel_imbalance:
        out["imbalance_lK"] = imbalance_lk(df, config.data.top_k)
    if feat.include_best_level_ofi:
        e = ofi_event_series(bp1, bq1, ap1, aq1)
        for L in ofi_lookbacks(config):
            out[f"ofi_{L}"] = rolling_ofi(e, L)
    for L in return_lags(config):
        out[f"return_lag_{L}"] = return_lag(mid, L)
    for W in vol_lookbacks(config):
        out[f"realised_vol_{W}"] = realised_vol(mid, W)
    if feat.include_raw_levels:
        from ..orderbook.book_table import level_columns

        for col in level_columns(config.data.top_k):
            out[col] = df[col].astype("float64")

    # regime descriptors + monthly context (causal; computed on this day's rows)
    if feat.include_regime_features:
        from .regimes import compute_regime_descriptors

        desc = compute_regime_descriptors(df, config)
        for col in desc.columns:
            out[col] = desc[col].to_numpy()

    if config.data.monthly_snapshot.enabled or feat.include_regime_features:
        from .regimes import time_of_day_bucket

        ts = out["timestamp_exchange_ns"]
        day = pd.to_datetime(ts, unit="ns", utc=True).dt.date.astype("string")
        out["monthly_date"] = day
        out["month_index"] = 0  # filled in by build_features (needs global order)
        out["is_first_day_of_month"] = pd.to_datetime(ts, unit="ns", utc=True).dt.day == 1
        out["time_of_day_bucket"] = time_of_day_bucket(ts).astype("string")

    out["quality_flags"] = df["quality_flags"].fillna("").astype("string")
    return enforce_feature_schema(out, config)
