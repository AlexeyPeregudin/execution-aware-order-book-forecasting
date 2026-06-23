"""Market-regime descriptors and bucketing.

Three causal descriptors per event -- realised volatility, relative spread and
top-5 depth -- plus a coarse UTC time-of-day bucket. The continuous descriptors
are computed in the feature stage (per monthly day, so nothing crosses a day
boundary). The low/high bucket edges are fitted on training rows only and saved,
so a row's regime depends only on past data and on frozen thresholds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from ..config import ExperimentConfig

# descriptor columns written by the feature stage
REGIME_VOL = "regime_vol"
REGIME_SPREAD = "regime_spread"
REGIME_DEPTH = "regime_depth"
REGIME_DESCRIPTORS: tuple[str, ...] = (REGIME_VOL, REGIME_SPREAD, REGIME_DEPTH)

# categorical regime columns written by the labels stage
VOL_REGIME = "vol_regime"
SPREAD_REGIME = "spread_regime"
LIQ_REGIME = "liq_regime"
TIME_BUCKET = "time_of_day_bucket"
REGIME_CATEGORICALS: tuple[str, ...] = (VOL_REGIME, SPREAD_REGIME, LIQ_REGIME)

VOL_LABELS = ("low", "medium", "high")
SPREAD_LABELS = ("tight", "normal", "wide")
LIQ_LABELS = ("thin", "normal", "deep")

# coarse 4-hour UTC buckets
TIME_BUCKET_LABELS = (
    "00:00-03:59",
    "04:00-07:59",
    "08:00-11:59",
    "12:00-15:59",
    "16:00-19:59",
    "20:00-23:59",
)

_NS_PER_HOUR = 3_600_000_000_000


def time_of_day_bucket(timestamp_ns: pd.Series) -> pd.Series:
    """Map exchange timestamps (ns, UTC) to one of the six 4-hour buckets."""
    ts = timestamp_ns.to_numpy(dtype="int64")
    hour = (ts // _NS_PER_HOUR) % 24
    idx = np.clip(hour // 4, 0, len(TIME_BUCKET_LABELS) - 1)
    labels = np.array(TIME_BUCKET_LABELS)
    return pd.Series(labels[idx], index=timestamp_ns.index)


def _realised_vol(mid: pd.Series, window: int) -> pd.Series:
    log_ret = np.log(mid / mid.shift(1))
    return np.sqrt((log_ret**2).rolling(window=window, min_periods=window).sum())


def compute_regime_descriptors(
    book_df: pd.DataFrame, config: ExperimentConfig
) -> pd.DataFrame:
    """Causal descriptors for one monthly day's book sequence.

    Assumes book_df is already time-sorted and from a single monthly day.
    """
    windows = config.features.regime_windows_events
    k = config.data.top_k

    mid = book_df["mid"].astype("float64")
    vol = _realised_vol(mid, int(windows.get("volatility", 1000)))

    rel_spread = book_df["relative_spread"].astype("float64")

    bid_cols = [f"bid_qty_{i}" for i in range(1, k + 1)]
    ask_cols = [f"ask_qty_{i}" for i in range(1, k + 1)]
    bid_depth = book_df[bid_cols].to_numpy(dtype="float64")
    ask_depth = book_df[ask_cols].to_numpy(dtype="float64")
    depth = np.nansum(bid_depth, axis=1) + np.nansum(ask_depth, axis=1)

    return pd.DataFrame(
        {
            REGIME_VOL: vol.to_numpy(),
            REGIME_SPREAD: rel_spread.to_numpy(),
            REGIME_DEPTH: depth,
        },
        index=book_df.index,
    )


class RegimeThresholds(BaseModel):
    """Fitted low/high bucket edges for the three regime descriptors."""

    quantiles: dict[str, float] = Field(default_factory=lambda: {"low": 0.33, "high": 0.67})
    vol: dict[str, float] = Field(default_factory=dict)
    spread: dict[str, float] = Field(default_factory=dict)
    liquidity: dict[str, float] = Field(default_factory=dict)
    n_train_rows: int = 0

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            yaml.dump(self.model_dump(mode="json"), fh, default_flow_style=False, sort_keys=False)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "RegimeThresholds":
        with Path(path).open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh) or {})


def _edges(values: np.ndarray, low_q: float, high_q: float) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"low": 0.0, "high": 0.0}
    return {
        "low": float(np.quantile(finite, low_q)),
        "high": float(np.quantile(finite, high_q)),
    }


def fit_regime_thresholds(
    descriptors_train: pd.DataFrame, config: ExperimentConfig
) -> RegimeThresholds:
    """Fit low/high edges from the training-row descriptors."""
    q = config.features.regime_quantiles
    low_q, high_q = float(q.get("low", 0.33)), float(q.get("high", 0.67))
    return RegimeThresholds(
        quantiles={"low": low_q, "high": high_q},
        vol=_edges(descriptors_train[REGIME_VOL].to_numpy(dtype="float64"), low_q, high_q),
        spread=_edges(descriptors_train[REGIME_SPREAD].to_numpy(dtype="float64"), low_q, high_q),
        liquidity=_edges(descriptors_train[REGIME_DEPTH].to_numpy(dtype="float64"), low_q, high_q),
        n_train_rows=int(len(descriptors_train)),
    )


def _bucketize(values: np.ndarray, edges: dict[str, float], labels: tuple[str, ...]) -> np.ndarray:
    low, high = edges.get("low", 0.0), edges.get("high", 0.0)
    out = np.full(len(values), labels[1], dtype=object)  # default = middle
    with np.errstate(invalid="ignore"):
        out[values <= low] = labels[0]
        out[values > high] = labels[2]
    # leave NaN descriptors as <NA>
    out[~np.isfinite(values)] = None
    return out


_CONTEXT_VOCAB: tuple[tuple[str, tuple[str, ...]], ...] = (
    (VOL_REGIME, VOL_LABELS),
    (SPREAD_REGIME, SPREAD_LABELS),
    (LIQ_REGIME, LIQ_LABELS),
    (TIME_BUCKET, TIME_BUCKET_LABELS),
)


def context_one_hot_names() -> list[str]:
    """Fixed one-hot column names for the regime/time context vector."""
    names: list[str] = []
    for col, vocab in _CONTEXT_VOCAB:
        names += [f"{col}={v}" for v in vocab]
    return names


def encode_context_one_hot(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """One-hot encode the regime/time columns with a fixed vocabulary.

    Unknown / missing values map to an all-zero block, so the encoding is stable
    and never depends on which categories happen to appear.
    """
    names = context_one_hot_names()
    n = len(frame)
    out = np.zeros((n, len(names)), dtype="float32")
    offset = 0
    for col, vocab in _CONTEXT_VOCAB:
        values = frame[col].astype("string").to_numpy() if col in frame.columns else np.array([None] * n)
        index = {v: j for j, v in enumerate(vocab)}
        for i, val in enumerate(values):
            j = index.get(val)
            if j is not None:
                out[i, offset + j] = 1.0
        offset += len(vocab)
    return out, names


def assign_regime_labels(
    descriptors: pd.DataFrame, thresholds: RegimeThresholds
) -> pd.DataFrame:
    """Bucket the descriptors into the three categorical regime columns."""
    return pd.DataFrame(
        {
            VOL_REGIME: _bucketize(
                descriptors[REGIME_VOL].to_numpy(dtype="float64"), thresholds.vol, VOL_LABELS
            ),
            SPREAD_REGIME: _bucketize(
                descriptors[REGIME_SPREAD].to_numpy(dtype="float64"), thresholds.spread, SPREAD_LABELS
            ),
            LIQ_REGIME: _bucketize(
                descriptors[REGIME_DEPTH].to_numpy(dtype="float64"), thresholds.liquidity, LIQ_LABELS
            ),
        },
        index=descriptors.index,
    )
