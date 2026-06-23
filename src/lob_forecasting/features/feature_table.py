"""The feature table: which columns it has, their dtypes, and the index.

This only covers the feature columns. The label columns (r_h*, y_dir_h*, ...)
get added later by the labels module, so features and labels stay separate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from ..config import ExperimentConfig
from ..orderbook.book_table import level_columns
from .regimes import REGIME_DESCRIPTORS, TIME_BUCKET

ID_COLUMNS: tuple[str, ...] = ("event_id", "timestamp_exchange_ns", "venue", "symbol")
# these come straight from the book table, we just copy them over
_CARRIED: tuple[str, ...] = ("mid", "spread", "relative_spread", "microprice")

# monthly-context columns (added when monthly mode or regime features are on)
CONTEXT_COLUMNS: tuple[str, ...] = (
    "monthly_date",
    "month_index",
    "is_first_day_of_month",
    TIME_BUCKET,
)


def ofi_lookbacks(config: ExperimentConfig) -> list[int]:
    return list(config.sampling.feature_lookbacks_events)


def return_lags(config: ExperimentConfig) -> list[int]:
    return list(config.sampling.feature_lookbacks_events)


def vol_lookbacks(config: ExperimentConfig) -> list[int]:
    return list(config.features.realised_vol_lookbacks_events)


def feature_value_columns(config: ExperimentConfig) -> list[str]:
    """The feature columns (no ids, no quality_flags), in order."""
    feat = config.features
    cols: list[str] = [*_CARRIED]
    if feat.include_basic_microstructure:
        cols.append("imbalance_l1")
    if feat.include_multilevel_imbalance:
        cols.append("imbalance_lK")
    if feat.include_best_level_ofi:
        cols += [f"ofi_{L}" for L in ofi_lookbacks(config)]
    cols += [f"return_lag_{L}" for L in return_lags(config)]
    cols += [f"realised_vol_{W}" for W in vol_lookbacks(config)]
    if feat.include_raw_levels:
        cols += level_columns(config.data.top_k)
    return cols


def regime_descriptor_columns(config: ExperimentConfig) -> list[str]:
    """Continuous regime descriptors, present when regime features are enabled."""
    return list(REGIME_DESCRIPTORS) if config.features.include_regime_features else []


def context_columns(config: ExperimentConfig) -> list[str]:
    """Monthly-context columns, present in monthly mode or with regime features."""
    if config.data.monthly_snapshot.enabled or config.features.include_regime_features:
        return list(CONTEXT_COLUMNS)
    return []


def feature_columns(config: ExperimentConfig) -> list[str]:
    """All feature-table columns, in order."""
    return [
        *ID_COLUMNS,
        *feature_value_columns(config),
        *regime_descriptor_columns(config),
        *context_columns(config),
        "quality_flags",
    ]


_CONTEXT_DTYPES: dict[str, str] = {
    "monthly_date": "string",
    "month_index": "int64",
    "is_first_day_of_month": "bool",
    TIME_BUCKET: "string",
}


def feature_dtypes(config: ExperimentConfig) -> dict[str, str]:
    dtypes: dict[str, str] = {
        "event_id": "int64",
        "timestamp_exchange_ns": "int64",
        "venue": "string",
        "symbol": "string",
    }
    for col in feature_value_columns(config):
        dtypes[col] = "float64"
    for col in regime_descriptor_columns(config):
        dtypes[col] = "float64"
    for col in context_columns(config):
        dtypes[col] = _CONTEXT_DTYPES[col]
    return dtypes


def enforce_feature_schema(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Reorder/retype df to match the feature schema."""
    out = df.copy()
    cols = feature_columns(config)
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[cols]
    out["quality_flags"] = out["quality_flags"].fillna("").astype("string")
    for col, dtype in feature_dtypes(config).items():
        out[col] = out[col].astype(dtype)
    return out


def flag_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw in df["quality_flags"].fillna("").astype(str):
        if not raw:
            continue
        for flag in raw.split("|"):
            if flag:
                counts[flag] = counts.get(flag, 0) + 1
    return counts


class FeaturePartition(BaseModel):
    """One features.parquet file (one venue/symbol, all dates together)."""

    venue: str
    symbol: str
    date_range: tuple[str, str] | None = None
    file_path: str
    row_count: int
    n_dates: int = 1
    feature_columns: list[str] = Field(default_factory=list)
    flag_counts: dict[str, int] = Field(default_factory=dict)


class FeatureTableIndex(BaseModel):
    """All the feature partitions from a run."""

    partitions: list[FeaturePartition] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.partitions)

    def total_rows(self) -> int:
        return sum(p.row_count for p in self.partitions)

    def paths(self) -> list[str]:
        return [p.file_path for p in self.partitions]

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "partitions": [p.model_dump(mode="json") for p in self.partitions]
        }
        with out.open("w", encoding="utf-8") as fh:
            yaml.dump(payload, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "FeatureTableIndex":
        p = Path(path)
        if not p.exists():
            return cls(partitions=[])
        with p.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        parts = raw.get("partitions") or []
        return cls(partitions=[FeaturePartition.model_validate(x) for x in parts])
