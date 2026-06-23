"""Label columns, the fitted-threshold file, and the labelled-table index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from ..config import ExperimentConfig
from ..features.feature_table import feature_columns
from ..features.regimes import REGIME_CATEGORICALS


def horizons(config: ExperimentConfig) -> list[int]:
    return list(config.sampling.horizons_events)


def markout_columns(config: ExperimentConfig) -> list[str]:
    """Passive markout / adverse-selection label columns (when enabled)."""
    lab = config.labels
    if not (lab.include_markout_targets or lab.include_adverse_selection_targets):
        return []
    hs = horizons(config)
    cols: list[str] = []
    if lab.include_markout_targets:
        cols += [f"markout_bid_h{h}" for h in hs]
        cols += [f"markout_ask_h{h}" for h in hs]
    if lab.include_adverse_selection_targets:
        cols += [f"adverse_bid_h{h}" for h in hs]
        cols += [f"adverse_ask_h{h}" for h in hs]
    cols += [f"markout_available_h{h}" for h in hs]
    return cols


def label_columns(config: ExperimentConfig) -> list[str]:
    """Label columns in order: r_h*, y_dir_h*, flags, then markout/adverse."""
    hs = horizons(config)
    cols = [f"r_h{h}" for h in hs]
    cols += [f"y_dir_h{h}" for h in hs]
    cols += [f"label_available_h{h}" for h in hs]
    cols += markout_columns(config)
    return cols


def regime_label_columns(config: ExperimentConfig) -> list[str]:
    """Categorical regime columns added by the labels stage (when enabled)."""
    return list(REGIME_CATEGORICALS) if config.features.include_regime_features else []


def label_dtypes(config: ExperimentConfig) -> dict[str, str]:
    dtypes: dict[str, str] = {}
    for h in horizons(config):
        dtypes[f"r_h{h}"] = "float64"
        dtypes[f"y_dir_h{h}"] = "Int64"  # -1/0/1 or <NA>
        dtypes[f"label_available_h{h}"] = "bool"
    for col in markout_columns(config):
        dtypes[col] = "bool" if col.startswith("markout_available_") else "float64"
    return dtypes


def labelled_columns(config: ExperimentConfig) -> list[str]:
    """Full features-labels column order: features, regimes, labels, quality_flags."""
    feat = [c for c in feature_columns(config) if c != "quality_flags"]
    return [*feat, *regime_label_columns(config), *label_columns(config), "quality_flags"]


def enforce_labelled_schema(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Reorder/retype df into the features-labels schema."""
    out = df.copy()
    cols = labelled_columns(config)
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[cols]
    for col, dtype in label_dtypes(config).items():
        out[col] = out[col].astype(dtype)
    for col in regime_label_columns(config):
        out[col] = out[col].astype("string")
    out["quality_flags"] = out["quality_flags"].fillna("").astype("string")
    return out


class LabelThresholds(BaseModel):
    """The fitted direction thresholds, saved to a yaml file."""

    thresholds: dict[str, float]
    source: str
    alpha: float
    median_relative_spread_train: float | None = None
    n_train_rows: int = 0

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            yaml.dump(
                self.model_dump(mode="json"),
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "LabelThresholds":
        with Path(path).open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))


class LabelPartition(BaseModel):
    """One features_labels.parquet file plus its label distribution."""

    venue: str
    symbol: str
    file_path: str
    row_count: int
    # per horizon: {"available": n, "up": n, "neutral": n, "down": n}
    label_distribution: dict[str, dict[str, int]] = Field(default_factory=dict)


class LabelledTableIndex(BaseModel):
    """All features-labels partitions plus the fitted thresholds."""

    partitions: list[LabelPartition] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
    threshold_source: str = ""
    alpha: float = 0.0

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
            "thresholds": self.thresholds,
            "threshold_source": self.threshold_source,
            "alpha": self.alpha,
            "partitions": [p.model_dump(mode="json") for p in self.partitions],
        }
        with out.open("w", encoding="utf-8") as fh:
            yaml.dump(payload, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "LabelledTableIndex":
        p = Path(path)
        if not p.exists():
            return cls()
        with p.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        parts = raw.get("partitions") or []
        return cls(
            partitions=[LabelPartition.model_validate(x) for x in parts],
            thresholds=raw.get("thresholds") or {},
            threshold_source=raw.get("threshold_source", ""),
            alpha=raw.get("alpha", 0.0),
        )
