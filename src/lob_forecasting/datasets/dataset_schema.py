"""Dataset column lists and the dataset metadata that build_datasets returns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..config import ExperimentConfig
from ..features.feature_table import feature_value_columns
from ..features.regimes import LIQ_REGIME, SPREAD_REGIME, TIME_BUCKET, VOL_REGIME
from ..labels.label_schema import label_columns

SPLITS: tuple[str, ...] = ("train", "validation", "test")


def context_feature_columns(config: ExperimentConfig) -> list[str]:
    """Categorical regime/time columns carried per window for the context branch."""
    if not config.features.include_regime_features:
        return []
    return [VOL_REGIME, SPREAD_REGIME, LIQ_REGIME, TIME_BUCKET]

# columns that identify each row
_ID_COLUMNS: tuple[str, ...] = ("venue", "symbol", "event_id", "timestamp_exchange_ns")


def tabular_columns(config: ExperimentConfig) -> list[str]:
    """Columns of a tabular_{split}.parquet: ids, split, scaled features, labels."""
    return [
        *_ID_COLUMNS,
        "split",
        *feature_value_columns(config),
        *label_columns(config),
        "quality_flags",
    ]


def sequence_index_columns(config: ExperimentConfig) -> list[str]:
    """Columns of a sequence_index_{split}.parquet (one row per TCN window)."""
    return [
        "venue",
        "symbol",
        "split",
        "end_event_id",
        "end_timestamp_exchange_ns",
        "window_start_index",
        "window_end_index",
        "seq_len",
        *label_columns(config),
        *context_feature_columns(config),
    ]


class DatasetSplitStats(BaseModel):
    split: str
    n_tabular: int = 0
    n_sequence: int = 0
    time_min: int | None = None
    time_max: int | None = None


class DatasetIndex(BaseModel):
    """Everything build_datasets produced. Also the dataset_metadata.yaml file."""

    run_id: str
    fold_id: int | None = None
    fold_name: str | None = None
    symbols: list[str] = Field(default_factory=list)
    feature_columns: list[str] = Field(default_factory=list)
    label_columns: list[str] = Field(default_factory=list)
    sequence_length: int = 100
    split_fractions: dict[str, float] = Field(default_factory=dict)
    embargo_events: int = 0
    tabular_paths: dict[str, str] = Field(default_factory=dict)
    sequence_paths: dict[str, str] = Field(default_factory=dict)
    scaler_path: str = ""
    # latent state-space context; empty when disabled
    latent_context_path: str = ""
    latent_transform_path: str = ""
    latent_state_dim: int = 0
    stats: list[DatasetSplitStats] = Field(default_factory=list)
    created_at_utc: str = ""

    def stat(self, split: str) -> DatasetSplitStats | None:
        for s in self.stats:
            if s.split == split:
                return s
        return None

    def total_tabular_rows(self) -> int:
        return sum(s.n_tabular for s in self.stats)

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
    def load(cls, path: str | Path) -> "DatasetIndex":
        with Path(path).open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh) or {})
