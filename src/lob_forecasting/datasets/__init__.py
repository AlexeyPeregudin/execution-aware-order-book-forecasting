"""Datasets: temporal splits and the train/val/test datasets."""

from .build import build_datasets
from .dataset_schema import (
    DatasetIndex,
    DatasetSplitStats,
    sequence_index_columns,
    tabular_columns,
)
from .monthly_splits import (
    SPLIT_UNUSED,
    MonthlyFold,
    assign_monthly_splits,
    generate_folds,
    monthly_dates,
)
from .scaler import FeatureScaler
from .splits import (
    SPLIT_EMBARGO,
    SPLIT_NAMES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    assign_splits,
    split_boundaries,
    training_mask,
)

__all__ = [
    # Orchestration
    "build_datasets",
    "DatasetIndex",
    "DatasetSplitStats",
    "tabular_columns",
    "sequence_index_columns",
    "FeatureScaler",
    # Splits
    "assign_splits",
    "training_mask",
    "split_boundaries",
    "SPLIT_TRAIN",
    "SPLIT_VALIDATION",
    "SPLIT_TEST",
    "SPLIT_EMBARGO",
    "SPLIT_NAMES",
    # Monthly splits
    "MonthlyFold",
    "generate_folds",
    "assign_monthly_splits",
    "monthly_dates",
    "SPLIT_UNUSED",
]
