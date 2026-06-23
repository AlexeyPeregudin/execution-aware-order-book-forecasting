"""Feature generation: book tables -> causal features."""

from .build import build_features
from .compute import (
    compute_features,
    imbalance_l1,
    imbalance_lk,
    ofi_event_series,
    realised_vol,
    return_lag,
    rolling_ofi,
)
from .feature_table import (
    FeaturePartition,
    FeatureTableIndex,
    enforce_feature_schema,
    feature_columns,
    feature_dtypes,
    feature_value_columns,
    flag_counts,
)

__all__ = [
    # Orchestration
    "build_features",
    "compute_features",
    # Feature math
    "imbalance_l1",
    "imbalance_lk",
    "ofi_event_series",
    "rolling_ofi",
    "return_lag",
    "realised_vol",
    # Schema
    "feature_columns",
    "feature_value_columns",
    "feature_dtypes",
    "enforce_feature_schema",
    "flag_counts",
    "FeatureTableIndex",
    "FeaturePartition",
]
