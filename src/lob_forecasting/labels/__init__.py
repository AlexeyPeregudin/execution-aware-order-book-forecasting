"""Labels: forward returns and direction labels per horizon."""

from .build import build_labels, fit_label_thresholds
from .compute import compute_labels, fit_thresholds, label_distribution
from .label_schema import (
    LabelledTableIndex,
    LabelPartition,
    LabelThresholds,
    enforce_labelled_schema,
    horizons,
    label_columns,
    label_dtypes,
    labelled_columns,
)

__all__ = [
    # Orchestration
    "build_labels",
    "fit_label_thresholds",
    # Label math
    "compute_labels",
    "fit_thresholds",
    "label_distribution",
    # Schema / artefacts
    "label_columns",
    "labelled_columns",
    "label_dtypes",
    "enforce_labelled_schema",
    "horizons",
    "LabelThresholds",
    "LabelledTableIndex",
    "LabelPartition",
]
