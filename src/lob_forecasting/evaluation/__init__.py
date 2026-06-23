"""Evaluation: prediction files -> metric tables."""

from . import metrics
from .bootstrap import block_bootstrap_ci, bootstrap_from_config
from .distributional import distributional_metrics, has_quantiles
from .evaluate import (
    CALIBRATION_COLUMNS,
    CONFUSION_COLUMNS,
    METRICS_COLUMNS,
    EvaluationError,
    EvaluationResult,
    evaluate_predictions,
)
from .robustness import (
    attach_context,
    load_context,
    month_stability_summary,
    monthly_and_regime_metrics,
)

__all__ = [
    "evaluate_predictions",
    "EvaluationResult",
    "EvaluationError",
    "METRICS_COLUMNS",
    "CONFUSION_COLUMNS",
    "CALIBRATION_COLUMNS",
    "metrics",
    # robustness / distributional / bootstrap
    "monthly_and_regime_metrics",
    "month_stability_summary",
    "load_context",
    "attach_context",
    "distributional_metrics",
    "has_quantiles",
    "block_bootstrap_ci",
    "bootstrap_from_config",
]
