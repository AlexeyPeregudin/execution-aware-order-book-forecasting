"""Distributional / uncertainty evaluation for the quantile heads.

Scores the q05/q50/q95 return-quantile predictions with the pinball loss and
checks whether the 90% interval is empirically calibrated, both pooled and
stratified by month and regime.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..features.regimes import LIQ_REGIME, SPREAD_REGIME, VOL_REGIME
from ..labels.quantiles import interval_coverage, mean_pinball_loss

DISTRIBUTIONAL_COLUMNS = (
    "run_id", "fold_id", "model_name", "split", "group_kind", "group_value",
    "horizon", "metric_name", "metric_value", "n_observations", "created_at_utc",
)

_QUANTILE_COLS = ("pred_q05", "pred_q50", "pred_q95")


def has_quantiles(frame: pd.DataFrame) -> bool:
    return all(c in frame.columns for c in _QUANTILE_COLS) and frame["pred_q05"].notna().any()


def _quantile_metrics(g: pd.DataFrame) -> list[tuple[str, float, int]]:
    m = g[g["true_return"].notna() & g["pred_q50"].notna()]
    if len(m) == 0:
        return []
    r = m["true_return"].to_numpy(dtype="float64")
    q05 = m["pred_q05"].to_numpy(dtype="float64")
    q50 = m["pred_q50"].to_numpy(dtype="float64")
    q95 = m["pred_q95"].to_numpy(dtype="float64")
    width = q95 - q05
    return [
        ("quantile_loss_q05", mean_pinball_loss(r, q05, 0.05), len(m)),
        ("quantile_loss_q50", mean_pinball_loss(r, q50, 0.50), len(m)),
        ("quantile_loss_q95", mean_pinball_loss(r, q95, 0.95), len(m)),
        ("empirical_coverage_90", interval_coverage(r, q05, q95), len(m)),
        ("mean_interval_width", float(np.mean(width)), len(m)),
    ]


def distributional_metrics(
    config: ExperimentConfig,
    predictions: list[pd.DataFrame],
    fold_id: int,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Quantile-loss / coverage metrics, pooled and grouped by month + regime."""
    from .robustness import attach_context

    created = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []

    for frame in predictions:
        if frame.empty or not has_quantiles(frame):
            continue
        run_id = str(frame["run_id"].iloc[0])
        model = str(frame["model_name"].iloc[0])
        joined = attach_context(frame, context) if context is not None else frame

        groupings = [("all", None)]
        if "monthly_date" in joined.columns:
            groupings.append(("monthly_date", "monthly_date"))
        for kind in (VOL_REGIME, SPREAD_REGIME, LIQ_REGIME):
            if kind in joined.columns:
                groupings.append((kind, kind))

        for group_kind, col in groupings:
            keys = ["split", "horizon"] if col is None else ["split", col, "horizon"]
            for key_vals, g in joined.groupby(keys, dropna=(col is not None)):
                if col is None:
                    split, horizon = key_vals
                    group_value = "all"
                else:
                    split, group_value, horizon = key_vals
                for name, value, n in _quantile_metrics(g):
                    rows.append({
                        "run_id": run_id, "fold_id": fold_id, "model_name": model,
                        "split": split, "group_kind": group_kind,
                        "group_value": str(group_value), "horizon": int(horizon),
                        "metric_name": name, "metric_value": float(value),
                        "n_observations": int(n), "created_at_utc": created,
                    })

    return pd.DataFrame(rows, columns=list(DISTRIBUTIONAL_COLUMNS))
