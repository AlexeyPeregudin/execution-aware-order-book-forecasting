"""Month-level and regime-stratified predictive metrics.

The base evaluation pools metrics over all events. This module instead reports
metrics by test month and by market regime, so weak or unstable signals
can't hide inside one big pooled number. It joins the long prediction table to
the per-event context (monthly_date + regime buckets) carried in the
features-labels table, then groups and scores.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..features.regimes import LIQ_REGIME, SPREAD_REGIME, TIME_BUCKET, VOL_REGIME
from . import metrics as M

CONTEXT_COLS = ("monthly_date", VOL_REGIME, SPREAD_REGIME, LIQ_REGIME, TIME_BUCKET)

MONTHLY_METRIC_COLUMNS = (
    "run_id", "fold_id", "model_name", "split", "monthly_date", "horizon",
    "metric_name", "metric_value", "n_observations", "created_at_utc",
)
REGIME_METRIC_COLUMNS = (
    "run_id", "fold_id", "model_name", "split", "regime_kind", "regime_value",
    "horizon", "metric_name", "metric_value", "n_observations", "created_at_utc",
)
STABILITY_COLUMNS = (
    "run_id", "model_name", "split", "horizon", "metric_name",
    "mean_metric_across_months", "std_metric_across_months",
    "best_month", "best_value", "worst_month", "worst_value",
    "n_months", "n_positive_months", "fraction_positive",
    "n_months_beating_baseline", "frac_months_beating_baseline",
)


# event_id resets per monthly day, so the join key is the (unique) timestamp
_JOIN_KEYS = ["venue", "symbol", "timestamp_exchange_ns"]


def load_context(config: ExperimentConfig, project_root: str | Path | None = None) -> pd.DataFrame:
    """Per-event context (monthly_date, regimes) keyed by venue/symbol/timestamp."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    features_root = root / config.data.processed_dir.parent / "features"
    frames: list[pd.DataFrame] = []
    keep = [*_JOIN_KEYS, *CONTEXT_COLS]
    for path in features_root.glob("venue=*/symbol=*/features_labels.parquet"):
        df = pd.read_parquet(path)
        present = [c for c in keep if c in df.columns]
        frames.append(df[present])
    if not frames:
        return pd.DataFrame(columns=keep)
    return pd.concat(frames, ignore_index=True)


def attach_context(pred: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Left-join the regime/monthly context onto a prediction frame."""
    if context.empty:
        return pred
    return pred.merge(context, on=_JOIN_KEYS, how="left")


def _train_means(frame: pd.DataFrame) -> dict[int, float]:
    """Mean training-period return per horizon (the R2 benchmark)."""
    train = frame[frame["split"] == "train"]
    means: dict[int, float] = {}
    for h, g in train.groupby("horizon"):
        r = g["true_return"]
        means[int(h)] = float(r.mean()) if r.notna().any() else 0.0
    return means


def _group_metrics(g: pd.DataFrame, train_mean: float) -> list[tuple[str, float, int]]:
    """Classification + regression metrics for one grouped subframe."""
    out: list[tuple[str, float, int]] = []
    has_class = g["pred_class"].notna().any()
    has_proba = g["pred_down"].notna().any()
    if has_class or has_proba:
        cm = g[g["true_direction"].notna() & g["prediction_available"]]
        cm = cm[cm["pred_class"].notna() | cm["pred_down"].notna()]
        if len(cm) > 0:
            yt = cm["true_direction"].to_numpy().astype(int)
            if has_class:
                yp = cm["pred_class"].to_numpy().astype(int)
                out.append(("accuracy", M.accuracy(yt, yp), len(cm)))
                out.append(("balanced_accuracy", M.balanced_accuracy(yt, yp), len(cm)))
            if has_proba:
                proba = cm[["pred_down", "pred_neutral", "pred_up"]].to_numpy()
                out.append(("cross_entropy", M.cross_entropy(yt, proba), len(cm)))
                out.append(("brier_score", M.brier_score(yt, proba), len(cm)))
                if has_class:
                    yp = cm["pred_class"].to_numpy().astype(int)
                    out.append(("ece", M.expected_calibration_error(yt, proba, yp), len(cm)))
    if g["pred_return"].notna().any():
        rm = g[g["true_return"].notna() & g["pred_return"].notna()]
        if len(rm) > 0:
            yt = rm["true_return"].to_numpy()
            yp = rm["pred_return"].to_numpy()
            out.append(("mae", M.mae(yt, yp), len(rm)))
            out.append(("rmse", M.rmse(yt, yp), len(rm)))
            out.append(("r2_oos", M.r2_oos(yt, yp, train_mean), len(rm)))
            out.append(("rank_ic", M.rank_ic(yt, yp), len(rm)))
            out.append(("sign_correlation", M.sign_correlation(yt, yp), len(rm)))
    return out


def monthly_and_regime_metrics(
    config: ExperimentConfig,
    predictions: list[pd.DataFrame],
    fold_id: int,
    project_root: str | Path | None = None,
    context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-month and per-regime metric tables for one fold's predictions."""
    created = datetime.now(timezone.utc).isoformat()
    if context is None:
        context = load_context(config, project_root)

    monthly_rows: list[dict] = []
    regime_rows: list[dict] = []

    for frame in predictions:
        if frame.empty:
            continue
        run_id = str(frame["run_id"].iloc[0])
        model = str(frame["model_name"].iloc[0])
        joined = attach_context(frame, context)
        train_means = _train_means(joined)

        # by month
        for (split, monthly_date, horizon), g in joined.groupby(
            ["split", "monthly_date", "horizon"], dropna=False
        ):
            tmean = train_means.get(int(horizon), 0.0)
            for name, value, n in _group_metrics(g, tmean):
                monthly_rows.append({
                    "run_id": run_id, "fold_id": fold_id, "model_name": model,
                    "split": split, "monthly_date": _iso(monthly_date), "horizon": int(horizon),
                    "metric_name": name, "metric_value": float(value),
                    "n_observations": int(n), "created_at_utc": created,
                })

        # by regime (one table, tagged by regime kind)
        for kind in (VOL_REGIME, SPREAD_REGIME, LIQ_REGIME, TIME_BUCKET):
            if kind not in joined.columns:
                continue
            for (split, regime_value, horizon), g in joined.groupby(
                ["split", kind, "horizon"], dropna=True
            ):
                tmean = train_means.get(int(horizon), 0.0)
                for name, value, n in _group_metrics(g, tmean):
                    regime_rows.append({
                        "run_id": run_id, "fold_id": fold_id, "model_name": model,
                        "split": split, "regime_kind": kind, "regime_value": str(regime_value),
                        "horizon": int(horizon), "metric_name": name,
                        "metric_value": float(value), "n_observations": int(n),
                        "created_at_utc": created,
                    })

    monthly_df = pd.DataFrame(monthly_rows, columns=list(MONTHLY_METRIC_COLUMNS))
    regime_df = pd.DataFrame(regime_rows, columns=list(REGIME_METRIC_COLUMNS))
    return monthly_df, regime_df


_HIGHER_IS_WORSE = {"mae", "rmse", "cross_entropy", "brier_score", "ece"}


def month_stability_summary(
    monthly_metrics: pd.DataFrame,
    baseline_accuracy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Across test months: mean/std, best/worst month, count beating zero.

    For the `accuracy` metric, also count how many test months each model beats
    the per-month majority-class baseline, when `baseline_accuracy` (columns
    `monthly_date`/`baseline_accuracy`) is supplied. For every other metric
    the two baseline columns are left as NaN.
    """
    rows: list[dict] = []
    if monthly_metrics.empty:
        return pd.DataFrame(rows, columns=list(STABILITY_COLUMNS))
    test = monthly_metrics[monthly_metrics["split"] == "test"]
    run_id = str(monthly_metrics["run_id"].iloc[0])
    base_by_month: dict[str, float] = {}
    if baseline_accuracy is not None and not baseline_accuracy.empty:
        base_by_month = {
            str(m)[:10]: float(b)
            for m, b in zip(baseline_accuracy["monthly_date"], baseline_accuracy["baseline_accuracy"])
        }
    for (model, horizon, name), g in test.groupby(["model_name", "horizon", "metric_name"]):
        vals = g.dropna(subset=["metric_value"])
        if vals.empty:
            continue
        v = vals["metric_value"].to_numpy()
        months = vals["monthly_date"].to_numpy()
        higher_worse = name in _HIGHER_IS_WORSE
        best_i = int(np.argmin(v)) if higher_worse else int(np.argmax(v))
        worst_i = int(np.argmax(v)) if higher_worse else int(np.argmin(v))
        n_pos = int((v > 0).sum())
        n_beat = float("nan")
        frac_beat = float("nan")
        if name == "accuracy" and base_by_month:
            beats = [
                1 for val, m in zip(v, months)
                if val > base_by_month.get(str(m)[:10], 0.5) + 1e-9
            ]
            n_beat = int(sum(beats))
            frac_beat = float(n_beat / len(v)) if len(v) else 0.0
        rows.append({
            "run_id": run_id, "model_name": model, "split": "test", "horizon": int(horizon),
            "metric_name": name,
            "mean_metric_across_months": float(np.mean(v)),
            "std_metric_across_months": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "best_month": str(months[best_i]), "best_value": float(v[best_i]),
            "worst_month": str(months[worst_i]), "worst_value": float(v[worst_i]),
            "n_months": int(len(v)), "n_positive_months": n_pos,
            "fraction_positive": float(n_pos / len(v)),
            "n_months_beating_baseline": n_beat,
            "frac_months_beating_baseline": frac_beat,
        })
    return pd.DataFrame(rows, columns=list(STABILITY_COLUMNS))


def _iso(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    return str(value)[:10]
