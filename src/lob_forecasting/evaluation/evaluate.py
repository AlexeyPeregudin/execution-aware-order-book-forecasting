"""Turn prediction files into metric tables.

We only look at the prediction frames (which already carry the true values), so
this never needs to know how a model works. It produces three tables: the
metrics, the confusion matrices, and a calibration table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from . import metrics as M

METRICS_COLUMNS = (
    "run_id", "model_name", "split", "venue", "symbol", "horizon",
    "metric_name", "metric_value", "n_observations", "created_at_utc",
)
CONFUSION_COLUMNS = (
    "run_id", "model_name", "split", "venue", "symbol", "horizon",
    "true_class", "pred_class", "count",
)
CALIBRATION_COLUMNS = (
    "run_id", "model_name", "split", "venue", "symbol", "horizon",
    "bin", "mean_confidence", "empirical_accuracy", "count",
)

_GROUP_KEYS = ["split", "venue", "symbol", "horizon"]
_ID_KEYS = ("run_id", "model_name", "split", "venue", "symbol", "horizon")


class EvaluationError(RuntimeError):
    """Something is wrong with the predictions (e.g. no test labels)."""


@dataclass
class EvaluationResult:
    """The three tables, with a helper to write them out."""

    metrics: pd.DataFrame
    confusion: pd.DataFrame
    calibration: pd.DataFrame
    run_id: str = ""

    def test_metrics(self) -> pd.DataFrame:
        return self.metrics[self.metrics["split"] == "test"]

    def save(self, config: ExperimentConfig, project_root: str | Path | None = None) -> Path:
        root = Path(project_root) if project_root is not None else Path.cwd()
        out_dir = root / config.data.artefact_dir / "runs" / self.run_id / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics.to_parquet(out_dir / "predictive_metrics.parquet", engine="pyarrow", index=False)
        self.confusion.to_parquet(out_dir / "confusion_matrices.parquet", engine="pyarrow", index=False)
        self.calibration.to_parquet(out_dir / "calibration.parquet", engine="pyarrow", index=False)
        return out_dir


def _train_means(frame: pd.DataFrame) -> dict[tuple, float]:
    """Mean training return per (venue, symbol, horizon), used as the R2 baseline."""
    train = frame[frame["split"] == "train"]
    means: dict[tuple, float] = {}
    for (venue, symbol, h), g in train.groupby(["venue", "symbol", "horizon"]):
        r = g["true_return"]
        means[(venue, symbol, int(h))] = float(r.mean()) if r.notna().any() else 0.0
    return means


def _calibration_rows(
    ids: dict, y_true: np.ndarray, proba: np.ndarray, y_pred: np.ndarray, n_bins: int
) -> list[dict]:
    """Reliability rows: bin by the model's confidence, then compare to accuracy."""
    confidence = proba.max(axis=1)
    correct = (y_pred == y_true).astype("float64")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    rows: list[dict] = []
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        row = dict(ids)
        row["bin"] = b
        row["mean_confidence"] = float(confidence[sel].mean())
        row["empirical_accuracy"] = float(correct[sel].mean())
        row["count"] = int(sel.sum())
        rows.append(row)
    return rows


def evaluate_predictions(
    config: ExperimentConfig,
    predictions: list[pd.DataFrame],
    project_root: str | Path | None = None,
    write: bool = True,
    n_calibration_bins: int = 10,
) -> EvaluationResult:
    """Compute metrics by model, split, symbol and horizon."""
    if not predictions:
        raise EvaluationError("No prediction frames supplied.")

    created_at = datetime.now(timezone.utc).isoformat()
    run_id = str(predictions[0]["run_id"].iloc[0])
    metric_rows: list[dict] = []
    confusion_rows: list[dict] = []
    calibration_rows: list[dict] = []

    for frame in predictions:
        model_name = str(frame["model_name"].iloc[0])

        # if there are test rows but no test labels, something went wrong upstream
        test = frame[frame["split"] == "test"]
        if len(test) > 0 and not (test["true_return"].notna().any() or test["true_direction"].notna().any()):
            raise EvaluationError(f"Model {model_name!r}: test labels are missing.")

        train_means = _train_means(frame)

        for (split, venue, symbol, horizon), g in frame.groupby(_GROUP_KEYS):
            horizon = int(horizon)
            ids = {
                "run_id": run_id, "model_name": model_name, "split": split,
                "venue": venue, "symbol": symbol, "horizon": horizon,
            }

            def add(name, value, n):
                row = dict(ids)
                row["metric_name"] = name
                row["metric_value"] = float(value)
                row["n_observations"] = int(n)
                row["created_at_utc"] = created_at
                metric_rows.append(row)

            has_class = g["pred_class"].notna().any()
            has_proba = g["pred_down"].notna().any()
            if has_class or has_proba:
                # rows we can score: have a true label and a prediction
                cm = g[g["true_direction"].notna() & g["prediction_available"]]
                cm = cm[cm["pred_class"].notna() | cm["pred_down"].notna()]
                if len(cm) > 0:
                    yt = cm["true_direction"].to_numpy().astype(int)
                    if has_class:
                        yp = cm["pred_class"].to_numpy().astype(int)
                        add("accuracy", M.accuracy(yt, yp), len(cm))
                        add("balanced_accuracy", M.balanced_accuracy(yt, yp), len(cm))
                        mat = M.confusion(yt, yp)
                        for i, tc in enumerate(M.CLASS_LABELS):
                            for j, pc in enumerate(M.CLASS_LABELS):
                                conf_row = {k: ids[k] for k in _ID_KEYS}
                                conf_row["true_class"] = tc
                                conf_row["pred_class"] = pc
                                conf_row["count"] = int(mat[i, j])
                                confusion_rows.append(conf_row)
                    if has_proba:
                        proba = cm[["pred_down", "pred_neutral", "pred_up"]].to_numpy()
                        add("cross_entropy", M.cross_entropy(yt, proba), len(cm))
                        add("brier_score", M.brier_score(yt, proba), len(cm))
                        if has_class:
                            conf_ids = {k: ids[k] for k in _ID_KEYS}
                            yp = cm["pred_class"].to_numpy().astype(int)
                            calibration_rows.extend(
                                _calibration_rows(conf_ids, yt, proba, yp, n_calibration_bins)
                            )

            if g["pred_return"].notna().any():
                rm = g[g["true_return"].notna() & g["pred_return"].notna()]
                if len(rm) > 0:
                    yt = rm["true_return"].to_numpy()
                    yp = rm["pred_return"].to_numpy()
                    add("mae", M.mae(yt, yp), len(rm))
                    add("rmse", M.rmse(yt, yp), len(rm))
                    add("r2_oos", M.r2_oos(yt, yp, train_means.get((venue, symbol, horizon), 0.0)), len(rm))
                    add("rank_ic", M.rank_ic(yt, yp), len(rm))

            # count how many predictions are present vs missing
            add("n_predictions", int(g["prediction_available"].sum()), len(g))
            add("n_missing", int((~g["prediction_available"]).sum()), len(g))

    result = EvaluationResult(
        metrics=pd.DataFrame(metric_rows, columns=list(METRICS_COLUMNS)),
        confusion=pd.DataFrame(confusion_rows, columns=list(CONFUSION_COLUMNS)),
        calibration=pd.DataFrame(calibration_rows, columns=list(CALIBRATION_COLUMNS)),
        run_id=run_id,
    )
    if write:
        result.save(config, project_root)
    return result
