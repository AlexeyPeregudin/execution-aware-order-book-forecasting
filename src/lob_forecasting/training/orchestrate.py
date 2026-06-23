"""Train a model and write its predictions, model file, and a log line.

Loads the datasets, fits on train (validation is only for threshold picking),
predicts every split, and saves everything under the run folder.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..datasets.dataset_schema import DatasetIndex
from ..models import build_model, get_model_class, load_model_data

MODEL_FILENAME = "model.bin"
_SPLITS = ("train", "validation", "test")


def _run_dir(config: ExperimentConfig, run_id: str, root: Path, subdir: str = "") -> Path:
    base = root / config.data.artefact_dir / "runs" / run_id
    return base / subdir if subdir else base


def compute_validation_metric(predictions: pd.DataFrame) -> tuple[float, str]:
    """One validation number for the log (higher is better).

    Classifiers report accuracy, regressors report out-of-sample R2. Returns
    (value, name).
    """
    val = predictions[predictions["split"] == "validation"]
    if len(val) == 0:
        return float("nan"), "none"

    if val["pred_class"].notna().any():
        m = val[val["true_direction"].notna() & val["pred_class"].notna()]
        if len(m) == 0:
            return float("nan"), "accuracy"
        return float((m["pred_class"] == m["true_direction"]).mean()), "accuracy"

    if val["pred_return"].notna().any():
        m = val[val["true_return"].notna() & val["pred_return"].notna()]
        if len(m) == 0:
            return float("nan"), "oos_r2"
        y = m["true_return"].to_numpy()
        yhat = m["pred_return"].to_numpy()
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        if ss_tot > 0:
            return 1.0 - ss_res / ss_tot, "oos_r2"
        return float("nan"), "oos_r2"

    return float("nan"), "none"


def _row_counts(data: dict, requires_sequences: bool) -> dict[str, int]:
    counts = {}
    for s in _SPLITS:
        counts[f"{s}_rows"] = data[s].n_sequences if requires_sequences else data[s].n_rows
    return counts


def _write_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def train_and_predict(
    config: ExperimentConfig,
    dataset_index: DatasetIndex,
    model_name: str,
    project_root: str | Path | None = None,
    model_config_dir: str | Path | None = "configs/model",
    overrides: dict | None = None,
    run_subdir: str = "",
    output_name: str | None = None,
    include_latent_context: bool = False,
    resume: bool = False,
) -> pd.DataFrame:
    """Train one model and write its model file, predictions, and log.

    Everything goes under artefacts/runs/{run_id}/{run_subdir}/. `output_name`
    overrides the file/label used for an ablation variant of the same model.
    With `resume=True`, a model whose predictions already exist is skipped
    (loaded from disk) and an interrupted deep model continues from its last
    per-epoch checkpoint. Returns the predictions for all splits.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    run_id = dataset_index.run_id
    run_dir = _run_dir(config, run_id, root, run_subdir)
    label = output_name or model_name
    pred_path = run_dir / "predictions" / f"{label}.parquet"

    # model-level resume: a completed model's predictions are reused as-is
    if resume and pred_path.exists():
        from ..utils.progress import log as _log
        _log(f"  [resume] {label}: predictions exist, skipping training")
        return pd.read_parquet(pred_path)

    model = build_model(model_name, model_config_dir=model_config_dir, overrides=overrides)
    needs_seq = model.requires_sequences
    # epoch-level resume for the deep model: point it at a checkpoint file
    if needs_seq:
        ckpt = run_dir / "models" / label / "train_ckpt.pt"
        model.params["_checkpoint_path"] = str(ckpt)
        model.params["_resume"] = resume

    data = {}
    for s in _SPLITS:
        data[s] = load_model_data(config, dataset_index, s, project_root=root,
                                  with_sequences=needs_seq,
                                  include_latent_context=include_latent_context)

    start = datetime.now(timezone.utc)
    model.fit(data["train"], data["validation"], config)
    pred_frames = [model.predict(data[s], config, run_id) for s in _SPLITS]
    predictions = pd.concat(pred_frames, ignore_index=True)
    if output_name is not None:
        predictions["model_name"] = output_name
    end = datetime.now(timezone.utc)

    # save predictions and the model
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(pred_path, engine="pyarrow", index=False)

    model_path = run_dir / "models" / label / MODEL_FILENAME
    model.save(model_path)

    # write the log line
    metric_value, metric_name = compute_validation_metric(predictions)
    log = {
        "model_name": label,
        "model_version": model.version,
        "run_id": run_id,
        "start_time_utc": start.isoformat(),
        "end_time_utc": end.isoformat(),
        "random_seed": config.random_seed,
        **_row_counts(data, needs_seq),
        "hyperparameters": model.hyperparameters(),
        "best_validation_metric": None if np.isnan(metric_value) else metric_value,
        "validation_metric_name": metric_name,
        "model_path": model_path.relative_to(root).as_posix(),
        "prediction_path": pred_path.relative_to(root).as_posix(),
    }
    _write_log(run_dir / "logs" / f"{label}.jsonl", log)
    return predictions


def predict_with_saved_model(
    config: ExperimentConfig,
    dataset_index: DatasetIndex,
    model_name: str,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load an already-trained model and rewrite its predictions (no refitting)."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    run_id = dataset_index.run_id
    run_dir = _run_dir(config, run_id, root)

    model_path = run_dir / "models" / model_name / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}; train it first.")
    model = get_model_class(model_name).load(model_path)
    needs_seq = model.requires_sequences

    pred_frames = []
    for s in _SPLITS:
        md = load_model_data(config, dataset_index, s, project_root=root, with_sequences=needs_seq)
        pred_frames.append(model.predict(md, config, run_id))
    predictions = pd.concat(pred_frames, ignore_index=True)

    pred_path = run_dir / "predictions" / f"{model_name}.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(pred_path, engine="pyarrow", index=False)
    return predictions
