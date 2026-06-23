"""Tests for the training/prediction orchestration module."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from lob_forecasting.config import ExperimentConfig
from lob_forecasting.datasets import DatasetIndex, build_datasets
from lob_forecasting.features import build_features
from lob_forecasting.ingestion import ingest_raw_data
from lob_forecasting.labels import build_labels
from lob_forecasting.models import PREDICTION_COLUMNS
from lob_forecasting.normalisation import normalise_events
from lob_forecasting.orderbook import build_order_books
from lob_forecasting.training import (
    compute_validation_metric,
    predict_with_saved_model,
    train_and_predict,
)
from lob_forecasting.utils import read_current_run, write_current_run

FAST_OVERRIDES = {
    "lightgbm": {"n_estimators": 20, "min_child_samples": 1},
    "tcn_small": {"epochs": 2, "channels": 8, "batch_size": 32},
}

REQUIRED_LOG_FIELDS = {
    "model_name", "model_version", "run_id", "start_time_utc", "end_time_utc",
    "random_seed", "train_rows", "validation_rows", "test_rows",
    "hyperparameters", "best_validation_metric", "model_path", "prediction_path",
}


def make_config() -> ExperimentConfig:
    cfg: dict = {
        "data": {"venue": "binance", "symbols": ["BTCUSDT"], "start_date": "2024-01-01", "end_date": "2024-03-31", "top_k": 2},
        "sampling": {"horizons_events": [1, 2], "feature_lookbacks_events": [1, 2]},
        "labels": {"direction_threshold_alpha": 0.5},
        "splits": {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "embargo_events": 2},
        "features": {"realised_vol_lookbacks_events": [2]},
        "models": {"run": ["no_change", "ridge_regression", "lightgbm", "tcn_small"]},
        "backtest": {"threshold_grid": [0.0, 0.0001]},
        "ingestion": {"mode": "synthetic", "synthetic": {"num_days": 1, "rows_per_day": 160, "seed": 7}},
        "datasets": {"sequence_length": 5},
        "random_seed": 42,
    }
    return ExperimentConfig.model_validate(cfg)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build datasets once; return (config, dataset_index, project_root)."""
    root = tmp_path_factory.mktemp("training")
    config = make_config()
    run_id = "run_train"
    manifest = ingest_raw_data(config, project_root=root)
    events = normalise_events(config, manifest, project_root=root)
    books = build_order_books(config, events, project_root=root)
    features = build_features(config, books, project_root=root)
    labelled = build_labels(config, features, run_id, project_root=root)
    ds_index = build_datasets(config, labelled, run_id, project_root=root)
    return config, ds_index, root


def _train(config, ds_index, root, model_name):
    return train_and_predict(
        config, ds_index, model_name, project_root=root,
        model_config_dir=None, overrides=FAST_OVERRIDES.get(model_name),
    )




@pytest.mark.parametrize("model_name", ["no_change", "ridge_regression", "lightgbm", "tcn_small"])
def test_train_and_predict_writes_artefacts(built, model_name):
    config, ds_index, root = built
    preds = _train(config, ds_index, root, model_name)

    # Predictions cover all three splits in the common 5.6 schema.
    assert list(preds.columns) == list(PREDICTION_COLUMNS)
    assert set(preds["split"].unique()) == {"train", "validation", "test"}

    run_dir = root / "artefacts" / "runs" / ds_index.run_id
    assert (run_dir / "predictions" / f"{model_name}.parquet").exists()
    assert (run_dir / "models" / model_name / "model.bin").exists()
    assert (run_dir / "logs" / f"{model_name}.jsonl").exists()


@pytest.mark.parametrize("model_name", ["no_change", "ridge_regression", "lightgbm", "tcn_small"])
def test_log_has_required_fields(built, model_name):
    config, ds_index, root = built
    _train(config, ds_index, root, model_name)
    log_path = root / "artefacts" / "runs" / ds_index.run_id / "logs" / f"{model_name}.jsonl"
    record = json.loads(log_path.read_text(encoding="utf-8").strip())

    assert REQUIRED_LOG_FIELDS <= set(record)
    assert record["model_name"] == model_name
    assert record["run_id"] == ds_index.run_id
    assert record["random_seed"] == config.random_seed
    assert isinstance(record["hyperparameters"], dict)
    assert record["prediction_path"].endswith(f"{model_name}.parquet")


def test_persisted_predictions_match_return_value(built):
    config, ds_index, root = built
    preds = _train(config, ds_index, root, "ridge_regression")
    on_disk = pd.read_parquet(
        root / "artefacts" / "runs" / ds_index.run_id / "predictions" / "ridge_regression.parquet"
    )
    pd.testing.assert_frame_equal(preds, on_disk)




def test_predict_with_saved_model_matches_training(built):
    config, ds_index, root = built
    trained = _train(config, ds_index, root, "lightgbm")
    regenerated = predict_with_saved_model(config, ds_index, "lightgbm", project_root=root)
    pd.testing.assert_frame_equal(
        trained.reset_index(drop=True), regenerated.reset_index(drop=True)
    )


def test_predict_with_saved_model_requires_trained_model(built):
    config, ds_index, root = built
    with pytest.raises(FileNotFoundError, match="train it first"):
        predict_with_saved_model(config, ds_index, "logistic_regression", project_root=root)


# Validation metric and run-state pointer


def test_validation_metric_classifier_is_accuracy(built):
    config, ds_index, root = built
    preds = _train(config, ds_index, root, "lightgbm")
    value, name = compute_validation_metric(preds)
    assert name == "accuracy"
    assert 0.0 <= value <= 1.0


def test_validation_metric_regressor_is_r2(built):
    config, ds_index, root = built
    preds = _train(config, ds_index, root, "ridge_regression")
    _, name = compute_validation_metric(preds)
    assert name == "oos_r2"


def test_run_state_round_trip(tmp_path):
    assert read_current_run(tmp_path) is None
    write_current_run("20240101T000000_abc123", tmp_path)
    assert read_current_run(tmp_path) == "20240101T000000_abc123"
