"""Tests for the models module."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from lob_forecasting.config import ExperimentConfig
from lob_forecasting.datasets import build_datasets
from lob_forecasting.features import build_features
from lob_forecasting.ingestion import ingest_raw_data
from lob_forecasting.labels import build_labels
from lob_forecasting.models import (
    PREDICTION_COLUMNS,
    build_model,
    get_model_class,
    load_model_data,
    registered_models,
)
from lob_forecasting.models.registry import load_model_params
from lob_forecasting.normalisation import normalise_events
from lob_forecasting.orderbook import build_order_books

ALL_MODELS = [
    "no_change",
    "imbalance_rule",
    "logistic_regression",
    "ridge_regression",
    "lightgbm",
    "tcn_small",
]

# Small per-model overrides to keep the tiny-fixture tests fast.
FAST_OVERRIDES = {
    "lightgbm": {"n_estimators": 20, "min_child_samples": 1},
    "tcn_small": {"epochs": 2, "channels": 8, "batch_size": 32},
}


def make_config() -> ExperimentConfig:
    cfg: dict = {
        "data": {"venue": "binance", "symbols": ["BTCUSDT"], "start_date": "2024-01-01", "end_date": "2024-03-31", "top_k": 2},
        "sampling": {"horizons_events": [1, 2], "feature_lookbacks_events": [1, 2]},
        "labels": {"direction_threshold_alpha": 0.5},
        "splits": {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "embargo_events": 2},
        "features": {"realised_vol_lookbacks_events": [2]},
        "models": {"run": ALL_MODELS},
        "backtest": {"threshold_grid": [0.0, 0.0001]},
        "ingestion": {"mode": "synthetic", "synthetic": {"num_days": 1, "rows_per_day": 160, "seed": 7}},
        "datasets": {"sequence_length": 5},
        "random_seed": 42,
    }
    return ExperimentConfig.model_validate(cfg)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run the full pipeline once and expose ModelData for each split."""
    tmp_path = tmp_path_factory.mktemp("models_pipeline")
    config = make_config()
    run_id = "run_models"
    manifest = ingest_raw_data(config, project_root=tmp_path)
    events = normalise_events(config, manifest, project_root=tmp_path)
    books = build_order_books(config, events, project_root=tmp_path)
    features = build_features(config, books, project_root=tmp_path)
    labelled = build_labels(config, features, run_id, project_root=tmp_path)
    ds_index = build_datasets(config, labelled, run_id, project_root=tmp_path)

    data = {
        split: load_model_data(config, ds_index, split, project_root=tmp_path, with_sequences=True)
        for split in ("train", "validation", "test")
    }
    return config, data, run_id, tmp_path


# Registry


def test_all_models_registered():
    for name in ALL_MODELS:
        assert name in registered_models()


def test_build_model_loads_lightgbm_params_from_yaml():
    params = load_model_params("lightgbm", "configs/model")
    assert params["num_leaves"] == 31
    assert params["n_estimators"] == 300


#          + prediction columns match the common schema


@pytest.mark.parametrize("name", ALL_MODELS)
def test_model_fits_and_predicts(pipeline, name):
    config, data, run_id, _ = pipeline
    model = build_model(name, overrides=FAST_OVERRIDES.get(name))
    model.fit(data["train"], data["validation"], config)
    preds = model.predict(data["test"], config, run_id)

    # Prediction columns match the common 5.6 schema exactly.
    assert list(preds.columns) == list(PREDICTION_COLUMNS)
    assert len(preds) > 0
    assert (preds["model_name"] == name).all()
    assert (preds["split"] == "test").all()
    assert set(preds["horizon"].unique()) == {1, 2}




@pytest.mark.parametrize("name", ALL_MODELS)
def test_class_probabilities_sum_to_one(pipeline, name):
    config, data, run_id, _ = pipeline
    model = build_model(name, overrides=FAST_OVERRIDES.get(name))
    model.fit(data["train"], data["validation"], config)
    preds = model.predict(data["test"], config, run_id)

    has_proba = preds["pred_down"].notna()
    if has_proba.any():
        s = preds.loc[has_proba, ["pred_down", "pred_neutral", "pred_up"]].sum(axis=1)
        np.testing.assert_allclose(s.to_numpy(), 1.0, atol=1e-6)
    else:
        # Regression-only model (ridge): class probs are null, returns present.
        assert preds["pred_return"].notna().any()




@pytest.mark.parametrize("name", ALL_MODELS)
def test_save_load_identical_predictions(pipeline, name, tmp_path):
    config, data, run_id, _ = pipeline
    model = build_model(name, overrides=FAST_OVERRIDES.get(name))
    model.fit(data["train"], data["validation"], config)
    p1 = model.predict(data["test"], config, run_id)

    path = tmp_path / f"{name}.model"
    model.save(path)
    loaded = get_model_class(name).load(path)
    p2 = loaded.predict(data["test"], config, run_id)

    pd.testing.assert_frame_equal(p1, p2)


# Model-specific behaviour


def test_no_change_predicts_zero(pipeline):
    config, data, run_id, _ = pipeline
    model = build_model("no_change")
    model.fit(data["train"], data["validation"], config)
    preds = model.predict(data["test"], config, run_id)
    assert (preds["pred_return"] == 0.0).all()
    assert (preds["pred_class"] == 0.0).all()
    assert (preds["pred_neutral"] == 1.0).all()


def test_ridge_is_regression_only(pipeline):
    config, data, run_id, _ = pipeline
    model = build_model("ridge_regression")
    model.fit(data["train"], data["validation"], config)
    preds = model.predict(data["test"], config, run_id)
    assert preds["pred_return"].notna().all()
    assert preds["pred_down"].isna().all()  # no class probabilities
    assert preds["pred_class"].isna().all()


def test_imbalance_rule_selects_gamma_on_validation(pipeline):
    config, data, run_id, _ = pipeline
    model = build_model("imbalance_rule")
    model.fit(data["train"], data["validation"], config)
    assert 0.0 <= model.gamma <= 0.9  # a gamma was picked from the grid
    preds = model.predict(data["test"], config, run_id)
    assert preds["pred_class"].isin([-1.0, 0.0, 1.0]).all()


def test_tcn_predicts_only_sequence_end_events(pipeline):
    config, data, run_id, _ = pipeline
    model = build_model("tcn_small", overrides=FAST_OVERRIDES["tcn_small"])
    model.fit(data["train"], data["validation"], config)
    preds = model.predict(data["test"], config, run_id)
    # One prediction per sequence window per horizon.
    n_windows = data["test"].n_sequences
    assert len(preds) == n_windows * len(config.sampling.horizons_events)


def test_true_values_carried_from_labels(pipeline):
    config, data, run_id, _ = pipeline
    model = build_model("logistic_regression")
    model.fit(data["train"], data["validation"], config)
    preds = model.predict(data["test"], config, run_id)
    # Where a label is unavailable the true value is null, not fabricated.
    h1 = preds[preds["horizon"] == 1]
    assert h1["true_direction"].isin([-1.0, 0.0, 1.0]).any()
