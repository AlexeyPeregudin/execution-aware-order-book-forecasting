"""Tests for the compact execution-aware multi-task TCN."""

from __future__ import annotations

import numpy as np
import pytest

from lob_forecasting.datasets import build_datasets, generate_folds
from lob_forecasting.models import build_model, get_model_class, load_model_data
from lob_forecasting.models.prediction import EXEC_PREDICTION_COLUMNS

from ._monthly_helpers import monthly_config, run_to_labels


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("mt")
    cfg = monthly_config()
    labelled = run_to_labels(cfg, root)
    fold = generate_folds(cfg)[0]
    idx = build_datasets(cfg, labelled, "test_run", root, fold=fold)
    train = load_model_data(cfg, idx, "train", root, with_sequences=True)
    val = load_model_data(cfg, idx, "validation", root, with_sequences=True)
    test = load_model_data(cfg, idx, "test", root, with_sequences=True)
    model = build_model("tcn_exec_multitask", "configs/model",
                        overrides={"epochs": 2, "channels": 8, "num_layers": 3})
    model.fit(train, val, cfg)
    return cfg, root, idx, model, test


def test_fits_on_tiny_fixture(fitted):
    _, _, _, model, _ = fitted
    assert model.net is not None
    assert model.in_features > 0


def test_prediction_schema_contains_required_columns(fitted):
    cfg, _, _, model, test = fitted
    preds = model.predict(test, cfg)
    for col in EXEC_PREDICTION_COLUMNS:
        assert col in preds.columns


def test_quantiles_are_monotone(fitted):
    cfg, _, _, model, test = fitted
    p = model.predict(test, cfg).dropna(subset=["pred_q05", "pred_q50", "pred_q95"])
    assert (p["pred_q05"] <= p["pred_q50"] + 1e-9).all()
    assert (p["pred_q50"] <= p["pred_q95"] + 1e-9).all()


def test_adverse_predictions_nonnegative(fitted):
    cfg, _, _, model, test = fitted
    p = model.predict(test, cfg)
    assert (p["pred_adverse_bid"].dropna() >= -1e-9).all()
    assert (p["pred_adverse_ask"].dropna() >= -1e-9).all()


def test_save_load_reproduces_predictions(fitted, tmp_path):
    cfg, _, _, model, test = fitted
    p0 = model.predict(test, cfg)
    path = tmp_path / "mt.bin"
    model.save(path)
    reloaded = get_model_class("tcn_exec_multitask").load(path)
    p1 = reloaded.predict(test, cfg)
    assert np.allclose(p0["pred_return"].to_numpy(), p1["pred_return"].to_numpy(), equal_nan=True)
    assert np.allclose(p0["pred_q50"].to_numpy(), p1["pred_q50"].to_numpy(), equal_nan=True)


def test_lazy_sequence_batching_not_full_tensor(fitted):
    _, _, _, _, test = fitted
    # windows are stored as per-symbol 2D matrices, not one dense (n, L, F) array
    assert isinstance(test.sequence_matrices, list)
    for mat in test.sequence_matrices:
        assert mat.ndim == 2
    # a single batch builds only that batch's windows
    batch = test.seq_window_batch(np.arange(min(4, test.n_sequences)))
    assert batch.shape[1:] == (test.sequence_length, len(test.feature_names))


def test_loss_masks_unavailable_labels(fitted):
    cfg, _, _, model, test = fitted
    # unavailable rows keep NaN true values; predictions still produced everywhere
    p = model.predict(test, cfg)
    assert p["true_return"].isna().any()  # last-h-of-day rows are unavailable
    assert p["pred_return"].notna().all()
