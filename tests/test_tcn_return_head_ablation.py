"""Return-head ablation tests for the multi-task TCN."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from lob_forecasting.datasets import build_datasets, generate_folds
from lob_forecasting.models import build_model, load_model_data, merge_ridge_sidecar
from lob_forecasting.models.variants import return_head_variants

from ._monthly_helpers import monthly_config, run_to_labels


@pytest.fixture(scope="module")
def fold_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("ablation")
    cfg = monthly_config()
    labelled = run_to_labels(cfg, root)
    fold = generate_folds(cfg)[0]
    idx = build_datasets(cfg, labelled, "test_run", root, fold=fold)
    train = load_model_data(cfg, idx, "train", root, with_sequences=True)
    val = load_model_data(cfg, idx, "validation", root, with_sequences=True)
    test = load_model_data(cfg, idx, "test", root, with_sequences=True)
    return cfg, train, val, test


def _build(overrides):
    base = {"epochs": 1, "channels": 8, "num_layers": 2, "two_phase": {"enabled": False}}
    return build_model("tcn_exec_multitask", "configs/model", overrides={**base, **overrides})


def test_zero_weight_removes_return_loss(fold_data):
    cfg, train, _, _ = fold_data
    model = _build({"execution_heads": {"return_head": {"enabled": False, "loss_weight": 0.0,
                                                        "prediction_source": "none"}}})
    model.in_features = len(train.feature_names)
    model.context_dim = train.context_dim
    model.horizons = list(train.horizons)
    model.net = model._build_net()
    targets = model._gather_targets(train)
    idx = np.arange(min(16, train.n_sequences))
    weights = model._loss_weights()
    assert weights["return"] == 0.0
    # the return head's parameters must receive no gradient when its weight is 0
    loss = model._loss_for_batch(train, idx, targets, weights)
    loss.backward()
    grads = [p.grad for p in model.net.return_heads.parameters() if p.grad is not None]
    assert all(float(g.abs().sum()) == 0.0 for g in grads) or not grads


def test_detached_return_gives_zero_encoder_gradient(fold_data):
    cfg, train, _, _ = fold_data
    # only the return head is active, and it is detached from the encoder
    model = _build({"execution_heads": {
        "return_head": {"enabled": True, "loss_weight": 1.0, "detach_from_encoder": True,
                        "prediction_source": "neural_head"},
        "direction_head": {"enabled": False, "loss_weight": 0.0},
        "quantile_head": {"enabled": False, "loss_weight": 0.0},
        "markout_head": {"enabled": False, "loss_weight": 0.0},
        "adverse_head": {"enabled": False, "loss_weight": 0.0},
    }})
    model.in_features = len(train.feature_names)
    model.context_dim = train.context_dim
    model.horizons = list(train.horizons)
    model.net = model._build_net()
    targets = model._gather_targets(train)
    idx = np.arange(min(16, train.n_sequences))
    loss = model._loss_for_batch(train, idx, targets, model._loss_weights())
    loss.backward()
    # encoder (body) parameters must get zero gradient; the return head must not
    enc_grad = sum(float(p.grad.abs().sum()) for p in model.net.body.parameters() if p.grad is not None)
    head_grad = sum(float(p.grad.abs().sum()) for p in model.net.return_heads.parameters() if p.grad is not None)
    assert enc_grad == 0.0
    assert head_grad > 0.0


def test_disabled_return_head_writes_nan_pred_return(fold_data):
    cfg, train, val, test = fold_data
    model = _build({"execution_heads": {"return_head": {"enabled": False, "loss_weight": 0.0,
                                                        "prediction_source": "none"}}})
    model.fit(train, val, cfg)
    preds = model.predict(test, cfg)
    assert preds["pred_return"].isna().all()
    assert (preds["pred_return_source"] == "none").all()
    # the execution heads still produce predictions
    assert preds["pred_q50"].notna().any()
    assert preds["pred_markout_bid"].notna().any()


def test_ridge_sidecar_merges_on_timestamp_not_event_id():
    # two monthly days reuse event_id 0/1 (event_id resets per day); the merge
    # must key on timestamp so the right return lands on the right row.
    keys = dict(venue="binance", symbol="BTCUSDT", split="test")
    neural = pd.DataFrame({
        **{k: [v] * 4 for k, v in keys.items()},
        "event_id": [0, 1, 0, 1],
        "timestamp_exchange_ns": [100, 200, 300, 400],
        "horizon": [10, 10, 10, 10],
        "true_return": [0.1, 0.2, 0.3, 0.4],
        "true_direction": [1, 1, -1, -1],
        "pred_return": [np.nan] * 4,
        "pred_down": [0.1, 0.1, 0.1, 0.1], "pred_neutral": [0.1, 0.1, 0.1, 0.1],
        "pred_up": [0.8, 0.8, 0.8, 0.8], "pred_class": [1, 1, 1, 1],
        "prediction_available": [True] * 4,
        "pred_return_source": ["ridge_sidecar"] * 4,
    })
    ridge = pd.DataFrame({
        **{k: [v] * 4 for k, v in keys.items()},
        "event_id": [0, 1, 0, 1],
        "timestamp_exchange_ns": [400, 300, 200, 100],  # deliberately shuffled
        "horizon": [10, 10, 10, 10],
        "true_return": [0.4, 0.3, 0.2, 0.1], "true_direction": [-1, -1, 1, 1],
        "pred_return": [0.04, 0.03, 0.02, 0.01],
        "pred_down": [0.0] * 4, "pred_neutral": [0.0] * 4, "pred_up": [0.0] * 4,
        "pred_class": [np.nan] * 4, "prediction_available": [True] * 4,
        "pred_return_source": [pd.NA] * 4,
    })
    merged = merge_ridge_sidecar(neural, ridge)
    by_ts = merged.set_index("timestamp_exchange_ns")["pred_return"]
    assert by_ts[100] == pytest.approx(0.01)
    assert by_ts[200] == pytest.approx(0.02)
    assert by_ts[300] == pytest.approx(0.03)
    assert by_ts[400] == pytest.approx(0.04)
    assert (merged["pred_return_source"] == "ridge_sidecar").all()


def test_variant_matrix_wiring():
    variants = return_head_variants()
    names = {v.output_name for v in variants}
    assert names == {"tcn_exec_base", "tcn_exec_ret0", "tcn_exec_ret_detached",
                     "tcn_exec_ret0_ridge_sidecar"}
    by = {v.output_name: v for v in variants}
    assert by["tcn_exec_ret0"].return_source == "none"
    assert by["tcn_exec_ret_detached"].overrides["execution_heads"]["return_head"]["detach_from_encoder"]
    assert by["tcn_exec_ret0_ridge_sidecar"].return_source == "ridge_sidecar"
