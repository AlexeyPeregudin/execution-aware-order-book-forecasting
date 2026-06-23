"""Tests for the distributional (quantile) evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lob_forecasting.evaluation import distributional_metrics, has_quantiles

from ._monthly_helpers import monthly_config


def _pred_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 1.0, n)
    return pd.DataFrame({
        "run_id": "r", "model_name": "tcn_exec_multitask", "model_version": "1.0",
        "split": "test", "venue": "binance", "symbol": "BTCUSDT",
        "event_id": np.arange(n), "timestamp_exchange_ns": np.arange(n),
        "horizon": 50, "true_return": r, "true_direction": np.sign(r),
        "pred_return": 0.0, "pred_q05": -1.645, "pred_q50": 0.0, "pred_q95": 1.645,
        "prediction_available": True,
    })


def test_has_quantiles_detects_columns():
    assert has_quantiles(_pred_frame())
    assert not has_quantiles(_pred_frame().drop(columns=["pred_q05"]))


def test_coverage_close_to_nominal_for_standard_normal():
    cfg = monthly_config()
    df = _pred_frame(n=5000)
    out = distributional_metrics(cfg, [df], fold_id=0, context=None)
    cov = out[(out["metric_name"] == "empirical_coverage_90") & (out["group_kind"] == "all")]
    # a standard-normal target inside [-1.645, 1.645] is covered ~90% of the time
    assert cov["metric_value"].iloc[0] == pytest.approx(0.90, abs=0.03)


def test_metric_names_present():
    cfg = monthly_config()
    out = distributional_metrics(cfg, [_pred_frame()], fold_id=0, context=None)
    names = set(out["metric_name"])
    assert {"quantile_loss_q05", "quantile_loss_q50", "quantile_loss_q95",
            "empirical_coverage_90", "mean_interval_width"} <= names


def test_no_quantiles_returns_empty():
    cfg = monthly_config()
    df = _pred_frame().drop(columns=["pred_q05", "pred_q50", "pred_q95"])
    out = distributional_metrics(cfg, [df], fold_id=0, context=None)
    assert out.empty
