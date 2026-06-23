"""Tests for the evaluation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lob_forecasting.config import ExperimentConfig
from lob_forecasting.datasets import build_datasets
from lob_forecasting.evaluation import EvaluationError, evaluate_predictions
from lob_forecasting.features import build_features
from lob_forecasting.ingestion import ingest_raw_data
from lob_forecasting.labels import build_labels
from lob_forecasting.models import enforce_prediction_schema
from lob_forecasting.normalisation import normalise_events
from lob_forecasting.orderbook import build_order_books
from lob_forecasting.training import train_and_predict


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build a 5.6-schema prediction frame from partial row dicts."""
    defaults = {
        "run_id": "r", "model_name": "m", "model_version": "1.0", "split": "test",
        "venue": "binance", "symbol": "BTCUSDT", "event_id": 0, "timestamp_exchange_ns": 0,
        "horizon": 10, "true_return": np.nan, "true_direction": np.nan,
        "pred_return": np.nan, "pred_down": np.nan, "pred_neutral": np.nan,
        "pred_up": np.nan, "pred_class": np.nan, "prediction_available": True,
    }
    out = []
    for i, r in enumerate(rows):
        row = {**defaults, **r}
        row.setdefault("event_id", i)
        row["event_id"] = r.get("event_id", i)
        out.append(row)
    return enforce_prediction_schema(pd.DataFrame(out))


# A toy model frame with both classification and regression outputs.
TOY_TRAIN = [
    {"split": "train", "horizon": 10, "true_return": 0.0, "true_direction": 0, "pred_return": 0.0},
    {"split": "train", "horizon": 10, "true_return": 0.0, "true_direction": 0, "pred_return": 0.0},
]
TOY_TEST = [
    {"split": "test", "horizon": 10, "true_return": 0.01, "true_direction": 1,
     "pred_return": 0.01, "pred_down": 0.1, "pred_neutral": 0.2, "pred_up": 0.7, "pred_class": 1},
    {"split": "test", "horizon": 10, "true_return": 0.02, "true_direction": 1,
     "pred_return": 0.0, "pred_down": 0.2, "pred_neutral": 0.5, "pred_up": 0.3, "pred_class": 0},
    {"split": "test", "horizon": 10, "true_return": 0.0, "true_direction": 0,
     "pred_return": 0.0, "pred_down": 0.2, "pred_neutral": 0.6, "pred_up": 0.2, "pred_class": 0},
    {"split": "test", "horizon": 10, "true_return": -0.01, "true_direction": -1,
     "pred_return": -0.01, "pred_down": 0.7, "pred_neutral": 0.2, "pred_up": 0.1, "pred_class": -1},
]


def _mval(res, name, split="test", model="m"):
    m = res.metrics
    sel = (m["metric_name"] == name) & (m["split"] == split) & (m["model_name"] == model)
    return float(m.loc[sel, "metric_value"].iloc[0])




def test_known_toy_metrics():
    frame = _frame(TOY_TRAIN + TOY_TEST)
    res = evaluate_predictions(_DummyConfig(), [frame], write=False)

    assert _mval(res, "accuracy") == pytest.approx(0.75)
    assert _mval(res, "balanced_accuracy") == pytest.approx((0.5 + 1 + 1) / 3)
    assert _mval(res, "brier_score") == pytest.approx(0.325)
    assert _mval(res, "cross_entropy") == pytest.approx(0.6070, abs=1e-3)
    assert _mval(res, "mae") == pytest.approx(0.005)
    assert _mval(res, "rmse") == pytest.approx(0.01)
    # r2_oos: train mean 0 -> 1 - 0.0004/0.0006
    assert _mval(res, "r2_oos") == pytest.approx(1 - 0.0004 / 0.0006, abs=1e-6)


def test_confusion_matrix_counts():
    frame = _frame(TOY_TRAIN + TOY_TEST)
    res = evaluate_predictions(_DummyConfig(), [frame], write=False)
    c = res.confusion
    c = c[c["split"] == "test"]

    def cell(t, p):
        return int(c[(c["true_class"] == t) & (c["pred_class"] == p)]["count"].iloc[0])

    assert cell(1, 1) == 1   # one true-up predicted up
    assert cell(1, 0) == 1   # one true-up predicted neutral
    assert cell(0, 0) == 1
    assert cell(-1, -1) == 1




def test_fails_when_test_labels_missing():
    frame = _frame([
        {"split": "test", "horizon": 10, "true_return": np.nan, "true_direction": np.nan, "pred_class": 1,
         "pred_down": 0.1, "pred_neutral": 0.2, "pred_up": 0.7},
    ])
    with pytest.raises(EvaluationError, match="test labels are missing"):
        evaluate_predictions(_DummyConfig(), [frame], write=False)




def test_classification_only_model():
    frame = _frame([
        {"split": "train", "horizon": 10, "true_direction": 0, "pred_class": 0, "pred_down": 0.2, "pred_neutral": 0.6, "pred_up": 0.2},
        {"split": "test", "horizon": 10, "true_direction": 1, "pred_class": 1, "pred_down": 0.1, "pred_neutral": 0.2, "pred_up": 0.7},
    ])
    res = evaluate_predictions(_DummyConfig(), [frame], write=False)
    names = set(res.metrics["metric_name"])
    assert "accuracy" in names
    assert "mae" not in names  # no pred_return -> no regression metrics


def test_regression_only_model():
    frame = _frame([
        {"split": "train", "horizon": 10, "true_return": 0.0, "pred_return": 0.0},
        {"split": "test", "horizon": 10, "true_return": 0.01, "pred_return": 0.008},
    ])
    res = evaluate_predictions(_DummyConfig(), [frame], write=False)
    names = set(res.metrics["metric_name"])
    assert "mae" in names
    assert "accuracy" not in names  # no class predictions -> no classification metrics


# Schema, separation, coverage


def test_metrics_schema_and_split_separation():
    frame = _frame(TOY_TRAIN + TOY_TEST)
    res = evaluate_predictions(_DummyConfig(), [frame], write=False)
    from lob_forecasting.evaluation import METRICS_COLUMNS

    assert list(res.metrics.columns) == list(METRICS_COLUMNS)
    # Validation and test metrics are clearly separated by the split column.
    assert "test" in set(res.metrics["split"])
    assert (res.test_metrics()["split"] == "test").all()


def test_missing_predictions_counted():
    frame = _frame([
        {"split": "test", "horizon": 10, "true_direction": 1, "pred_class": 1, "pred_down": 0.1, "pred_neutral": 0.2, "pred_up": 0.7, "prediction_available": True},
        {"split": "test", "horizon": 10, "true_direction": 1, "prediction_available": False},
    ])
    res = evaluate_predictions(_DummyConfig(), [frame], write=False)
    assert _mval(res, "n_missing") == 1.0
    assert _mval(res, "n_predictions") == 1.0


# End-to-end


def _real_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "data": {"venue": "binance", "symbols": ["BTCUSDT"], "start_date": "2024-01-01", "end_date": "2024-03-31", "top_k": 2},
        "sampling": {"horizons_events": [1, 2], "feature_lookbacks_events": [1, 2]},
        "labels": {"direction_threshold_alpha": 0.5},
        "splits": {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "embargo_events": 2},
        "features": {"realised_vol_lookbacks_events": [2]},
        "models": {"run": ["no_change", "ridge_regression", "logistic_regression"]},
        "backtest": {"threshold_grid": [0.0, 0.0001]},
        "ingestion": {"mode": "synthetic", "synthetic": {"num_days": 1, "rows_per_day": 160, "seed": 7}},
        "datasets": {"sequence_length": 5},
        "random_seed": 42,
    })


def test_evaluate_end_to_end(tmp_path):
    config = _real_config()
    run_id = "run_eval"
    manifest = ingest_raw_data(config, project_root=tmp_path)
    events = normalise_events(config, manifest, project_root=tmp_path)
    books = build_order_books(config, events, project_root=tmp_path)
    features = build_features(config, books, project_root=tmp_path)
    labelled = build_labels(config, features, run_id, project_root=tmp_path)
    ds_index = build_datasets(config, labelled, run_id, project_root=tmp_path)

    preds = [
        train_and_predict(config, ds_index, name, project_root=tmp_path, model_config_dir=None)
        for name in ("no_change", "ridge_regression", "logistic_regression")
    ]
    res = evaluate_predictions(config, preds, project_root=tmp_path, write=True)

    # Metrics file written under the run.
    metrics_path = tmp_path / "artefacts" / "runs" / run_id / "metrics" / "predictive_metrics.parquet"
    assert metrics_path.exists()

    # Ridge (regression) has r2_oos; logistic (classification) has accuracy.
    names_by_model = res.metrics.groupby("model_name")["metric_name"].agg(set)
    assert "r2_oos" in names_by_model["ridge_regression"]
    assert "accuracy" in names_by_model["logistic_regression"]
    # Grouped by model, split, symbol, horizon.
    assert set(res.metrics["horizon"].unique()) == {1, 2}
    assert {"train", "validation", "test"} <= set(res.metrics["split"].unique())


# Minimal config stub for the unit tests (only data.artefact_dir is used).


class _DummyConfig:
    class _Data:
        artefact_dir = "artefacts"

    data = _Data()
