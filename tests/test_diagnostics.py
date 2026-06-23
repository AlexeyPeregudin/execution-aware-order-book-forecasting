"""Tests for the diagnostics and reporting module."""

from __future__ import annotations

import pandas as pd
import pytest

from lob_forecasting.backtesting import run_backtest
from lob_forecasting.config import ExperimentConfig
from lob_forecasting.datasets import build_datasets
from lob_forecasting.diagnostics import analyse, build_report, build_tables, load_report_data
from lob_forecasting.evaluation import evaluate_predictions
from lob_forecasting.features import build_features
from lob_forecasting.ingestion import ingest_raw_data
from lob_forecasting.labels import build_labels
from lob_forecasting.normalisation import normalise_events
from lob_forecasting.orderbook import BookTableIndex, build_order_books
from lob_forecasting.training import train_and_predict


def _config() -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "data": {"venue": "binance", "symbols": ["BTCUSDT"], "start_date": "2024-01-01", "end_date": "2024-03-31", "top_k": 2},
        "sampling": {"horizons_events": [1, 2], "feature_lookbacks_events": [1, 2]},
        "labels": {"direction_threshold_alpha": 0.5},
        "splits": {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "embargo_events": 2},
        "features": {"realised_vol_lookbacks_events": [2]},
        "models": {"run": ["no_change", "ridge_regression", "logistic_regression"]},
        "backtest": {"horizon": 2, "threshold_grid": [0.0, 0.00001], "fee_bps": 5.0,
                     "latency_events": 1, "max_position": 1.0, "trade_size": 1.0},
        "ingestion": {"mode": "synthetic", "synthetic": {"num_days": 1, "rows_per_day": 200, "seed": 7}},
        "datasets": {"sequence_length": 5},
        "random_seed": 42,
    })


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("report")
    config = _config()
    run_id = "run_report"
    manifest = ingest_raw_data(config, project_root=root)
    events = normalise_events(config, manifest, project_root=root)
    books = build_order_books(config, events, project_root=root)
    features = build_features(config, books, project_root=root)
    labelled = build_labels(config, features, run_id, project_root=root)
    ds_index = build_datasets(config, labelled, run_id, project_root=root)
    preds = [
        train_and_predict(config, ds_index, name, project_root=root, model_config_dir=None)
        for name in ("no_change", "ridge_regression", "logistic_regression")
    ]
    evaluate_predictions(config, preds, project_root=root, write=True)
    run_backtest(config, preds, books, project_root=root, write=True)
    return config, run_id, root


# End-to-end report build


def test_build_report_writes_markdown(full_run):
    config, run_id, root = full_run
    result = build_report(config, run_id, project_root=root)

    md = root / "reports" / "mvp_results.md"
    assert md.exists()
    assert result.markdown_path == md
    text = md.read_text(encoding="utf-8")
    # Acceptance condition: the reader can understand the study (6.12).
    for heading in ("Research question", "Data used", "Labels predicted",
                    "Models compared", "backtest", "Limitations"):
        assert heading.lower() in text.lower()


def test_report_tables_written(full_run):
    config, run_id, root = full_run
    result = build_report(config, run_id, project_root=root)
    tdir = root / "reports" / "tables"
    for name in ("row_counts", "label_distribution", "split_time_ranges",
                 "predictive_metrics_test", "backtest_metrics_test"):
        assert (tdir / f"{name}.csv").exists(), name
        assert name in result.tables


def test_report_figures_written(full_run):
    config, run_id, root = full_run
    result = build_report(config, run_id, project_root=root)
    fdir = root / "reports" / "figures"
    # Feature distributions always render from features-labels.
    assert (fdir / "spread_distribution.png").exists()
    assert (fdir / "imbalance_distribution.png").exists()
    assert (fdir / "label_distribution.png").exists()
    assert "spread" in result.figures and "labels" in result.figures


def test_report_assets_self_contained(full_run):
    config, run_id, root = full_run
    result = build_report(config, run_id, project_root=root)
    assets = root / "artefacts" / "runs" / run_id / "report_assets"
    assert (assets / "mvp_results.md").exists()
    assert (assets / "tables").is_dir()
    assert (assets / "figures").is_dir()
    assert result.report_assets_dir == assets


# Tables and conclusion


def test_row_counts_table_covers_stages(full_run):
    config, run_id, root = full_run
    rd = load_report_data(config, run_id, project_root=root)
    tables = build_tables(rd)
    stages = set(tables["row_counts"]["stage"])
    assert "raw_rows" in stages
    assert "normalised_events" in stages
    assert "book_snapshots" in stages
    assert any(s.startswith("dataset_") for s in stages)


def test_label_distribution_table(full_run):
    config, run_id, root = full_run
    rd = load_report_data(config, run_id, project_root=root)
    tables = build_tables(rd)
    ld = tables["label_distribution"]
    assert set(["horizon", "available", "up", "neutral", "down"]) <= set(ld.columns)
    # available == up + neutral + down per horizon.
    assert (ld["available"] == ld[["up", "neutral", "down"]].sum(axis=1)).all()


def test_analyse_produces_conclusion(full_run):
    config, run_id, root = full_run
    rd = load_report_data(config, run_id, project_root=root)
    a = analyse(rd)
    assert a.conclusion  # non-empty research statement
    # On synthetic data there is no genuine signal; one of the honest outcomes.
    assert any(phrase in a.conclusion for phrase in (
        "no reliable predictive signal", "does not survive", "survives", "too small"))


# Graceful degradation


def test_report_builds_without_backtest(tmp_path):
    """Report should build even if the backtest stage has not run."""
    config = _config()
    run_id = "run_partial"
    manifest = ingest_raw_data(config, project_root=tmp_path)
    events = normalise_events(config, manifest, project_root=tmp_path)
    books = build_order_books(config, events, project_root=tmp_path)
    features = build_features(config, books, project_root=tmp_path)
    labelled = build_labels(config, features, run_id, project_root=tmp_path)
    ds_index = build_datasets(config, labelled, run_id, project_root=tmp_path)
    preds = [train_and_predict(config, ds_index, "no_change", project_root=tmp_path, model_config_dir=None)]
    evaluate_predictions(config, preds, project_root=tmp_path, write=True)
    # No backtest run.

    result = build_report(config, run_id, project_root=tmp_path)
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "No backtest results" in text
