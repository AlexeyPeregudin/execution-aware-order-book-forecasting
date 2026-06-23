"""Tests for the label generation module."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from lob_forecasting.config import ExperimentConfig
from lob_forecasting.features import build_features
from lob_forecasting.ingestion import ingest_raw_data
from lob_forecasting.labels import (
    LabelThresholds,
    LabelledTableIndex,
    build_labels,
    compute_labels,
    fit_label_thresholds,
    fit_thresholds,
    labelled_columns,
)
from lob_forecasting.normalisation import normalise_events
from lob_forecasting.orderbook import build_order_books


def make_config(**over) -> ExperimentConfig:
    cfg: dict = {
        "data": {"venue": "binance", "symbols": ["BTCUSDT"], "start_date": "2024-01-01", "end_date": "2024-03-31", "top_k": 2},
        "sampling": {"horizons_events": [1, 2], "feature_lookbacks_events": [1, 2]},
        "labels": {"direction_threshold_mode": "train_median_relative_spread", "direction_threshold_alpha": 0.5},
        "splits": {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "embargo_events": 2},
        "features": {"realised_vol_lookbacks_events": [2]},
        "models": {"run": ["no_change"]},
        "backtest": {"threshold_grid": [0.0, 0.0001]},
        "ingestion": {"mode": "synthetic", "synthetic": {"num_days": 1, "rows_per_day": 60, "seed": 7}},
    }
    for k, v in over.items():
        if isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return ExperimentConfig.model_validate(cfg)




def test_labels_match_known_mid_sequence():
    mid = pd.Series([100.0, 101.0, 101.0, 99.0])
    thresholds = {"h1": 0.005}
    labels = compute_labels(mid, [1], thresholds)

    # r_h1[t] = (m_{t+1} - m_t)/m_t
    assert labels["r_h1"].iloc[0] == pytest.approx((101 - 100) / 100)   # +0.01
    assert labels["r_h1"].iloc[1] == pytest.approx(0.0)
    assert labels["r_h1"].iloc[2] == pytest.approx((99 - 101) / 101)    # -0.0198

    # y_dir with deadband eps=0.005: +1, 0, -1, <NA>
    assert labels["y_dir_h1"].iloc[0] == 1
    assert labels["y_dir_h1"].iloc[1] == 0
    assert labels["y_dir_h1"].iloc[2] == -1
    assert pd.isna(labels["y_dir_h1"].iloc[3])




def test_final_h_rows_unavailable():
    mid = pd.Series(np.arange(10, dtype="float64") + 100.0)
    labels = compute_labels(mid, [1, 3], thresholds={"h1": 0.0, "h3": 0.0})

    # Horizon 1: only the last row is unavailable.
    assert bool(labels["label_available_h1"].iloc[-1]) is False
    assert bool(labels["label_available_h1"].iloc[-2]) is True
    assert pd.isna(labels["r_h1"].iloc[-1])

    # Horizon 3: the last 3 rows are unavailable.
    assert (~labels["label_available_h3"].iloc[-3:]).all()
    assert bool(labels["label_available_h3"].iloc[-4]) is True




def test_thresholds_use_training_rows_only():
    config = make_config(
        sampling={"horizons_events": [1], "feature_lookbacks_events": [1]},
        splits={"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "embargo_events": 1},
    )
    n = 10
    # Train rows (first 5 after embargo) have small spread; test rows huge.
    rel = np.array([0.002] * 5 + [0.5] * 3 + [100.0] * 2)
    frame = pd.DataFrame({"relative_spread": rel})

    thresholds, median, n_train = fit_label_thresholds([frame], config)
    # Median of training rows only is 0.002; the huge test spreads are ignored.
    assert median == pytest.approx(0.002)
    assert thresholds["h1"] == pytest.approx(0.5 * 0.002)
    assert n_train == 5


def test_fit_thresholds_is_horizon_independent():
    rel = np.array([0.001, 0.002, 0.003])  # median 0.002
    thresholds, median = fit_thresholds(rel, alpha=0.5, horizon_list=[1, 2, 3])
    assert median == pytest.approx(0.002)
    assert thresholds == {"h1": 0.001, "h2": 0.001, "h3": 0.001}




def test_threshold_artefact_is_written(tmp_path):
    config = make_config()
    run_id = "testrun_001"
    manifest = ingest_raw_data(config, project_root=tmp_path)
    events = normalise_events(config, manifest, project_root=tmp_path)
    books = build_order_books(config, events, project_root=tmp_path)
    features = build_features(config, books, project_root=tmp_path)
    build_labels(config, features, run_id, project_root=tmp_path)

    art_path = tmp_path / "artefacts" / "runs" / run_id / "transforms" / "label_thresholds.yaml"
    assert art_path.exists()
    art = LabelThresholds.load(art_path)
    assert art.source == "train_median_relative_spread"
    assert art.alpha == 0.5
    assert set(art.thresholds) == {"h1", "h2"}
    assert art.n_train_rows > 0


# End-to-end: features-labels table schema and content


def test_build_labels_end_to_end(tmp_path):
    config = make_config()
    run_id = "run_e2e"
    manifest = ingest_raw_data(config, project_root=tmp_path)
    events = normalise_events(config, manifest, project_root=tmp_path)
    books = build_order_books(config, events, project_root=tmp_path)
    features = build_features(config, books, project_root=tmp_path)
    labelled = build_labels(config, features, run_id, project_root=tmp_path)

    assert len(labelled) == 1
    part = labelled.partitions[0]
    df = pd.read_parquet(tmp_path / part.file_path)

    # Schema: 5.5 order, features then labels then quality_flags last.
    assert list(df.columns) == labelled_columns(config)
    assert df.columns[-1] == "quality_flags"

    # Labels present and typed correctly.
    for h in (1, 2):
        assert f"r_h{h}" in df.columns
        assert df[f"y_dir_h{h}"].dropna().isin([-1, 0, 1]).all()
        # Final h rows unavailable.
        assert (~df[f"label_available_h{h}"].to_numpy()[-h:]).all()

    # Distribution recorded.
    assert part.label_distribution["h1"]["available"] == part.row_count - 1


def test_labelled_index_round_trips(tmp_path):
    config = make_config()
    manifest = ingest_raw_data(config, project_root=tmp_path)
    events = normalise_events(config, manifest, project_root=tmp_path)
    books = build_order_books(config, events, project_root=tmp_path)
    features = build_features(config, books, project_root=tmp_path)
    build_labels(config, features, "run_idx", project_root=tmp_path)

    idx_path = tmp_path / "data" / "features" / "labelled_index.yaml"
    assert idx_path.exists()
    reloaded = LabelledTableIndex.load(idx_path)
    assert reloaded.alpha == 0.5
    assert "h1" in reloaded.thresholds


def test_directional_label_deadband():
    # Returns inside +/- eps are neutral (0); outside are signed.
    mid = pd.Series([100.0, 100.05, 100.0, 105.0, 100.0])  # small then large move
    labels = compute_labels(mid, [1], thresholds={"h1": 0.001})
    # r0 = 0.0005 < eps -> neutral
    assert labels["y_dir_h1"].iloc[0] == 0
    # r2 = 0.05 > eps -> up
    assert labels["y_dir_h1"].iloc[2] == 1
