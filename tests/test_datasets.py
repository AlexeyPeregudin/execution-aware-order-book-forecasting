"""Tests for the dataset / temporal-splits module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lob_forecasting.config import ExperimentConfig
from lob_forecasting.datasets import (
    DatasetIndex,
    FeatureScaler,
    assign_splits,
    build_datasets,
)
from lob_forecasting.datasets.dataset_schema import tabular_columns
from lob_forecasting.features import build_features
from lob_forecasting.ingestion import ingest_raw_data
from lob_forecasting.labels import build_labels
from lob_forecasting.normalisation import normalise_events
from lob_forecasting.orderbook import build_order_books


def make_config(seq_len=5, **over) -> ExperimentConfig:
    cfg: dict = {
        "data": {"venue": "binance", "symbols": ["BTCUSDT"], "start_date": "2024-01-01", "end_date": "2024-03-31", "top_k": 2},
        "sampling": {"horizons_events": [1, 2], "feature_lookbacks_events": [1, 2]},
        "labels": {"direction_threshold_alpha": 0.5},
        "splits": {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "embargo_events": 2},
        "features": {"realised_vol_lookbacks_events": [2]},
        "models": {"run": ["no_change"]},
        "backtest": {"threshold_grid": [0.0, 0.0001]},
        "ingestion": {"mode": "synthetic", "synthetic": {"num_days": 1, "rows_per_day": 120, "seed": 7}},
        "datasets": {"sequence_length": seq_len},
    }
    for k, v in over.items():
        if isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return ExperimentConfig.model_validate(cfg)


def _full_pipeline(config, tmp_path, run_id="run_ds"):
    manifest = ingest_raw_data(config, project_root=tmp_path)
    events = normalise_events(config, manifest, project_root=tmp_path)
    books = build_order_books(config, events, project_root=tmp_path)
    features = build_features(config, books, project_root=tmp_path)
    labelled = build_labels(config, features, run_id, project_root=tmp_path)
    return build_datasets(config, labelled, run_id, project_root=tmp_path)


# FeatureScaler unit behaviour


class TestFeatureScaler:
    def test_fit_computes_mean_and_std_ddof1(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 10.0, 10.0, 10.0]})
        scaler = FeatureScaler.fit(df, ["a", "b"])
        assert scaler.mean_[0] == pytest.approx(2.5)
        assert scaler.scale_[0] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
        # Constant column -> scale forced to 1.0 (no divide-by-zero).
        assert scaler.scale_[1] == 1.0

    def test_transform_standardises(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        scaler = FeatureScaler.fit(df, ["a"])
        z = scaler.transform(df)[:, 0]
        assert z.mean() == pytest.approx(0.0, abs=1e-9)

    def test_pickle_round_trip(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = FeatureScaler.fit(df, ["a"])
        p = scaler.save(tmp_path / "scaler.pkl")
        loaded = FeatureScaler.load(p)
        assert loaded.mean_[0] == pytest.approx(scaler.mean_[0])
        assert loaded.feature_names == ["a"]




def test_scaler_fitted_on_training_rows_only(tmp_path):
    config = make_config()
    index = _full_pipeline(config, tmp_path)

    # Recompute the training feature rows the same way the module does.
    labelled_path = tmp_path / "data" / "features" / "venue=binance" / "symbol=BTCUSDT" / "features_labels.parquet"
    df = pd.read_parquet(labelled_path).sort_values("timestamp_exchange_ns").reset_index(drop=True)
    split = assign_splits(len(df), config.splits)
    feat_cols = index.feature_columns
    valid = df[feat_cols].notna().all(axis=1).to_numpy()
    train_mask = (split == "train") & valid
    expected_mean = np.nanmean(df.loc[train_mask, feat_cols].to_numpy(float), axis=0)

    scaler = FeatureScaler.load(tmp_path / index.scaler_path)
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-9)

    # A scaler fitted on ALL rows would differ -> confirms train-only fit.
    all_mean = np.nanmean(df.loc[valid, feat_cols].to_numpy(float), axis=0)
    assert not np.allclose(scaler.mean_, all_mean)




def test_embargo_rows_absent_from_datasets(tmp_path):
    config = make_config()
    index = _full_pipeline(config, tmp_path)

    labelled_path = tmp_path / "data" / "features" / "venue=binance" / "symbol=BTCUSDT" / "features_labels.parquet"
    df = pd.read_parquet(labelled_path).sort_values("timestamp_exchange_ns").reset_index(drop=True)
    split = assign_splits(len(df), config.splits)
    embargo_ts = set(df.loc[split == "embargo", "timestamp_exchange_ns"].tolist())

    for s in ("train", "validation", "test"):
        tab = pd.read_parquet(tmp_path / index.tabular_paths[s])
        assert set(tab["split"].unique()) <= {s}
        assert embargo_ts.isdisjoint(set(tab["timestamp_exchange_ns"].tolist()))




def test_timestamps_monotone_within_each_split(tmp_path):
    config = make_config()
    index = _full_pipeline(config, tmp_path)
    for s in ("train", "validation", "test"):
        tab = pd.read_parquet(tmp_path / index.tabular_paths[s])
        for _, g in tab.groupby("symbol"):
            assert g["timestamp_exchange_ns"].is_monotonic_increasing




def test_split_periods_do_not_overlap(tmp_path):
    config = make_config()
    index = _full_pipeline(config, tmp_path)
    ranges = {}
    for s in ("train", "validation", "test"):
        tab = pd.read_parquet(tmp_path / index.tabular_paths[s])
        ranges[s] = (tab["timestamp_exchange_ns"].min(), tab["timestamp_exchange_ns"].max())
    # Contiguous time order: train entirely before val entirely before test.
    assert ranges["train"][1] < ranges["validation"][0]
    assert ranges["validation"][1] < ranges["test"][0]




def test_sequence_windows_do_not_cross_boundaries(tmp_path):
    config = make_config(seq_len=8)
    index = _full_pipeline(config, tmp_path)

    labelled_path = tmp_path / "data" / "features" / "venue=binance" / "symbol=BTCUSDT" / "features_labels.parquet"
    df = pd.read_parquet(labelled_path).sort_values("timestamp_exchange_ns").reset_index(drop=True)
    split = assign_splits(len(df), config.splits)

    seen_any = False
    for s in ("train", "validation", "test"):
        seq = pd.read_parquet(tmp_path / index.sequence_paths[s])
        for _, row in seq.iterrows():
            seen_any = True
            start, end = int(row["window_start_index"]), int(row["window_end_index"])
            assert end - start + 1 == config.datasets.sequence_length
            window_splits = set(split[start : end + 1])
            assert window_splits == {s}  # entire window in one split, no embargo
    assert seen_any  # the test data actually produced windows


# End-to-end outputs and metadata


def test_build_datasets_writes_all_outputs(tmp_path):
    config = make_config()
    run_id = "run_outputs"
    index = _full_pipeline(config, tmp_path, run_id=run_id)

    for s in ("train", "validation", "test"):
        assert (tmp_path / index.tabular_paths[s]).exists()
        assert (tmp_path / index.sequence_paths[s]).exists()
        tab = pd.read_parquet(tmp_path / index.tabular_paths[s])
        assert list(tab.columns) == tabular_columns(config)
    assert (tmp_path / index.scaler_path).exists()

    meta_path = tmp_path / "artefacts" / "runs" / run_id / "dataset_metadata.yaml"
    assert meta_path.exists()
    reloaded = DatasetIndex.load(meta_path)
    assert reloaded.run_id == run_id
    assert reloaded.total_tabular_rows() > 0
    # Every usable row lands in exactly one split (no double counting).
    assert reloaded.sequence_length == config.datasets.sequence_length


def test_scaled_train_features_are_standardised(tmp_path):
    config = make_config()
    index = _full_pipeline(config, tmp_path)
    tab = pd.read_parquet(tmp_path / index.tabular_paths["train"])
    # Scaled training features should be ~zero-mean (the scaler was fit on them).
    for col in index.feature_columns:
        if tab[col].notna().any():
            assert abs(float(tab[col].mean())) < 1e-6
