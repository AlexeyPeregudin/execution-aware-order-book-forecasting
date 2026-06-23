"""Shared helpers for the monthly-extension tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lob_forecasting.config import ExperimentConfig
from lob_forecasting.features import build_features
from lob_forecasting.ingestion import RawDataManifest, ingest_raw_data
from lob_forecasting.labels import LabelledTableIndex, build_labels
from lob_forecasting.normalisation import EventTableIndex, normalise_events
from lob_forecasting.orderbook import BookTableIndex, build_order_books

MONTHLY_DATES = [
    "2024-01-01", "2024-02-01", "2024-03-01",
    "2024-04-01", "2024-05-01", "2024-06-01",
]


def monthly_config(**over) -> ExperimentConfig:
    """A small but valid monthly top-5 BTCUSDT config for tests."""
    cfg: dict = {
        "data": {
            "venue": "binance", "symbols": ["BTCUSDT"], "top_k": 5,
            "monthly_snapshot": {"enabled": True, "day_of_month": 1, "dates": MONTHLY_DATES},
        },
        "sampling": {"horizons_events": [5, 10], "feature_lookbacks_events": [5, 10]},
        "labels": {
            "direction_threshold_mode": "train_median_relative_spread",
            "direction_threshold_alpha": 0.5, "quantiles": [0.05, 0.5, 0.95],
            "include_markout_targets": True, "include_adverse_selection_targets": True,
        },
        "splits": {
            "mode": "expanding_monthly_snapshot", "min_train_months": 3,
            "validation_months": 1, "test_months": 1, "embargo_events": 10,
        },
        "features": {
            "include_raw_levels": True, "include_regime_features": True,
            "realised_vol_lookbacks_events": [10, 50],
            "regime_windows_events": {"volatility": 50, "spread": 50, "liquidity": 50},
            "regime_quantiles": {"low": 0.33, "high": 0.67},
        },
        "datasets": {"sequence_length": 20},
        "models": {"run": ["no_change", "tcn_exec_multitask"]},
        "backtest": {"horizon": 10, "threshold_grid": [0.0, 0.0001], "run_market_making": True},
        "market_making": {"enabled": True},
        "ingestion": {"mode": "synthetic", "synthetic": {"num_days": 6, "rows_per_day": 300, "seed": 7}},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return ExperimentConfig.model_validate(cfg)


def run_to_labels(config: ExperimentConfig, root: Path, run_id: str = "test_run") -> LabelledTableIndex:
    """Run ingestion -> features -> labels under a temp project root."""
    ingest_raw_data(config, root)
    manifest = RawDataManifest.load(root / config.ingestion.manifest_path)
    normalise_events(config, manifest, root)
    events = EventTableIndex.load(root / config.data.processed_dir / "events" / "events_index.yaml")
    build_order_books(config, events, root)
    books = BookTableIndex.load(root / config.data.processed_dir / "books" / "books_index.yaml")
    features = build_features(config, books, root)
    return build_labels(config, features, run_id, root)


def features_labels_frame(config: ExperimentConfig, root: Path) -> pd.DataFrame:
    path = root / config.data.processed_dir.parent / "features" / "venue=binance" / "symbol=BTCUSDT" / "features_labels.parquet"
    return pd.read_parquet(path)
