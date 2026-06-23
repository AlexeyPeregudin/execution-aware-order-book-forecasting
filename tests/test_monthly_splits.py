"""Tests for monthly snapshot splits and cross-day leakage prevention."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from lob_forecasting.datasets import (
    DatasetIndex,
    build_datasets,
    generate_folds,
    monthly_dates,
)
from lob_forecasting.datasets.monthly_splits import SPLIT_UNUSED, assign_monthly_splits
from lob_forecasting.datasets.splits import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALIDATION

from ._monthly_helpers import (
    MONTHLY_DATES as MONTHLY_DATES_FOR_INVARIANT,
)
from ._monthly_helpers import features_labels_frame, monthly_config, run_to_labels


def test_non_first_day_dates_rejected():
    with pytest.raises(ValidationError):
        monthly_config(data={"venue": "binance", "symbols": ["BTCUSDT"], "top_k": 5,
                             "monthly_snapshot": {"enabled": True, "dates": ["2024-01-15"]}})


def test_non_btcusdt_symbol_rejected():
    with pytest.raises(ValidationError):
        monthly_config(data={"venue": "binance", "symbols": ["ETHUSDT"], "top_k": 5,
                             "monthly_snapshot": {"enabled": True, "dates": MONTHLY_DATES_FOR_INVARIANT}})


def test_top_k_must_be_five():
    with pytest.raises(ValidationError):
        monthly_config(data={"venue": "binance", "symbols": ["BTCUSDT"], "top_k": 10,
                             "monthly_snapshot": {"enabled": True, "dates": MONTHLY_DATES_FOR_INVARIANT}})


def test_market_making_requires_markout_targets():
    with pytest.raises(ValidationError):
        monthly_config(labels={"include_markout_targets": False})


def test_too_few_monthly_dates_rejected():
    with pytest.raises(ValidationError):
        monthly_config(data={"venue": "binance", "symbols": ["BTCUSDT"], "top_k": 5,
                             "monthly_snapshot": {"enabled": True, "dates": ["2024-01-01", "2024-02-01"]}})


def test_monthly_dates_sorted():
    cfg = monthly_config(
        data={"venue": "binance", "symbols": ["BTCUSDT"], "top_k": 5,
              "monthly_snapshot": {"enabled": True,
                                   "dates": ["2024-03-01", "2024-01-01", "2024-02-01"]}},
        splits={"mode": "expanding_monthly_snapshot", "min_train_months": 1,
                "validation_months": 1, "test_months": 1, "embargo_events": 10},
    )
    got = [d.isoformat() for d in monthly_dates(cfg)]
    assert got == ["2024-01-01", "2024-02-01", "2024-03-01"]


def test_expanding_folds_generated_correctly():
    cfg = monthly_config()
    folds = generate_folds(cfg)
    assert len(folds) == 2
    assert [d.isoformat() for d in folds[0].train_dates] == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert [d.isoformat() for d in folds[0].validation_dates] == ["2024-04-01"]
    assert [d.isoformat() for d in folds[0].test_dates] == ["2024-05-01"]
    assert [d.isoformat() for d in folds[1].test_dates] == ["2024-06-01"]


def test_no_train_validation_test_overlap():
    cfg = monthly_config()
    for fold in generate_folds(cfg):
        tr = set(fold.train_dates); va = set(fold.validation_dates); te = set(fold.test_dates)
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)


def test_assign_monthly_splits_labels_rows():
    cfg = monthly_config()
    fold = generate_folds(cfg)[0]
    dates = np.array(["2024-01-01", "2024-04-01", "2024-05-01", "2024-06-01"])
    out = assign_monthly_splits(dates, fold)
    assert list(out) == [SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST, SPLIT_UNUSED]


def test_labels_do_not_cross_monthly_day_boundary(tmp_path):
    cfg = monthly_config()
    run_to_labels(cfg, tmp_path)
    fl = features_labels_frame(cfg, tmp_path)
    max_h = max(cfg.sampling.horizons_events)
    # the last max_h rows of every monthly day must be unavailable for that horizon
    for _, day in fl.groupby("monthly_date"):
        avail = day[f"label_available_h{max_h}"].to_numpy()
        assert not avail[-max_h:].any()
        assert avail[: len(day) - max_h].all()


def test_sequence_windows_do_not_cross_monthly_day_boundary(tmp_path):
    cfg = monthly_config()
    labelled = run_to_labels(cfg, tmp_path)
    fold = generate_folds(cfg)[0]
    idx: DatasetIndex = build_datasets(cfg, labelled, "test_run", tmp_path, fold=fold)
    fl = features_labels_frame(cfg, tmp_path).sort_values("timestamp_exchange_ns").reset_index(drop=True)
    day_of_row = fl["monthly_date"].to_numpy()
    import pandas as pd

    for split in ("train", "validation", "test"):
        seq = pd.read_parquet(tmp_path / idx.sequence_paths[split])
        for r in seq.itertuples(index=False):
            assert day_of_row[r.window_start_index] == day_of_row[r.window_end_index]
