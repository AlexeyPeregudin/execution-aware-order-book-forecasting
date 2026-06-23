"""12-month walk-forward split tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from lob_forecasting.config import load_config
from lob_forecasting.datasets.monthly_splits import generate_folds

CONFIG = Path("configs/experiment/btcusdt_top5_monthly_real_12m.yaml")


@pytest.fixture(scope="module")
def folds():
    cfg, _ = load_config(str(CONFIG))
    return cfg, generate_folds(cfg)


def test_canonical_config_produces_five_folds(folds):
    _, f = folds
    assert len(f) == 5


def test_first_and_last_fold_dates(folds):
    _, f = folds
    first = f[0]
    assert [d.isoformat() for d in first.train_dates] == [
        "2025-07-01", "2025-08-01", "2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]
    assert [d.isoformat() for d in first.validation_dates] == ["2026-01-01"]
    assert [d.isoformat() for d in first.test_dates] == ["2026-02-01"]

    last = f[-1]
    assert last.train_dates[0].isoformat() == "2025-07-01"
    assert last.train_dates[-1].isoformat() == "2026-04-01"
    assert [d.isoformat() for d in last.validation_dates] == ["2026-05-01"]
    assert [d.isoformat() for d in last.test_dates] == ["2026-06-01"]


def test_no_leakage_within_fold(folds):
    _, f = folds
    for fold in f:
        train = set(fold.train_dates)
        val = set(fold.validation_dates)
        test = set(fold.test_dates)
        assert not (train & val) and not (train & test) and not (val & test)
        # validation precedes test; both are after the whole training window
        assert max(train) < min(val)
        assert max(val) < min(test)


def test_expanding_train_window_grows_by_step(folds):
    cfg, f = folds
    sizes = [len(fold.train_dates) for fold in f]
    assert sizes == [6, 7, 8, 9, 10]  # min_train_months=6, step_months=1


def test_embargo_covers_max_horizon(folds):
    cfg, _ = folds
    assert cfg.splits.embargo_events >= max(cfg.sampling.horizons_events)


def test_thresholds_frozen_on_first_train(folds):
    cfg, f = folds
    # canonical training rows = first fold's training months (earliest data)
    assert cfg.splits.freeze_global_label_thresholds_on_first_train
    assert cfg.splits.freeze_global_regime_thresholds_on_first_train
    assert f[0].train_dates[0].isoformat() == "2025-07-01"
    assert f[0].train_dates[-1].isoformat() == "2025-12-01"
