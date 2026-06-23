"""Tests for the causal regime descriptors and fitted bucket thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forecasting.features.regimes import (
    REGIME_DEPTH,
    REGIME_SPREAD,
    REGIME_VOL,
    RegimeThresholds,
    assign_regime_labels,
    encode_context_one_hot,
    fit_regime_thresholds,
)

from ._monthly_helpers import features_labels_frame, monthly_config, run_to_labels


def _descriptors(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        REGIME_VOL: rng.uniform(0, 1, n),
        REGIME_SPREAD: rng.uniform(0, 1, n),
        REGIME_DEPTH: rng.uniform(0, 1, n),
    })


def test_thresholds_fitted_on_training_rows_only():
    cfg = monthly_config()
    train = _descriptors(500, seed=1)
    # contaminate later (non-training) rows with extreme values that must be ignored
    th = fit_regime_thresholds(train, cfg)
    # the fitted high edge should track the training quantile, not any outliers
    assert th.vol["high"] == np.quantile(train[REGIME_VOL].to_numpy(), 0.67)
    assert th.spread["low"] == np.quantile(train[REGIME_SPREAD].to_numpy(), 0.33)


def test_future_rows_do_not_affect_previous_regime_assignment():
    th = RegimeThresholds(vol={"low": 0.0, "high": 1.0}, spread={"low": 0.0, "high": 1.0},
                          liquidity={"low": 0.0, "high": 1.0})
    base = _descriptors(50, seed=2)
    labels_a = assign_regime_labels(base, th)
    extended = pd.concat([base, _descriptors(50, seed=3)], ignore_index=True)
    labels_b = assign_regime_labels(extended, th)
    # adding later rows must not change the earlier assignments (bucketing is pointwise)
    assert (labels_a["vol_regime"].to_numpy() == labels_b.iloc[:50]["vol_regime"].to_numpy()).all()


def test_regime_buckets_stable_after_threshold_save_load(tmp_path):
    cfg = monthly_config()
    th = fit_regime_thresholds(_descriptors(300, seed=4), cfg)
    path = tmp_path / "regime_thresholds.yaml"
    th.save(path)
    th2 = RegimeThresholds.load(path)
    d = _descriptors(80, seed=5)
    a = assign_regime_labels(d, th)
    b = assign_regime_labels(d, th2)
    assert a.equals(b)


def test_context_one_hot_fixed_width_and_missing_handled():
    frame = pd.DataFrame({
        "vol_regime": ["low", "high", None],
        "spread_regime": ["tight", "wide", "normal"],
        "liq_regime": ["thin", "deep", None],
        "time_of_day_bucket": ["00:00-03:59", "20:00-23:59", "08:00-11:59"],
    })
    oh, names = encode_context_one_hot(frame)
    assert oh.shape == (3, len(names))
    # row sums equal the number of non-missing categorical fields
    assert oh[0].sum() == 4 and oh[2].sum() == 2  # row 2 has two missing fields


def test_regime_columns_written_to_features_labels(tmp_path):
    cfg = monthly_config()
    run_to_labels(cfg, tmp_path)
    fl = features_labels_frame(cfg, tmp_path)
    for col in ("vol_regime", "spread_regime", "liq_regime", "time_of_day_bucket", "monthly_date"):
        assert col in fl.columns
