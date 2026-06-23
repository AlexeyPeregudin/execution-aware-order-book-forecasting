"""Composite validation-score tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forecasting.models.deep.selection import (
    HorizonScales,
    composite_score,
    fit_scales,
    horizon_components,
)


def _preds_for_coverage(cover_lo: float, cover_hi: float, n: int = 1000) -> pd.DataFrame:
    """Build a one-horizon validation frame whose [q05,q95] covers a given fraction."""
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, n)
    half = np.quantile(np.abs(r), (cover_lo + cover_hi))  # width controlling coverage
    return pd.DataFrame({
        "horizon": 10, "true_return": r, "pred_q05": -half, "pred_q50": 0.0, "pred_q95": half,
        "pred_class": np.sign(r).astype(int), "true_direction": np.sign(r).astype(int),
        "true_markout_bid": r, "pred_markout_bid": r, "true_markout_ask": r, "pred_markout_ask": r,
        "true_adverse_bid": np.abs(r), "pred_adverse_bid": np.abs(r),
        "true_adverse_ask": np.abs(r), "pred_adverse_ask": np.abs(r),
    })


def test_coverage_error_penalises_under_and_over_coverage():
    scales = HorizonScales(ret=1.0, markout=1.0, adverse=1.0)
    # narrow intervals -> under-coverage; wide -> over-coverage; both raise CErr
    narrow = horizon_components(_preds_for_coverage(0.1, 0.1), scales)
    perfect = horizon_components(_preds_for_coverage(0.45, 0.45), scales)
    wide = horizon_components(_preds_for_coverage(0.49, 0.49), scales)
    assert narrow["coverage_error"] > perfect["coverage_error"]
    assert wide["coverage_error"] >= perfect["coverage_error"] - 1e-6
    assert perfect["coverage_error"] < 0.2


def test_disabled_head_terms_are_renormalised():
    preds = _preds_for_coverage(0.45, 0.45)
    scales = {10: HorizonScales(ret=1.0, markout=1.0, adverse=1.0)}
    all_heads = {"direction", "quantile", "markout", "adverse"}
    s_all, comp_all = composite_score(preds, horizons=[10], scales=scales, enabled_heads=all_heads)
    # with only the direction head, the score should be the (renormalised) BA terms
    s_dir, comp_dir = composite_score(preds, horizons=[10], scales=scales, enabled_heads={"direction"})
    assert "val_quantile_loss_mean" in comp_all
    assert "val_quantile_loss_mean" not in comp_dir
    # direction-only score is dominated by balanced accuracy (renormalised to ~1)
    assert s_dir > 0.0
    # an empty enabled set yields a zero score, not an error
    s_none, _ = composite_score(preds, horizons=[10], scales=scales, enabled_heads=set())
    assert s_none == 0.0


def test_score_uses_validation_rows_only():
    # extra rows tagged to a different horizon must not leak into the h=10 score
    preds = _preds_for_coverage(0.45, 0.45)
    poison = preds.copy()
    poison["horizon"] = 50
    poison["pred_class"] = -poison["true_direction"]  # would wreck accuracy if counted
    combined = pd.concat([preds, poison], ignore_index=True)
    scales = {10: HorizonScales(ret=1.0, markout=1.0, adverse=1.0)}
    s_clean, _ = composite_score(preds, horizons=[10], scales=scales,
                                 enabled_heads={"direction", "quantile", "markout", "adverse"})
    s_filtered, _ = composite_score(combined, horizons=[10], scales=scales,
                                    enabled_heads={"direction", "quantile", "markout", "adverse"})
    assert s_clean == s_filtered  # only horizon 10 is scored


def test_scales_fit_on_training_rows_only():
    horizons = [10]
    # MAD of returns ~ 0.674 for a standard normal; markout/adverse scales add up
    rng = np.random.default_rng(1)
    r = {10: rng.normal(0, 1, 5000)}
    mkb = {10: rng.normal(0, 2, 5000)}
    mka = {10: rng.normal(0, 2, 5000)}
    avb = {10: np.abs(rng.normal(0, 1, 5000))}
    ava = {10: np.abs(rng.normal(0, 1, 5000))}
    scales = fit_scales(horizons, r, mkb, mka, avb, ava)
    assert 0.5 < scales[10].ret < 0.9
    assert scales[10].markout > scales[10].ret  # two MADs at sd=2
    assert scales[10].adverse > 0.0
