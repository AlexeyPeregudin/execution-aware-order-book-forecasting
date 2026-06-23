"""Tests for the passive markout / adverse-selection labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lob_forecasting.labels.markout import compute_markout_labels


def test_markout_matches_hand_calculation():
    mid = pd.Series([100.0, 101.0, 102.0])
    spread = pd.Series([2.0, 2.0, 2.0])  # best_bid = mid-1, best_ask = mid+1
    out = compute_markout_labels(mid, spread, [1])
    # bid markout_h1[0] = mid_{1} - best_bid_0 = 101 - 99 = 2
    assert out["markout_bid_h1"].iloc[0] == pytest.approx(101.0 - 99.0)
    # ask markout_h1[0] = best_ask_0 - mid_{1} = 101 - 101 = 0
    assert out["markout_ask_h1"].iloc[0] == pytest.approx(101.0 - 101.0)


def test_adverse_is_nonnegative_and_complementary():
    mid = pd.Series([100.0, 98.0, 96.0])  # falling price
    spread = pd.Series([2.0, 2.0, 2.0])
    out = compute_markout_labels(mid, spread, [1])
    # a resting bid filled into a falling market is adversely selected
    assert (out["adverse_bid_h1"].dropna() >= 0).all()
    assert (out["adverse_ask_h1"].dropna() >= 0).all()
    # adverse_bid = max(0, -markout_bid)
    mb = out["markout_bid_h1"].iloc[0]
    assert out["adverse_bid_h1"].iloc[0] == pytest.approx(max(0.0, -mb))


def test_last_h_rows_unavailable():
    mid = pd.Series(np.arange(6, dtype="float64") + 100.0)
    spread = pd.Series(np.ones(6))
    out = compute_markout_labels(mid, spread, [2])
    assert not out["markout_available_h2"].iloc[-2:].any()
    assert out["markout_available_h2"].iloc[:-2].all()
    assert pd.isna(out["markout_bid_h2"].iloc[-1])


def test_can_disable_markout_or_adverse():
    mid = pd.Series([100.0, 101.0])
    spread = pd.Series([1.0, 1.0])
    out = compute_markout_labels(mid, spread, [1], include_markout=False, include_adverse=True)
    assert "markout_bid_h1" not in out.columns
    assert "adverse_bid_h1" in out.columns
