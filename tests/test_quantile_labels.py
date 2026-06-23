"""Tests for the quantile (pinball) helpers used by the multi-task model."""

from __future__ import annotations

import numpy as np
import pytest

from lob_forecasting.labels.quantiles import interval_coverage, mean_pinball_loss, pinball_loss


def test_pinball_loss_matches_definition():
    r = np.array([1.0, -1.0])
    q = np.array([0.0, 0.0])
    # tau=0.5 pinball is 0.5*|r-q|
    assert np.allclose(pinball_loss(r, q, 0.5), 0.5 * np.abs(r - q))


def test_pinball_asymmetric_penalty():
    # under-prediction at tau=0.95 is penalised more than over-prediction
    r = np.array([1.0])
    under = mean_pinball_loss(r, np.array([0.0]), 0.95)  # q below r
    over = mean_pinball_loss(r, np.array([2.0]), 0.95)   # q above r
    assert under > over


def test_interval_coverage_counts_inside_fraction():
    r = np.array([0.0, 0.5, 1.5, -2.0])
    lo = np.full(4, -1.0)
    hi = np.full(4, 1.0)
    # 0.0 and 0.5 inside; 1.5 and -2.0 outside -> 0.5
    assert interval_coverage(r, lo, hi) == pytest.approx(0.5)


def test_empty_inputs_are_nan():
    assert np.isnan(mean_pinball_loss(np.array([]), np.array([]), 0.5))
    assert np.isnan(interval_coverage(np.array([]), np.array([]), np.array([])))
