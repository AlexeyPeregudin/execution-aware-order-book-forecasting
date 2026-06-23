"""Tests for the deterministic market-making policies and validation-only selection."""

from __future__ import annotations

import numpy as np

from lob_forecasting.backtesting.market_making.environment import (
    ACTION_ASK, ACTION_BID, ACTION_BOTH, MarketArrays,
)
from lob_forecasting.backtesting.market_making.policies import build_policy


def _arrays(n=10, monthly_date="2024-01-01", pred=0.0, seed=0):
    rng = np.random.default_rng(seed)
    mid = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    return MarketArrays(
        mid=mid, bid=mid - 0.5, ask=mid + 0.5,
        event_id=np.arange(n), timestamp=np.arange(n) * 100 + 1_700_000_000_000_000_000,
        tick_size=0.01, monthly_date=monthly_date,
        signals={"pred_return": np.full(n, pred), "interval_width": np.zeros(n),
                 "adv_bid": np.zeros(n), "adv_ask": np.zeros(n)},
        state_matrix=None, state_names=[],
    )


def test_naive_always_quotes_both():
    a = _arrays()
    p = build_policy("naive_symmetric_mm")
    assert all(p.act(a, t, 0.0) == ACTION_BOTH for t in range(a.n))


def test_inventory_skew_avoids_increasing_inventory():
    a = _arrays()
    p = build_policy("inventory_skewed_mm", rho=0.5, max_inventory=1.0)
    assert p.act(a, 0, 0.0) == ACTION_BOTH        # flat -> both
    assert p.act(a, 0, 0.8) == ACTION_ASK         # long -> stop quoting bid
    assert p.act(a, 0, -0.8) == ACTION_BID        # short -> stop quoting ask


def test_forecast_aware_follows_prediction():
    up = build_policy("forecast_aware_mm", theta=0.001)
    assert up.act(_arrays(pred=0.01), 0, 0.0) == ACTION_BID
    assert up.act(_arrays(pred=-0.01), 0, 0.0) == ACTION_ASK
    assert up.act(_arrays(pred=0.0), 0, 0.0) == ACTION_BOTH


def test_policy_selection_uses_validation_only():
    from lob_forecasting.backtesting.market_making.run import _select_policy

    from ._monthly_helpers import monthly_config

    cfg = monthly_config()
    val = _arrays(n=40, monthly_date="2024-04-01", pred=0.02, seed=1)
    # two different "test" days; selection must ignore them entirely
    days_a = {"2024-04-01": {"split": "validation", "arrays": val},
              "2024-05-01": {"split": "test", "arrays": _arrays(n=40, monthly_date="2024-05-01", pred=-0.5, seed=2)}}
    days_b = {"2024-04-01": {"split": "validation", "arrays": val},
              "2024-05-01": {"split": "test", "arrays": _arrays(n=40, monthly_date="2024-05-01", pred=99.0, seed=3)}}
    _, params_a, _ = _select_policy("forecast_aware_mm", cfg, days_a, {"2024-04-01"}, 0, "m")
    _, params_b, _ = _select_policy("forecast_aware_mm", cfg, days_b, {"2024-04-01"}, 0, "m")
    assert params_a == params_b  # test-day data never influenced the selection
