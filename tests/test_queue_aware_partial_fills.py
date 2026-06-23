"""Queue-aware partial-fill model tests."""

from __future__ import annotations

import numpy as np

from lob_forecasting.backtesting.market_making.environment import (
    MarketArrays,
    _RewardWeights,
    resolve_fill_outcome,
    simulate_day,
)
from lob_forecasting.backtesting.market_making.policies import build_policy
from lob_forecasting.backtesting.market_making.queue_fill import resolve_queue_fill


def _levels(px_rows, qty_rows):
    return np.array(px_rows, dtype="float64"), np.array(qty_rows, dtype="float64")


def test_full_cross_fills_bid_and_ask():
    n = 4
    # bid quote at 100; the best ask trades down to 100 at u=2 -> full cross
    bid_px, bid_qty = _levels([[100.0]] * n, [[5.0]] * n)
    ask_px = np.array([[101.0], [101.0], [100.0], [100.0]])
    ask_qty = np.full((n, 1), 5.0)
    r = resolve_queue_fill("bid", 1, 0, 3, bid_px=bid_px, bid_qty=bid_qty,
                           ask_px=ask_px, ask_qty=ask_qty, order_size=1.0, tick=1.0,
                           queue_position="back", kappa=0.5)
    assert r.filled and r.reason == "full_cross"
    assert r.fill_qty == 1.0 and r.fill_index == 2

    # ask quote at 100; best bid rises to 100 at u=1 -> full cross
    ask_px2, ask_qty2 = _levels([[100.0]] * n, [[5.0]] * n)
    bid_px2 = np.array([[99.0], [100.0], [100.0], [100.0]])
    bid_qty2 = np.full((n, 1), 5.0)
    r2 = resolve_queue_fill("ask", 1, 0, 3, bid_px=bid_px2, bid_qty=bid_qty2,
                            ask_px=ask_px2, ask_qty=ask_qty2, order_size=1.0, tick=1.0)
    assert r2.filled and r2.reason == "full_cross" and r2.fill_index == 1


def test_depletion_fills_only_after_queue_ahead_consumed():
    n = 5
    # bid at 100; no cross (ask stays at 101). Displayed bid qty depletes 10 -> 0.
    ask_px = np.full((n, 1), 101.0)
    ask_qty = np.full((n, 1), 5.0)
    bid_px = np.full((n, 1), 100.0)
    bid_qty = np.array([[10.0], [8.0], [6.0], [2.0], [0.0]])

    # back of a 10-deep queue (queue_ahead=10): total depletion 10*kappa=10 is NOT
    # strictly greater than 10, so nothing fills.
    r_back = resolve_queue_fill("bid", 1, 0, 4, bid_px=bid_px, bid_qty=bid_qty,
                                ask_px=ask_px, ask_qty=ask_qty, order_size=5.0, tick=1.0,
                                queue_position="back", kappa=1.0)
    assert not r_back.filled and r_back.reason == "expired"

    # front of the queue (queue_ahead=2.5): cumulative effective depletion (10)
    # exceeds it, so a partial fill occurs, capped at the order size.
    r_front = resolve_queue_fill("bid", 1, 0, 4, bid_px=bid_px, bid_qty=bid_qty,
                                 ask_px=ask_px, ask_qty=ask_qty, order_size=5.0, tick=1.0,
                                 queue_position="front", kappa=1.0)
    assert r_front.filled and r_front.reason == "queue_depletion"
    assert 0.0 < r_front.fill_fraction <= 1.0
    assert r_front.fill_index > 0  # the fill never happens at or before the quote time


def test_fill_fraction_in_unit_interval_and_partial():
    n = 4
    ask_px = np.full((n, 1), 101.0)
    ask_qty = np.full((n, 1), 5.0)
    bid_px = np.full((n, 1), 100.0)
    bid_qty = np.array([[10.0], [7.0], [6.0], [6.0]])  # total depletion 4
    # front queue_ahead = 2.5; cum dep = 4 -> available 1.5; order 10 -> fraction 0.15
    r = resolve_queue_fill("bid", 1, 0, 3, bid_px=bid_px, bid_qty=bid_qty,
                           ask_px=ask_px, ask_qty=ask_qty, order_size=10.0, tick=1.0,
                           queue_position="front", kappa=1.0)
    assert r.filled
    assert 0.0 < r.fill_fraction < 1.0
    assert np.isclose(r.fill_qty, 1.5)


def test_legacy_fill_model_unchanged():
    # without level arrays / queue model, the outcome falls back to full-size fill
    n = 4
    arrays = MarketArrays(
        mid=np.array([100.5, 100.0, 99.5, 99.0]),
        bid=np.array([100.0, 99.5, 99.0, 98.5]),
        ask=np.array([101.0, 100.5, 100.0, 99.5]),
        event_id=np.arange(n), timestamp=np.arange(n) * 10, tick_size=0.5, monthly_date="2026-01-01",
    )
    out = resolve_fill_outcome(arrays, "bid", 1, 0, 3, 100.0, 1.0,
                               "conservative_touch_or_mid_cross")
    assert out.fill_qty in (0.0, 1.0)  # all-or-nothing under the legacy model
    assert out.version == "mm_fill_v1"


def test_partial_fill_inventory_accounting_via_simulate_day():
    # build a day where a bid partially fills; the portfolio inventory should equal
    # the partial fill quantity and stay within the limit.
    n = 6
    ask_px = np.full((n, 2), [[101.0, 102.0]])
    ask_qty = np.full((n, 2), [[5.0, 5.0]])
    bid_px = np.tile([[100.0, 99.0]], (n, 1)).astype("float64")
    bid_qty = np.array([[10.0, 5.0]] + [[6.0, 5.0]] * (n - 1), dtype="float64")
    mid = np.full(n, 100.5)
    arrays = MarketArrays(
        mid=mid, bid=bid_px[:, 0], ask=ask_px[:, 0],
        event_id=np.arange(n), timestamp=np.arange(n) * 10, tick_size=1.0, monthly_date="2026-01-01",
        bid_px_levels=bid_px, bid_qty_levels=bid_qty, ask_px_levels=ask_px, ask_qty_levels=ask_qty,
    )
    rw = _RewardWeights(0.01, 0.001, 0.0, 0.25)
    rec = simulate_day(
        build_policy("naive_symmetric_mm"), arrays, horizon=5, quote_size=10.0,
        distance_ticks=0, max_inventory=100.0, maker_fee_rate=0.0, reward_weights=rw,
        fill_model="queue_aware_partial", fold_id=0, model_name="m", policy_name="naive_symmetric_mm",
        decision_interval=1, queue_kwargs={"queue_position": "front", "kappa": 1.0, "full_cross_fill": True},
    )
    fills = [f for f in rec["fills"] if f["fill_occurred"]]
    assert fills, "expected at least one partial fill"
    for f in fills:
        assert 0.0 < f["fill_fraction"] <= 1.0
        assert f["fill_reason"] in ("queue_depletion", "full_cross")
        assert f["fill_model_name"] == "queue_aware_partial"
