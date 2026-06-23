"""Tests for the conservative passive-fill approximations."""

from __future__ import annotations

import numpy as np

from lob_forecasting.backtesting.market_making.fills import infer_tick_size, resolve_fill


def test_bid_fill_when_future_mid_drops_to_quote():
    mid = np.array([100.0, 100.0, 99.0, 98.0])
    bid = mid - 0.5
    ask = mid + 0.5
    # quote a bid at 99.0; the mid reaches 99.0 at index 2
    filled, idx = resolve_fill(mid, bid, ask, t=0, horizon=3, quote_price=99.0, side="bid")
    assert filled and idx == 2


def test_ask_fill_when_future_mid_rises_to_quote():
    mid = np.array([100.0, 100.5, 101.0, 101.0])
    bid = mid - 0.5
    ask = mid + 0.5
    filled, idx = resolve_fill(mid, bid, ask, t=0, horizon=3, quote_price=101.0, side="ask")
    assert filled and idx == 2


def test_unfilled_quote_expires():
    mid = np.array([100.0, 100.0, 100.0, 100.0])
    bid = mid - 0.5
    ask = mid + 0.5
    filled, idx = resolve_fill(mid, bid, ask, t=0, horizon=3, quote_price=95.0, side="bid")
    assert not filled and idx == -1


def test_fill_is_first_crossing_event():
    mid = np.array([100.0, 99.0, 98.0, 99.0])
    bid = mid - 0.5
    ask = mid + 0.5
    filled, idx = resolve_fill(mid, bid, ask, t=0, horizon=3, quote_price=99.0, side="bid")
    assert filled and idx == 1  # first event satisfying the condition


def test_touch_through_is_stricter():
    mid = np.array([100.0, 99.0, 99.0])
    bid = mid - 0.5
    ask = mid + 0.5  # ask = 99.5 at index 1 (mid 99)
    # mid-cross fills a bid at 99.0 (mid reaches 99), touch-through needs ask<=99
    f_mid, _ = resolve_fill(mid, bid, ask, 0, 2, 99.0, "bid", "conservative_touch_or_mid_cross")
    f_through, _ = resolve_fill(mid, bid, ask, 0, 2, 99.0, "bid", "touch_through")
    assert f_mid and not f_through


def test_infer_tick_size():
    prices = np.array([100.00, 100.01, 100.02, 100.03])
    assert abs(infer_tick_size(prices) - 0.01) < 1e-9
