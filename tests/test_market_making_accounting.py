"""Tests for the market-making inventory / cash / wealth accounting."""

from __future__ import annotations

import pytest

from lob_forecasting.backtesting.market_making.accounting import Portfolio


def test_bid_fill_updates_inventory_and_cash():
    p = Portfolio(max_inventory=5.0, maker_fee_rate=0.0)
    assert p.apply_fill("bid", 1.0, 100.0)
    assert p.inventory == 1.0
    assert p.cash == pytest.approx(-100.0)


def test_ask_fill_updates_inventory_and_cash():
    p = Portfolio(max_inventory=5.0, maker_fee_rate=0.0)
    assert p.apply_fill("ask", 1.0, 101.0)
    assert p.inventory == -1.0
    assert p.cash == pytest.approx(101.0)


def test_wealth_marks_inventory_at_mid():
    p = Portfolio(max_inventory=5.0)
    p.apply_fill("bid", 2.0, 100.0)  # cash -200, inventory +2
    assert p.wealth(101.0) == pytest.approx(-200.0 + 2.0 * 101.0)


def test_maker_fee_reduces_cash():
    p = Portfolio(max_inventory=5.0, maker_fee_rate=0.001)
    p.apply_fill("bid", 1.0, 100.0)
    # cash = -(1*100) - fee(0.001*100*1) = -100.1
    assert p.cash == pytest.approx(-100.1)


def test_inventory_limit_rejects_breaching_fill():
    p = Portfolio(max_inventory=1.0)
    assert p.apply_fill("bid", 1.0, 100.0)       # inventory -> 1.0
    assert not p.apply_fill("bid", 1.0, 100.0)   # would exceed limit -> rejected
    assert p.inventory == 1.0
    assert p.n_inventory_rejects == 1


def test_drawdown_tracks_peak():
    p = Portfolio(max_inventory=5.0)
    p.apply_fill("bid", 1.0, 100.0)  # inventory 1, cash -100
    p.mark(105.0)  # wealth = 5, peak 5
    assert p.drawdown(102.0) == pytest.approx(5.0 - 2.0)  # wealth now 2
