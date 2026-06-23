"""Control quote-optimiser tests."""

from __future__ import annotations

import numpy as np

from lob_forecasting.backtesting.market_making.control import (
    CONTROL_ACTIONS,
    ControlQuoteOptimizer,
)
from lob_forecasting.backtesting.market_making.environment import MarketArrays


class _StubFillModel:
    """Returns a fixed fill probability per side (ex-ante; no future access)."""

    def __init__(self, bid_p=0.9, ask_p=0.9):
        self.bid_p, self.ask_p = bid_p, ask_p

    def predict_one(self, x: np.ndarray) -> float:
        # the fill feature row appends [is_bid, level, log1p(depth)]
        is_bid, level = x[-3], x[-2]
        base = self.bid_p if is_bid > 0.5 else self.ask_p
        return base * (1.0 if level <= 1 else 0.4)  # deeper quotes fill less often


def _arrays(n=3, markout_bid=1.0, markout_ask=-1.0):
    bid_px = np.tile([[100.0, 99.0]], (n, 1)).astype("float64")
    ask_px = np.tile([[101.0, 102.0]], (n, 1)).astype("float64")
    qty = np.full((n, 2), 5.0)

    def sig(v):
        return np.full(n, v)

    return MarketArrays(
        mid=np.full(n, 100.5), bid=bid_px[:, 0], ask=ask_px[:, 0],
        event_id=np.arange(n), timestamp=np.arange(n) * 10, tick_size=1.0,
        monthly_date="2026-01-01",
        signals={"markout_bid": sig(markout_bid), "markout_ask": sig(markout_ask),
                 "adv_bid": sig(0.0), "adv_ask": sig(0.0),
                 "pred_q05": sig(-0.5), "pred_q50": sig(0.0), "pred_q95": sig(0.5),
                 "interval_width": sig(1.0)},
        bid_px_levels=bid_px, bid_qty_levels=qty, ask_px_levels=ask_px, ask_qty_levels=qty,
        fill_features=np.zeros((n, 3)), fill_feature_names=["f0", "f1", "f2"],
    )


def _opt(**params):
    base = dict(quote_size=1.0, max_inventory=1.0, horizon=2, maker_fee=0.0,
                action_levels=[1, 2], lambda_inv=0.0, lambda_turn=0.0,
                lambda_adv=0.25, lambda_unc=0.0, lambda_act=0.0)
    base.update(params)
    return ControlQuoteOptimizer(fill_model=_StubFillModel(), **base)


def test_inventory_breaching_actions_rejected():
    opt = _opt()
    # at the long limit, no action that submits a bid can be chosen
    quotes = opt.act(_arrays(), t=0, inventory=1.0)
    assert all(side != "bid" for side, _ in quotes)
    # feasible scored actions never include a bid quote either
    scores = opt.score_actions(_arrays(), 0, 1.0)
    for a in scores:
        assert all(side != "bid" for side, _ in CONTROL_ACTIONS[a])


def test_chooses_action_that_maximises_J():
    # bid is profitable (markout +1), ask is not (-1); with no penalties the
    # optimiser should quote the bid at level 1.
    opt = _opt()
    quotes = opt.act(_arrays(markout_bid=1.0, markout_ask=-1.0), t=0, inventory=0.0)
    assert quotes == [("bid", 1)]


def test_can_choose_no_quote_when_nothing_is_attractive():
    # both sides unprofitable + an action cost -> quote neither
    opt = _opt(markout=None, lambda_act=0.01)
    quotes = opt.act(_arrays(markout_bid=-1.0, markout_ask=-1.0), t=0, inventory=0.0)
    assert quotes == []


def test_controller_uses_only_present_time_no_future_leak():
    opt = _opt()
    arr = _arrays(n=5)
    a0 = opt.act(arr, t=0, inventory=0.0)
    # mutate the future (t>=2) signals; the decision at t=0 must not change
    arr.signals["markout_bid"][2:] = -100.0
    a1 = opt.act(arr, t=0, inventory=0.0)
    assert a0 == a1


def test_level_adjusted_markout_and_adverse():
    opt = _opt()
    arr = _arrays()
    # quoting deeper (level 2) improves markout and reduces adverse by the price gap
    m1, a1 = opt._markout_adverse(arr, 0, "bid", 1)
    m2, a2 = opt._markout_adverse(arr, 0, "bid", 2)
    assert m2 > m1            # deeper bid -> better markout
    assert a2 <= a1           # deeper bid -> no more adverse selection
