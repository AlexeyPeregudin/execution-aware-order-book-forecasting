"""Deterministic market-making policies.

Each policy maps the current market/forecast state and inventory to one of the
four quote actions. Parameters (soft inventory limit, forecast threshold,
uncertainty / adverse thresholds) are tuned on validation folds only; the
environment still enforces the hard inventory limit on top.
"""

from __future__ import annotations

from .environment import ACTION_ASK, ACTION_BID, ACTION_BOTH, ACTION_NEITHER, MarketArrays


def _combine(quote_bid: bool, quote_ask: bool) -> int:
    if quote_bid and quote_ask:
        return ACTION_BOTH
    if quote_bid:
        return ACTION_BID
    if quote_ask:
        return ACTION_ASK
    return ACTION_NEITHER


class Policy:
    """Base policy: subclasses implement act()."""

    name = "policy"

    def __init__(self, **params: float) -> None:
        self.params = params

    def act(self, arrays: MarketArrays, t: int, inventory: float) -> int:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        pass


class NoQuoteMM(Policy):
    """Risk-avoidance baseline: never quotes (zero fills, zero reward).

    Acts as the do-nothing reference the control optimiser has to beat.
    """

    name = "no_quote"

    def act(self, arrays: MarketArrays, t: int, inventory: float) -> int:
        return ACTION_NEITHER


class NaiveSymmetricMM(Policy):
    name = "naive_symmetric_mm"

    def act(self, arrays: MarketArrays, t: int, inventory: float) -> int:
        return ACTION_BOTH


class InventorySkewedMM(Policy):
    name = "inventory_skewed_mm"

    def act(self, arrays: MarketArrays, t: int, inventory: float) -> int:
        rho = float(self.params.get("rho", 0.5))
        x_soft = rho * float(self.params.get("max_inventory", 1.0))
        quote_bid = inventory < x_soft
        quote_ask = inventory > -x_soft
        return _combine(quote_bid, quote_ask)


class ForecastAwareMM(Policy):
    name = "forecast_aware_mm"

    def act(self, arrays: MarketArrays, t: int, inventory: float) -> int:
        theta = float(self.params.get("theta", 0.0))
        pred = arrays.signal("pred_return", t)
        if pred > theta:
            return ACTION_BID
        if pred < -theta:
            return ACTION_ASK
        # otherwise both sides, with an inventory skew
        x_soft = float(self.params.get("rho", 0.75)) * float(self.params.get("max_inventory", 1.0))
        return _combine(inventory < x_soft, inventory > -x_soft)


class UncertaintyAwareMM(Policy):
    name = "uncertainty_aware_mm"

    def act(self, arrays: MarketArrays, t: int, inventory: float) -> int:
        u_thresh = float(self.params.get("u_thresh", 1e9))
        adv_thresh = float(self.params.get("adv_thresh", 1e9))
        theta = float(self.params.get("theta", 0.0))

        width = arrays.signal("interval_width", t)
        if width > u_thresh:
            # too uncertain: only quote the inventory-reducing side
            if inventory > 0:
                return ACTION_ASK
            if inventory < 0:
                return ACTION_BID
            return ACTION_NEITHER

        pred = arrays.signal("pred_return", t)
        quote_bid = pred >= -theta
        quote_ask = pred <= theta
        if arrays.signal("adv_bid", t) > adv_thresh:
            quote_bid = False
        if arrays.signal("adv_ask", t) > adv_thresh:
            quote_ask = False
        return _combine(quote_bid, quote_ask)


_DETERMINISTIC = {
    NoQuoteMM.name: NoQuoteMM,
    NaiveSymmetricMM.name: NaiveSymmetricMM,
    InventorySkewedMM.name: InventorySkewedMM,
    ForecastAwareMM.name: ForecastAwareMM,
    UncertaintyAwareMM.name: UncertaintyAwareMM,
}


def deterministic_policies() -> list[str]:
    return list(_DETERMINISTIC)


def build_policy(name: str, **params: float) -> Policy:
    if name not in _DETERMINISTIC:
        raise KeyError(f"Unknown deterministic policy {name!r}; known: {sorted(_DETERMINISTIC)}")
    return _DETERMINISTIC[name](**params)
