"""Inventory, cash and wealth accounting for the market-making simulator.

A filled bid adds inventory and spends cash; a filled ask reduces inventory and
adds cash; a maker fee is charged per fill. Marked wealth is cash plus inventory
valued at the current mid. The inventory limit |x| <= x_max is enforced; an
action that would breach it is rejected (logged), per config.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Portfolio:
    """Running market-making state for one monthly day."""

    max_inventory: float
    maker_fee_rate: float = 0.0
    cash: float = 0.0
    inventory: float = 0.0
    wealth_peak: float = 0.0
    n_inventory_rejects: int = 0
    _started: bool = field(default=False, repr=False)

    def wealth(self, mid: float) -> float:
        return self.cash + self.inventory * mid

    def mark(self, mid: float) -> float:
        """Mark to market and update the running wealth peak / drawdown."""
        w = self.wealth(mid)
        if not self._started:
            self.wealth_peak = w
            self._started = True
        else:
            self.wealth_peak = max(self.wealth_peak, w)
        return w

    def drawdown(self, mid: float) -> float:
        return max(0.0, self.wealth_peak - self.wealth(mid))

    def can_fill(self, side: str, size: float) -> bool:
        signed = size if side == "bid" else -size
        return abs(self.inventory + signed) <= self.max_inventory + 1e-12

    def apply_fill(self, side: str, size: float, fill_price: float) -> bool:
        """Apply a fill if it respects the inventory limit; return whether it applied."""
        if not self.can_fill(side, size):
            self.n_inventory_rejects += 1
            return False
        fee = self.maker_fee_rate * fill_price * size
        if side == "bid":
            self.inventory += size
            self.cash -= size * fill_price + fee
        else:
            self.inventory -= size
            self.cash += size * fill_price - fee
        return True
