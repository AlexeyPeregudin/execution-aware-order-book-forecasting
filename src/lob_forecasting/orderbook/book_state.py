"""The in-memory order book.

Keeps a price -> size map for each side, applies updates (size 0 removes a
level), and hands back the best K levels. Small and easy to test on its own.
"""

from __future__ import annotations

import math


def _is_missing(x: float | None) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


class BookState:
    """Bid and ask books. Best bid is the highest price, best ask the lowest."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def clear(self) -> None:
        """Empty both sides (we do this when a full snapshot resets the book)."""
        self.bids.clear()
        self.asks.clear()

    def apply(self, side: str, price: float | None, quantity: float | None) -> bool:
        """Set price -> quantity on one side. Size 0 (or missing) removes the level.

        Returns True if this was a book update (a bid or ask), so the caller
        knows the row touched the book. A missing price is still "book-relevant"
        but we can't place it, so it shows up later as a missing level.
        """
        if side not in ("bid", "ask"):
            return False
        if _is_missing(price):
            return True
        book = self.bids if side == "bid" else self.asks
        if _is_missing(quantity) or quantity == 0:
            book.pop(float(price), None)
        else:
            book[float(price)] = float(quantity)
        return True

    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def top_k(self, k: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Best k levels each side as (price, size) lists, best first.

        Bids sorted high to low, asks low to high, so prices are strictly
        monotone by level.
        """
        bids = sorted(self.bids.items(), key=lambda kv: -kv[0])[:k]
        asks = sorted(self.asks.items(), key=lambda kv: kv[0])[:k]
        return bids, asks
