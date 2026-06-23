"""Queue-aware partial-fill model.

Top-5 snapshots carry no true queue position, but we can still do better than a
plain touch/mid-cross proxy:

  - a resting quote at a visible price has some queue ahead of it, sized as a
    fraction rho of the displayed quantity at that price (front/middle/back);
  - as that displayed quantity depletes over the holding horizon, a fraction
    kappa of the reduction counts as fill-relevant (the rest we treat as
    cancellations); once cumulative effective depletion exceeds the queue ahead,
    the quote starts filling, up to the order size;
  - if the opposite best price trades through the quote (a full cross), we assume
    the quote fills completely at the first such event.

We report one aggregated fill per quote side per decision: the fill timestamp is
the earliest event with a positive cumulative fill and the quantity is its value
by expiry. The model never looks at data before the quote time, so it stays
causal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QUEUE_AWARE_FILL_VERSION = "mm_queue_fill_v1"

# queue-position multiplier rho_q
QUEUE_POSITION_RHO: dict[str, float] = {"front": 0.25, "middle": 0.50, "back": 1.00}


@dataclass
class QueueFillResult:
    """Outcome of resolving one resting quote under the queue-aware model."""

    filled: bool
    fill_index: int           # first event with positive cumulative fill, -1 if none
    fill_qty: float           # aggregated fill quantity by expiry
    fill_fraction: float      # fill_qty / order_size in [0, 1]
    queue_ahead: float
    cum_effective_depletion: float
    reason: str               # full_cross | queue_depletion | expired


def displayed_qty_at_price(
    px_levels: np.ndarray, qty_levels: np.ndarray, u: int, price: float, tol: float
) -> float:
    """Displayed quantity at `price` in the top-K of one side at event `u`.

    px_levels / qty_levels are (n, K). Returns 0 if the price is not visible.
    """
    row_px = px_levels[u]
    row_qty = qty_levels[u]
    mask = np.isfinite(row_px) & (np.abs(row_px - price) <= tol)
    return float(np.nansum(row_qty[mask])) if mask.any() else 0.0


def resolve_queue_fill(
    side: str,
    level: int,
    t: int,
    horizon: int,
    *,
    bid_px: np.ndarray,
    bid_qty: np.ndarray,
    ask_px: np.ndarray,
    ask_qty: np.ndarray,
    order_size: float,
    tick: float,
    queue_position: str = "back",
    kappa: float = 0.5,
    full_cross_fill: bool = True,
) -> QueueFillResult:
    """Resolve a resting bid/ask quote at `level` (1 or 2) under the queue model.

    All `*_px` / `*_qty` arrays are (n, K) with level 1 in column 0. The quote
    price is the side's displayed price at `level` at quote time `t`.
    """
    n = bid_px.shape[0]
    if side == "bid":
        own_px, own_qty, opp_best = bid_px, bid_qty, ask_px[:, 0]
    else:
        own_px, own_qty, opp_best = ask_px, ask_qty, bid_px[:, 0]
    col = level - 1
    if col < 0 or col >= own_px.shape[1]:
        return QueueFillResult(False, -1, 0.0, 0.0, 0.0, 0.0, "expired")
    quote_price = own_px[t, col]
    if not np.isfinite(quote_price):
        return QueueFillResult(False, -1, 0.0, 0.0, 0.0, 0.0, "expired")

    tol = max(tick / 2.0, 1e-12)
    q_at_t = displayed_qty_at_price(own_px, own_qty, t, quote_price, tol)
    queue_ahead = QUEUE_POSITION_RHO.get(queue_position, 1.0) * q_at_t

    end = min(t + horizon, n - 1)
    cum_dep = 0.0
    prev_q = q_at_t
    first_fill_idx = -1
    fill_qty = 0.0
    for u in range(t + 1, end + 1):
        if full_cross_fill:
            ref = opp_best[u]
            crossed = np.isfinite(ref) and (
                (side == "bid" and ref <= quote_price) or (side == "ask" and ref >= quote_price)
            )
            if crossed:
                return QueueFillResult(True, u, order_size, 1.0, queue_ahead, cum_dep, "full_cross")
        cur_q = displayed_qty_at_price(own_px, own_qty, u, quote_price, tol)
        cum_dep += kappa * max(0.0, prev_q - cur_q)
        prev_q = cur_q
        q_fill = min(order_size, max(0.0, cum_dep - queue_ahead))
        if q_fill > 0.0 and first_fill_idx < 0:
            first_fill_idx = u
        fill_qty = q_fill

    if first_fill_idx >= 0 and fill_qty > 0.0:
        return QueueFillResult(
            True, first_fill_idx, fill_qty, fill_qty / order_size, queue_ahead, cum_dep, "queue_depletion"
        )
    return QueueFillResult(False, -1, 0.0, 0.0, queue_ahead, cum_dep, "expired")
