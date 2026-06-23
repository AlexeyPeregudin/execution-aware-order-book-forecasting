"""Conservative passive-fill approximations.

Snapshot data has no queue information, so a resting quote is assumed filled only
if the book path later reaches it. Two models:

  conservative_touch_or_mid_cross : bid fills if min future mid <= quote;
                                    ask fills if max future mid >= quote.
  touch_through                   : bid fills if min future ask <= quote;
                                    ask fills if max future bid >= quote (stricter).

The fill timestamp is the first event in (t, t+h] satisfying the condition.
These are proxies and do not claim true queue priority.
"""

from __future__ import annotations

import numpy as np

FILL_ASSUMPTION_VERSION = "mm_fill_v1"


def infer_tick_size(prices: np.ndarray) -> float:
    """Infer the tick from the smallest positive gap between adjacent clean prices."""
    p = np.asarray(prices, dtype="float64")
    p = p[np.isfinite(p)]
    if p.size < 2:
        return 0.01
    diffs = np.abs(np.diff(np.unique(np.round(p, 8))))
    diffs = diffs[diffs > 1e-9]
    if diffs.size == 0:
        return 0.01
    return float(np.min(diffs))


def resolve_fill(
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    t: int,
    horizon: int,
    quote_price: float,
    side: str,
    model: str = "conservative_touch_or_mid_cross",
) -> tuple[bool, int]:
    """Return (filled, fill_index). fill_index is -1 if the quote expires unfilled."""
    n = len(mid)
    end = min(t + horizon, n - 1)
    for u in range(t + 1, end + 1):
        if model == "touch_through":
            ref = ask[u] if side == "bid" else bid[u]
        else:
            ref = mid[u]
        if not np.isfinite(ref):
            continue
        if side == "bid" and ref <= quote_price:
            return True, u
        if side == "ask" and ref >= quote_price:
            return True, u
    return False, -1
