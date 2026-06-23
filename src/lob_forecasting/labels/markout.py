"""Passive markout and adverse-selection labels.

For a passive quote resting at the current touch, the markout after horizon h is
how favourable the midpoint move was:

    M_bid_{t,h} = m_{t+h} - p^{b,1}_t      (a resting bid filled, then mid rose)
    M_ask_{t,h} = p^{a,1}_t - m_{t+h}      (a resting ask filled, then mid fell)

Adverse-selection cost is the unfavourable part:

    A_bid = max(0, -M_bid),   A_ask = max(0, -M_ask).

These are proxies for a passive fill at the touch, not real realised fills.
The best bid/ask are reconstructed from mid and spread, so they do not require
the raw level columns. Everything shifts forward in time within a single day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_markout_labels(
    mid: pd.Series,
    spread: pd.Series,
    horizon_list: list[int],
    *,
    include_markout: bool = True,
    include_adverse: bool = True,
) -> pd.DataFrame:
    """Markout / adverse-selection labels for each horizon over one day's rows."""
    mid = mid.astype("float64")
    spread = spread.astype("float64")
    best_bid = mid - spread / 2.0
    best_ask = mid + spread / 2.0

    out: dict[str, pd.Series] = {}
    for h in horizon_list:
        future_mid = mid.shift(-h)
        available = future_mid.notna() & mid.notna()

        m_bid = (future_mid - best_bid).where(available)
        m_ask = (best_ask - future_mid).where(available)

        if include_markout:
            out[f"markout_bid_h{h}"] = m_bid
            out[f"markout_ask_h{h}"] = m_ask
        if include_adverse:
            out[f"adverse_bid_h{h}"] = np.maximum(0.0, -m_bid)
            out[f"adverse_ask_h{h}"] = np.maximum(0.0, -m_ask)
        out[f"markout_available_h{h}"] = available.fillna(False).astype("bool")

    return pd.DataFrame(out)
