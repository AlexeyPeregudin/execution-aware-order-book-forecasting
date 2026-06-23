"""Market-making policy metrics.

Computed from the order / fill / inventory records of a replay. The headline
objective used for validation-only policy selection is total reward (marked
wealth change net of the inventory, turnover and adverse-selection penalties).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown_from_wealth(wealth: np.ndarray) -> float:
    if len(wealth) == 0:
        return 0.0
    peak = np.maximum.accumulate(wealth)
    return float(np.max(peak - wealth))


def compute_policy_metrics(
    orders: pd.DataFrame, fills: pd.DataFrame, inventory: pd.DataFrame
) -> dict[str, float]:
    """Headline metrics for one policy over a set of replayed days."""
    n_dec = len(orders)
    n_bid_q = int(orders["quote_bid"].sum()) if n_dec else 0
    n_ask_q = int(orders["quote_ask"].sum()) if n_dec else 0

    filled = fills[fills["fill_occurred"]] if len(fills) else fills
    bid_fills = filled[filled["side"] == "bid"] if len(filled) else filled
    ask_fills = filled[filled["side"] == "ask"] if len(filled) else filled
    n_fills = len(filled)

    net = filled["net_pnl"].to_numpy() if n_fills else np.zeros(0)
    gross = filled["gross_pnl"].to_numpy() if n_fills else np.zeros(0)
    adverse = filled["adverse_selection_cost"].to_numpy() if n_fills else np.zeros(0)
    # queue-aware partial-fill diagnostics (nullable for the legacy fill models)
    frac = (filled["fill_fraction"].to_numpy() if n_fills and "fill_fraction" in filled.columns
            else np.zeros(0))
    n_partial = int(((frac > 0) & (frac < 1.0 - 1e-9)).sum()) if frac.size else 0

    inv = inventory["inventory"].to_numpy() if len(inventory) else np.zeros(0)
    wealth = inventory["wealth"].to_numpy() if len(inventory) else np.zeros(0)
    reward = orders["reward"].to_numpy() if n_dec else np.zeros(0)

    return {
        "number_of_decisions": float(n_dec),
        "number_of_quotes": float(n_bid_q + n_ask_q),
        "number_of_bid_quotes": float(n_bid_q),
        "number_of_ask_quotes": float(n_ask_q),
        "number_of_fills": float(n_fills),
        "bid_fill_rate": float(len(bid_fills) / n_bid_q) if n_bid_q else float("nan"),
        "ask_fill_rate": float(len(ask_fills) / n_ask_q) if n_ask_q else float("nan"),
        "gross_pnl": float(gross.sum()),
        "net_pnl": float(net.sum()),
        "mean_pnl_per_fill": float(net.mean()) if n_fills else float("nan"),
        "mean_reward": float(reward.mean()) if n_dec else float("nan"),
        "total_reward": float(reward.sum()),
        "turnover": float(np.abs(filled["fill_price"].to_numpy()).sum()) if n_fills else 0.0,
        "average_inventory": float(inv.mean()) if len(inv) else 0.0,
        "max_abs_inventory": float(np.max(np.abs(inv))) if len(inv) else 0.0,
        "inventory_variance": float(inv.var()) if len(inv) else 0.0,
        "max_drawdown": _max_drawdown_from_wealth(wealth),
        "adverse_selection_cost_total": float(adverse.sum()),
        "adverse_selection_cost_per_fill": float(adverse.mean()) if n_fills else float("nan"),
        "average_fill_fraction": float(frac.mean()) if frac.size else float("nan"),
        "number_of_partial_fills": float(n_partial),
    }


def metrics_long(
    metrics: dict[str, float], *, run_id: str, fold_id: int, model_name: str,
    policy_name: str, split: str, monthly_date: str, created_at: str,
) -> list[dict]:
    """Flatten a metrics dict into long metric rows."""
    rows = []
    for name, value in metrics.items():
        rows.append({
            "run_id": run_id, "fold_id": fold_id, "model_name": model_name,
            "policy_name": policy_name, "split": split, "monthly_date": monthly_date,
            "metric_name": name, "metric_value": float(value), "created_at_utc": created_at,
        })
    return rows
