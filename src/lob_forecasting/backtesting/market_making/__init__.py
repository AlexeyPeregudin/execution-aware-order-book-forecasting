"""Passive market-making simulator for top-5 BTCUSDT monthly snapshots.

An offline historical-replay environment with conservative book-path fill
approximations, inventory/adverse-selection accounting, deterministic policies
and a small learned contextual-bandit policy. All policy parameters are selected
on validation folds only; nothing is tuned on test months.
"""

from .accounting import Portfolio
from .environment import (
    ACTION_BID, ACTION_ASK, ACTION_BOTH, ACTION_NEITHER,
    MarketMakingEnv, MarketArrays, simulate_day,
)
from .fills import infer_tick_size, resolve_fill
from .metrics import compute_policy_metrics
from .policies import build_policy, deterministic_policies

__all__ = [
    "Portfolio",
    "MarketMakingEnv",
    "MarketArrays",
    "simulate_day",
    "ACTION_NEITHER", "ACTION_BID", "ACTION_ASK", "ACTION_BOTH",
    "resolve_fill",
    "infer_tick_size",
    "compute_policy_metrics",
    "build_policy",
    "deterministic_policies",
]
