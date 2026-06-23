"""Quantile-regression helpers.

The quantile target is just the realised forward return r_{t,h}; the quantile
heads of the multi-task model are trained with the pinball (quantile) loss. The
math lives here so the model, the evaluation layer and the tests all share one
definition.
"""

from __future__ import annotations

import numpy as np


def pinball_loss(r: np.ndarray, q_hat: np.ndarray, tau: float) -> np.ndarray:
    """Per-row pinball loss L_tau(r, q) = max(tau*(r-q), (tau-1)*(r-q))."""
    diff = r - q_hat
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def mean_pinball_loss(r: np.ndarray, q_hat: np.ndarray, tau: float) -> float:
    if len(r) == 0:
        return float("nan")
    return float(np.mean(pinball_loss(np.asarray(r), np.asarray(q_hat), tau)))


def interval_coverage(r: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    """Empirical fraction of realised returns inside [q_lo, q_hi]."""
    if len(r) == 0:
        return float("nan")
    r = np.asarray(r, dtype="float64")
    inside = (r >= np.asarray(q_lo)) & (r <= np.asarray(q_hi))
    return float(np.mean(inside))
