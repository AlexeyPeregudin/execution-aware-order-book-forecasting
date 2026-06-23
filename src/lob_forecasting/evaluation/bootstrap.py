"""Block bootstrap confidence intervals.

Event-time metrics are autocorrelated, so an i.i.d. bootstrap understates the
uncertainty. We resample contiguous blocks of rows (preserving short-range
dependence), recompute the metric, and report empirical quantiles. Used for the
headline per-test-month metrics: accuracy, rank_ic, R2_oos, net_pnl,
max_drawdown.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..config import BlockBootstrapConfig


def block_bootstrap_ci(
    data: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    *,
    block_size: int,
    n_bootstrap: int,
    confidence_level: float,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap a scalar metric by resampling contiguous blocks of rows.

    Returns the point estimate plus the empirical lower/upper bounds.
    """
    n = len(data)
    point = float(metric_fn(data)) if n > 0 else float("nan")
    if n == 0 or n < block_size:
        return {"point": point, "lower": float("nan"), "upper": float("nan"), "n": n}

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size
    samples: list[float] = []
    values = data.reset_index(drop=True)
    for _ in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        boot = values.iloc[idx]
        val = metric_fn(boot)
        if val is not None and np.isfinite(val):
            samples.append(float(val))
    if not samples:
        return {"point": point, "lower": float("nan"), "upper": float("nan"), "n": n}

    alpha = 1.0 - confidence_level
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return {"point": point, "lower": lo, "upper": hi, "n": n}


def bootstrap_from_config(
    data: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    cfg: BlockBootstrapConfig,
    seed: int = 0,
) -> dict[str, float]:
    return block_bootstrap_ci(
        data,
        metric_fn,
        block_size=cfg.block_size_events,
        n_bootstrap=cfg.n_bootstrap,
        confidence_level=cfg.confidence_level,
        seed=seed,
    )
