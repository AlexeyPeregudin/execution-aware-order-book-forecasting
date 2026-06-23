"""Label math: forward returns, direction labels, and fitting the threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_thresholds(
    relative_spread_train: np.ndarray,
    alpha: float,
    horizon_list: list[int],
) -> tuple[dict[str, float], float]:
    """Threshold = alpha * median(relative_spread) over the training rows.

    The median doesn't depend on the horizon, so every horizon gets the same
    threshold. We still store one per horizon in case that changes later.
    Returns (thresholds, median).
    """
    finite = relative_spread_train[np.isfinite(relative_spread_train)]
    median = float(np.median(finite)) if finite.size else 0.0
    if not np.isfinite(median):
        median = 0.0
    eps = alpha * median
    thresholds = {f"h{h}": float(eps) for h in horizon_list}
    return thresholds, median


def compute_labels(
    mid: pd.Series,
    horizon_list: list[int],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Make the labels for each horizon h.

    r_h{h} is the forward return (mid_{t+h} - mid_t) / mid_t. y_dir_h{h} is its
    sign with a dead band: +1 above eps, -1 below -eps, 0 in between. The last h
    rows have no future mid, so the label is marked unavailable.
    """
    mid = mid.astype("float64")
    out: dict[str, pd.Series] = {}

    for h in horizon_list:
        future = mid.shift(-h)  # mid at t+h; the last h rows are NaN
        r = (future - mid) / mid
        available = future.notna() & mid.notna() & (mid != 0)
        eps = thresholds[f"h{h}"]

        direction = np.select([r > eps, r < -eps], [1, -1], default=0).astype("float64")
        y = pd.Series(direction, index=mid.index).where(available)

        out[f"r_h{h}"] = r.where(available)
        out[f"y_dir_h{h}"] = y.astype("Int64")
        out[f"label_available_h{h}"] = available.fillna(False).astype("bool")

    # order: all r_h*, then all y_dir_h*, then all label_available_h*
    ordered = (
        [f"r_h{h}" for h in horizon_list]
        + [f"y_dir_h{h}" for h in horizon_list]
        + [f"label_available_h{h}" for h in horizon_list]
    )
    return pd.DataFrame({c: out[c] for c in ordered})


def label_distribution(
    labels_df: pd.DataFrame, horizon_list: list[int]
) -> dict[str, dict[str, int]]:
    """Count available / up / neutral / down per horizon (for the report)."""
    dist: dict[str, dict[str, int]] = {}
    for h in horizon_list:
        y = labels_df[f"y_dir_h{h}"]
        avail = labels_df[f"label_available_h{h}"]
        dist[f"h{h}"] = {
            "available": int(avail.sum()),
            "up": int((y == 1).sum()),
            "neutral": int((y == 0).sum()),
            "down": int((y == -1).sum()),
        }
    return dist
