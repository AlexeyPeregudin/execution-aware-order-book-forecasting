"""Composite validation score for the multi-task execution model.

Early stopping based on direction accuracy alone can restore a checkpoint where
the slower-moving quantile / markout / adverse heads are under-trained, even
though those heads are the execution-relevant ones. This module computes a single
scalar that balances direction skill against calibration and execution-head
accuracy, on validation rows only, using scale constants fitted on training rows.

The score per horizon h is

    S_h =  0.25 BA_h            (balanced direction accuracy)
         + 0.15 RIC_h           (rank-IC of the median quantile)
         - 0.20 QL_h^norm       (pinball loss / return scale)
         - 0.15 CErr_h          (|coverage90 - 0.90|)
         - 0.05 MW_h^norm       (interval width / return scale)
         - 0.10 MAE^M_h,norm    (markout MAE / markout scale)
         - 0.10 MAE^A_h,norm    (adverse MAE / adverse scale)

and S = mean_h S_h. When a head is disabled its term is dropped and the remaining
weights are renormalised so the score stays on a comparable scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_EPS = 1e-9

# default term weights, keyed by the head/diagnostic they belong to
TERM_WEIGHTS: dict[str, float] = {
    "balanced_accuracy": 0.25,   # direction head
    "rank_ic": 0.15,             # direction/return ordering (uses q50)
    "quantile_loss": 0.20,       # quantile head (penalty)
    "coverage_error": 0.15,      # quantile head (penalty)
    "interval_width": 0.05,      # quantile head (penalty)
    "markout_mae": 0.10,         # markout head (penalty)
    "adverse_mae": 0.10,         # adverse head (penalty)
}
# which terms count as a penalty (subtracted) vs a reward (added)
_PENALTY_TERMS = {"quantile_loss", "coverage_error", "interval_width", "markout_mae", "adverse_mae"}
# cap a normalised penalty term so one exploding ratio (e.g. a wide interval over a
# near-zero return scale on an under-trained head) can't dominate checkpoint
# selection; well above any sane converged value, so it only clips pathologies.
_PENALTY_CAP = 100.0
# which model head each term depends on (for disable/renormalise)
_TERM_HEAD = {
    "balanced_accuracy": "direction",
    "rank_ic": "direction",
    "quantile_loss": "quantile",
    "coverage_error": "quantile",
    "interval_width": "quantile",
    "markout_mae": "markout",
    "adverse_mae": "adverse",
}
_QUANTILE_TAUS = (0.05, 0.50, 0.95)


@dataclass
class HorizonScales:
    """Per-horizon scale constants (median absolute deviation), training rows only."""

    ret: float
    markout: float
    adverse: float


def mad(values: np.ndarray) -> float:
    """Median absolute deviation about the median (robust scale)."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    med = float(np.median(v))
    return float(np.median(np.abs(v - med)))


def fit_scales(
    horizons: list[int],
    true_return: dict[int, np.ndarray],
    markout_bid: dict[int, np.ndarray],
    markout_ask: dict[int, np.ndarray],
    adverse_bid: dict[int, np.ndarray],
    adverse_ask: dict[int, np.ndarray],
) -> dict[int, HorizonScales]:
    """Scale constants per horizon from training-row targets."""
    out: dict[int, HorizonScales] = {}
    for h in horizons:
        s_r = mad(true_return[h]) + _EPS
        s_m = mad(markout_bid[h]) + mad(markout_ask[h]) + _EPS
        s_a = mad(adverse_bid[h]) + mad(adverse_ask[h]) + _EPS
        out[h] = HorizonScales(ret=s_r, markout=s_m, adverse=s_a)
    return out


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = (-1, 0, 1)
    recalls = []
    for c in classes:
        m = y_true == c
        if m.sum() > 0:
            recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) < _EPS or np.std(rb) < _EPS:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _pinball(r: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    diff = r - q
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def horizon_components(
    sub: pd.DataFrame, scales: HorizonScales
) -> dict[str, float]:
    """Raw (un-weighted) diagnostic terms for one horizon's validation rows."""
    comp: dict[str, float] = {}
    cls = sub[sub["true_direction"].notna() & sub["pred_class"].notna()]
    if len(cls):
        yt = cls["true_direction"].to_numpy().astype(int)
        yp = cls["pred_class"].to_numpy().astype(int)
        comp["balanced_accuracy"] = _balanced_accuracy(yt, yp)

    q = sub[sub["true_return"].notna() & sub["pred_q50"].notna()]
    if len(q):
        r = q["true_return"].to_numpy()
        q50 = q["pred_q50"].to_numpy()
        comp["rank_ic"] = _rank_ic(r, q50)
        q05 = q["pred_q05"].to_numpy()
        q95 = q["pred_q95"].to_numpy()
        ql = np.mean([_pinball(r, q[col].to_numpy(), tau).mean()
                      for tau, col in zip(_QUANTILE_TAUS, ("pred_q05", "pred_q50", "pred_q95"))])
        comp["quantile_loss"] = float(ql) / scales.ret
        cover = float(((q05 <= r) & (r <= q95)).mean())
        comp["coverage_error"] = abs(cover - 0.90)
        comp["interval_width"] = float(np.mean(q95 - q05)) / scales.ret

    mk = sub[sub["true_markout_bid"].notna() & sub["pred_markout_bid"].notna()]
    if len(mk):
        mae_m = (np.abs(mk["true_markout_bid"].to_numpy() - mk["pred_markout_bid"].to_numpy()).mean()
                 + np.abs(mk["true_markout_ask"].to_numpy() - mk["pred_markout_ask"].to_numpy()).mean())
        comp["markout_mae"] = float(mae_m) / scales.markout
    adv = sub[sub["true_adverse_bid"].notna() & sub["pred_adverse_bid"].notna()]
    if len(adv):
        mae_a = (np.abs(adv["true_adverse_bid"].to_numpy() - adv["pred_adverse_bid"].to_numpy()).mean()
                 + np.abs(adv["true_adverse_ask"].to_numpy() - adv["pred_adverse_ask"].to_numpy()).mean())
        comp["adverse_mae"] = float(mae_a) / scales.adverse
    return comp


def _effective_weights(enabled_heads: set[str]) -> dict[str, float]:
    """Drop terms whose head is disabled and renormalise the remaining weights."""
    kept = {t: w for t, w in TERM_WEIGHTS.items() if _TERM_HEAD[t] in enabled_heads}
    total = sum(kept.values())
    if total <= 0:
        return {}
    return {t: w / total for t, w in kept.items()}


def composite_score(
    predictions: pd.DataFrame,
    *,
    horizons: list[int],
    scales: dict[int, HorizonScales],
    enabled_heads: set[str],
) -> tuple[float, dict[str, float]]:
    """Composite validation score and a flat dict of diagnostic components.

    `predictions` should be the validation-split prediction frame produced by the
    model. `enabled_heads` is the set of head names that are active (e.g.
    {"direction", "quantile", "markout", "adverse"}); disabled heads' terms are
    removed and the weights renormalised. Returns (score, components).
    """
    weights = _effective_weights(enabled_heads)
    if not weights:
        return 0.0, {}
    per_h_scores: list[float] = []
    agg: dict[str, list[float]] = {}
    for h in horizons:
        sub = predictions[predictions["horizon"] == h]
        if sub.empty:
            continue
        comp = horizon_components(sub, scales[h])
        s = 0.0
        for term, w in weights.items():
            if term not in comp:
                continue
            val = comp[term]
            if term in _PENALTY_TERMS:
                s += -w * min(val, _PENALTY_CAP)
            else:
                s += w * val
            agg.setdefault(term, []).append(val)
        per_h_scores.append(s)
    score = float(np.mean(per_h_scores)) if per_h_scores else 0.0
    components = {f"val_{k}_mean": float(np.mean(v)) for k, v in agg.items()}
    # expose per-horizon coverage explicitly so it shows up in the logs
    for h in horizons:
        sub = predictions[predictions["horizon"] == h]
        q = sub[sub["true_return"].notna() & sub["pred_q05"].notna()] if not sub.empty else sub
        if len(q):
            r = q["true_return"].to_numpy()
            cover = float(((q["pred_q05"].to_numpy() <= r) & (r <= q["pred_q95"].to_numpy())).mean())
            components[f"val_coverage90_h{h}"] = cover
    return score, components
