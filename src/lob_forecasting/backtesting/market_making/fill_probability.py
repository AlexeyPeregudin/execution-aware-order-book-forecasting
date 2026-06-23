"""Ex-ante fill-probability model for the control optimiser.

Before quoting, the control policy needs the probability that a resting quote at
a given side/level fills over the holding horizon. We learn that per fold on
training months only: replay candidate quotes through the queue-aware fill model
to get (features, filled?) examples, then fit a binary classifier (LightGBM if
available, else logistic regression) with optional isotonic calibration on the
validation months.

Features are market-only (no forecasts): the microstructure / regime / latent
state vector at the decision event, plus the quote side, quote level and the
displayed depth at the quoted level. At decision time the model only sees what is
known then, so it stays causal.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .queue_fill import resolve_queue_fill

FILL_PROB_VERSION = "mm_fill_prob_v1"


def _make_classifier(kind: str):
    if kind == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  verbose=-1), "lightgbm"
        except Exception:  # pragma: no cover - lightgbm optional
            pass
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=500), "logistic_regression"


@dataclass
class FillProbabilityModel:
    """A fitted ex-ante fill-probability classifier (optionally calibrated)."""

    classifier: object
    classifier_kind: str
    calibrator: object | None = None
    constant_prob: float | None = None  # used when training saw a single class
    feature_dim: int = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.nan_to_num(np.asarray(X, dtype="float64")))
        if self.constant_prob is not None:
            return np.full(X.shape[0], self.constant_prob)
        p = self.classifier.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 0.0, 1.0)

    def predict_one(self, x: np.ndarray) -> float:
        return float(self.predict(np.asarray(x).reshape(1, -1))[0])


def build_fill_examples(
    day_arrays_list: list, *, horizon: int, quote_size: float, levels: list[int],
    queue_kwargs: dict, decision_interval: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay candidate quotes through the queue model to label fills (train days)."""
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    qk = queue_kwargs or {}
    for arrays in day_arrays_list:
        if not arrays.has_levels():
            continue
        n = arrays.n
        for t in range(0, n, max(1, decision_interval)):
            if not np.isfinite(arrays.mid[t]):
                continue
            for side, level in product(("bid", "ask"), levels):
                r = resolve_queue_fill(
                    side, level, t, horizon,
                    bid_px=arrays.bid_px_levels, bid_qty=arrays.bid_qty_levels,
                    ask_px=arrays.ask_px_levels, ask_qty=arrays.ask_qty_levels,
                    order_size=quote_size, tick=arrays.tick_size,
                    queue_position=qk.get("queue_position", "back"),
                    kappa=float(qk.get("kappa", 0.5)),
                    full_cross_fill=bool(qk.get("full_cross_fill", True)),
                )
                X_rows.append(arrays.fill_feature_row(t, side, level))
                y_rows.append(1 if (r.filled and r.fill_qty > 0) else 0)
    if not X_rows:
        return np.zeros((0, 1)), np.zeros(0)
    return np.vstack(X_rows), np.asarray(y_rows, dtype="int64")


def fit_fill_probability(
    train_day_arrays: list, val_day_arrays: list, *, horizon: int, quote_size: float,
    levels: list[int], queue_kwargs: dict, decision_interval: int,
    classifier_kind: str = "lightgbm",
) -> FillProbabilityModel:
    """Fit the per-fold fill-probability model on training days."""
    X, y = build_fill_examples(
        train_day_arrays, horizon=horizon, quote_size=quote_size, levels=levels,
        queue_kwargs=queue_kwargs, decision_interval=decision_interval)
    if len(X) == 0:
        return FillProbabilityModel(classifier=None, classifier_kind="constant",
                                    constant_prob=0.0, feature_dim=1)
    if len(np.unique(y)) < 2:
        return FillProbabilityModel(classifier=None, classifier_kind="constant",
                                    constant_prob=float(y.mean()), feature_dim=X.shape[1])
    clf, kind = _make_classifier(classifier_kind)
    clf.fit(X, y)
    model = FillProbabilityModel(classifier=clf, classifier_kind=kind, feature_dim=X.shape[1])

    # isotonic calibration on validation months, when there are enough positives
    Xv, yv = build_fill_examples(
        val_day_arrays, horizon=horizon, quote_size=quote_size, levels=levels,
        queue_kwargs=queue_kwargs, decision_interval=decision_interval)
    if len(Xv) > 50 and 0 < int(yv.sum()) < len(yv):
        try:
            from sklearn.isotonic import IsotonicRegression

            raw = clf.predict_proba(Xv)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw, yv)
            model.calibrator = iso
        except Exception:  # pragma: no cover - defensive
            model.calibrator = None
    return model
