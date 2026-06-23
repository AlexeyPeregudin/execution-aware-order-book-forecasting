"""Small helper to get 3-class {-1, 0, 1} probabilities out of a classifier."""

from __future__ import annotations

from typing import Callable

import numpy as np

CLASSES = np.array([-1, 0, 1])
_CLASS_TO_COL = {-1: 0, 0: 1, 1: 2}


class MultiClassWrapper:
    """Wraps an sklearn classifier so it always gives back an (n, 3) probability.

    Handles the awkward cases: an empty training set just predicts neutral, and
    a single-class set predicts that one class. Without this, something like
    LogisticRegression would raise when it sees fewer than two classes.
    """

    def __init__(self, estimator=None, const_class: int | None = None) -> None:
        self.estimator = estimator
        self.const_class = const_class

    @classmethod
    def fit(cls, make_estimator: Callable[[], object], X: np.ndarray, y: np.ndarray) -> "MultiClassWrapper":
        y = np.asarray(y)
        if len(X) == 0:
            return cls(const_class=0)
        classes = np.unique(y)
        if len(classes) < 2:
            return cls(const_class=int(classes[0]))
        est = make_estimator()
        est.fit(X, y.astype(int))
        return cls(estimator=est)

    def proba3(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        out = np.zeros((n, 3), dtype="float64")
        if self.estimator is None:
            out[:, _CLASS_TO_COL[self.const_class]] = 1.0
            return out
        # the estimator may not have seen all three classes, so map each of its
        # columns to the right place and leave the missing ones at 0
        proba = self.estimator.predict_proba(X)
        for j, c in enumerate(self.estimator.classes_):
            out[:, _CLASS_TO_COL[int(c)]] = proba[:, j]
        return out

    def class3(self, X: np.ndarray) -> np.ndarray:
        return CLASSES[self.proba3(X).argmax(axis=1)].astype("float64")


def onehot(classes: np.ndarray) -> np.ndarray:
    """Turn {-1,0,1} labels into (n, 3) one-hot rows."""
    out = np.zeros((len(classes), 3), dtype="float64")
    for i, c in enumerate(classes):
        out[i, _CLASS_TO_COL[int(c)]] = 1.0
    return out
