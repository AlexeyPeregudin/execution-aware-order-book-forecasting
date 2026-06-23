"""The metric functions. Each takes arrays and returns a number, so they're
easy to test against values worked out by hand."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix as _sk_confusion
from sklearn.metrics import log_loss

CLASS_LABELS = [-1, 0, 1]  # the proba columns are in this order: down, neutral, up


def _normalise_proba(proba: np.ndarray) -> np.ndarray:
    """Make each row sum to exactly 1 (float32 softmax can drift a touch)."""
    proba = np.asarray(proba, dtype="float64")
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return proba / row_sums


# classification metrics


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float("nan") if len(y_true) == 0 else float(accuracy_score(y_true, y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    # small groups may be missing a class; sklearn warns, but that's fine here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return float(balanced_accuracy_score(y_true, y_pred))


def cross_entropy(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass log-loss. proba columns are down, neutral, up."""
    if len(y_true) == 0:
        return float("nan")
    return float(log_loss(y_true, _normalise_proba(proba), labels=CLASS_LABELS))


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared distance from the predicted probabilities to the one-hot truth."""
    if len(y_true) == 0:
        return float("nan")
    onehot = np.zeros_like(proba, dtype="float64")
    col = {-1: 0, 0: 1, 1: 2}
    for i, c in enumerate(y_true):
        onehot[i, col[int(c)]] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """3x3 confusion matrix for classes [-1, 0, 1]. Rows are true, cols are predicted."""
    return _sk_confusion(y_true, y_pred, labels=CLASS_LABELS)


# regression metrics


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float("nan") if len(y_true) == 0 else float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float("nan") if len(y_true) == 0 else float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray, train_mean: float) -> float:
    """Out-of-sample R2, comparing the model against just predicting the train mean.

    R2 = 1 - sum((r - r_hat)^2) / sum((r - train_mean)^2)
    """
    if len(y_true) == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - train_mean) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman correlation between true and predicted returns (no scipy needed)."""
    if len(y_true) < 2:
        return float("nan")
    a = pd.Series(y_true).rank()
    b = pd.Series(y_pred).rank()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(a.corr(b))  # pearson on the ranks is the same as spearman


def sign_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Correlation between the signs of true and predicted returns."""
    if len(y_true) < 2:
        return float("nan")
    a = np.sign(np.asarray(y_true, dtype="float64"))
    b = np.sign(np.asarray(y_pred, dtype="float64"))
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error: |confidence - accuracy| averaged over bins."""
    if len(y_true) == 0:
        return float("nan")
    proba = _normalise_proba(proba)
    confidence = proba.max(axis=1)
    correct = (np.asarray(y_pred) == np.asarray(y_true)).astype("float64")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        ece += (sel.sum() / n) * abs(confidence[sel].mean() - correct[sel].mean())
    return float(ece)
