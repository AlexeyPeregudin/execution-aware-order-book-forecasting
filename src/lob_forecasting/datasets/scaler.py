"""A simple standard scaler we fit on the training rows.

Kept small on purpose so it pickles cleanly and is easy to reproduce. Mean uses
1/N, std uses 1/(N-1).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class FeatureScaler:
    """Standardises each column: z = (x - mean) / scale.

    mean_ and scale_ are fit from the training rows and then used as-is on
    validation and test. A constant column has scale 0, which we replace with
    1.0 so we don't divide by zero (it just maps to 0).
    """

    def __init__(self, feature_names: list[str], mean: np.ndarray, scale: np.ndarray) -> None:
        self.feature_names = list(feature_names)
        self.mean_ = np.asarray(mean, dtype="float64")
        self.scale_ = np.asarray(scale, dtype="float64")

    @classmethod
    def fit(cls, df: pd.DataFrame, feature_names: list[str]) -> "FeatureScaler":
        x = df[feature_names].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore"):
            if x.shape[0]:
                mean = np.nanmean(x, axis=0)
            else:
                mean = np.zeros(len(feature_names))
            if x.shape[0] > 1:
                std = np.nanstd(x, axis=0, ddof=1)
            else:
                std = np.zeros(len(feature_names))
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=0.0)
        std[std == 0.0] = 1.0
        return cls(feature_names, mean, std)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_names].to_numpy(dtype="float64")
        return (x - self.mean_) / self.scale_

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Copy df and standardise its feature columns in place."""
        out = df.copy()
        out[self.feature_names] = self.transform(df)
        return out

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            pickle.dump(self, fh)
        return out

    @staticmethod
    def load(path: str | Path) -> "FeatureScaler":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)
