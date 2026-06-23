"""Two linear models: logistic regression (direction) and ridge (return)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from ._classify import MultiClassWrapper
from .base import ForecastModel, register_model
from .data import ModelData
from .prediction import build_predictions


@register_model
class LogisticRegressionModel(ForecastModel):
    """One 3-class logistic regression per horizon, predicting the direction."""

    name = "logistic_regression"
    version = "1.0"

    def __init__(self, C: float = 1.0, max_iter: int = 1000, **_: object) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.models: dict[int, MultiClassWrapper] = {}

    def _make(self) -> LogisticRegression:
        return LogisticRegression(C=self.C, max_iter=self.max_iter)

    def fit(self, train: ModelData, validation: ModelData, config) -> None:
        X = train.X()
        self.models = {}
        for h in train.horizons:
            avail = train.available(h)
            y = train.true_direction(h)[avail]
            self.models[h] = MultiClassWrapper.fit(self._make, X[avail], y)

    def predict(self, data: ModelData, config, run_id: str = "") -> pd.DataFrame:
        X = data.X()
        proba = {h: self.models[h].proba3(X) for h in data.horizons}
        cls = {h: self.models[h].class3(X) for h in data.horizons}
        return build_predictions(
            run_id=run_id,
            model_name=self.name,
            model_version=self.version,
            split=data.split,
            ids=data.ids(),
            horizons=data.horizons,
            true_return={h: data.true_return(h) for h in data.horizons},
            true_direction={h: data.true_direction(h) for h in data.horizons},
            pred_proba=proba,
            pred_class=cls,
        )

    def hyperparameters(self) -> dict:
        return {"C": self.C, "max_iter": self.max_iter}

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as fh:
            pickle.dump({"models": self.models, "C": self.C, "max_iter": self.max_iter}, fh)

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionModel":
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        model = cls(C=state["C"], max_iter=state["max_iter"])
        model.models = state["models"]
        return model


@register_model
class RidgeRegressionModel(ForecastModel):
    """One ridge regression per horizon, predicting the return."""

    name = "ridge_regression"
    version = "1.0"

    def __init__(self, alpha: float = 1.0, **_: object) -> None:
        self.alpha = float(alpha)
        self.models: dict[int, Ridge | None] = {}
        # if a horizon has no training rows we fall back to a constant
        self.fallback: dict[int, float] = {}

    def fit(self, train: ModelData, validation: ModelData, config) -> None:
        X = train.X()
        self.models = {}
        self.fallback = {}
        for h in train.horizons:
            avail = train.available(h)
            y = train.true_return(h)[avail]
            if len(y) == 0:
                self.models[h] = None
                self.fallback[h] = 0.0
            else:
                ridge = Ridge(alpha=self.alpha)
                ridge.fit(X[avail], y)
                self.models[h] = ridge
                self.fallback[h] = float(np.mean(y))

    def predict(self, data: ModelData, config, run_id: str = "") -> pd.DataFrame:
        X = data.X()
        n = len(X)
        pred_return = {}
        for h in data.horizons:
            model = self.models.get(h)
            if model is not None:
                pred_return[h] = model.predict(X)
            else:
                pred_return[h] = np.full(n, self.fallback.get(h, 0.0))
        return build_predictions(
            run_id=run_id,
            model_name=self.name,
            model_version=self.version,
            split=data.split,
            ids=data.ids(),
            horizons=data.horizons,
            true_return={h: data.true_return(h) for h in data.horizons},
            true_direction={h: data.true_direction(h) for h in data.horizons},
            pred_return=pred_return,
        )

    def hyperparameters(self) -> dict:
        return {"alpha": self.alpha}

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as fh:
            pickle.dump({"models": self.models, "alpha": self.alpha, "fallback": self.fallback}, fh)

    @classmethod
    def load(cls, path: Path) -> "RidgeRegressionModel":
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        model = cls(alpha=state["alpha"])
        model.models = state["models"]
        model.fallback = state["fallback"]
        return model
