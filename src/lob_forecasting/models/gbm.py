"""LightGBM: one 3-class direction classifier per horizon."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier

from ._classify import MultiClassWrapper
from .base import ForecastModel, register_model
from .data import ModelData
from .prediction import build_predictions

# default hyperparameters
DEFAULT_PARAMS = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


@register_model
class LightGBMModel(ForecastModel):
    """A gradient-boosted direction classifier, one per horizon."""

    name = "lightgbm"
    version = "1.0"

    def __init__(self, **params: object) -> None:
        self.params = {**DEFAULT_PARAMS, **params}
        self.models: dict[int, MultiClassWrapper] = {}

    def _make(self) -> LGBMClassifier:
        return LGBMClassifier(**self.params, verbose=-1)

    def fit(self, train: ModelData, validation: ModelData, config) -> None:
        # set a seed so results are reproducible (only matters if bagging is on)
        self.params.setdefault("random_state", config.random_seed)
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
        return dict(self.params)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as fh:
            pickle.dump({"models": self.models, "params": self.params}, fh)

    @classmethod
    def load(cls, path: Path) -> "LightGBMModel":
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        model = cls(**state["params"])
        model.models = state["models"]
        return model
