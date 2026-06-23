"""The two baselines: predict no change, and a rule on best-level imbalance."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ._classify import onehot
from .base import ForecastModel, register_model
from .data import ModelData
from .prediction import build_predictions


@register_model
class NoChangeModel(ForecastModel):
    """Always predicts return 0 and direction neutral."""

    name = "no_change"
    version = "1.0"

    def fit(self, train: ModelData, validation: ModelData, config) -> None:
        return None

    def predict(self, data: ModelData, config, run_id: str = "") -> pd.DataFrame:
        ids = data.ids()
        n = len(ids)
        neutral_proba = np.column_stack([np.zeros(n), np.ones(n), np.zeros(n)])
        return build_predictions(
            run_id=run_id,
            model_name=self.name,
            model_version=self.version,
            split=data.split,
            ids=ids,
            horizons=data.horizons,
            true_return={h: data.true_return(h) for h in data.horizons},
            true_direction={h: data.true_direction(h) for h in data.horizons},
            pred_return={h: np.zeros(n) for h in data.horizons},
            pred_proba={h: neutral_proba for h in data.horizons},
            pred_class={h: np.zeros(n) for h in data.horizons},
        )

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as fh:
            pickle.dump({"version": self.version}, fh)

    @classmethod
    def load(cls, path: Path) -> "NoChangeModel":
        return cls()


@register_model
class ImbalanceRuleModel(ForecastModel):
    """Trade on best-level imbalance: buy if it's above gamma, sell if below -gamma.

    gamma is picked on the validation set.
    """

    name = "imbalance_rule"
    version = "1.0"

    def __init__(self, gamma: float = 0.0, **_: object) -> None:
        self.gamma = float(gamma)

    @staticmethod
    def _classify(imbalance: np.ndarray, gamma: float) -> np.ndarray:
        return np.where(imbalance > gamma, 1, np.where(imbalance < -gamma, -1, 0)).astype("float64")

    def fit(self, train: ModelData, validation: ModelData, config) -> None:
        # try a grid of gammas on validation, keep the one with the best average
        # accuracy across horizons
        imb = validation.raw_feature("imbalance_l1")
        grid = np.linspace(0.0, 0.9, 19)
        best_gamma = 0.0
        best_score = -np.inf
        for g in grid:
            pred = self._classify(imb, g)
            scores = []
            for h in validation.horizons:
                avail = validation.available(h)
                if not avail.any():
                    continue
                truth = validation.true_direction(h)[avail]
                scores.append(float((pred[avail] == truth).mean()))
            score = float(np.mean(scores)) if scores else -np.inf
            if score > best_score:
                best_score = score
                best_gamma = float(g)
        self.gamma = best_gamma

    def predict(self, data: ModelData, config, run_id: str = "") -> pd.DataFrame:
        imb = data.raw_feature("imbalance_l1")
        cls = self._classify(imb, self.gamma)
        proba = onehot(cls)
        return build_predictions(
            run_id=run_id,
            model_name=self.name,
            model_version=self.version,
            split=data.split,
            ids=data.ids(),
            horizons=data.horizons,
            true_return={h: data.true_return(h) for h in data.horizons},
            true_direction={h: data.true_direction(h) for h in data.horizons},
            pred_proba={h: proba for h in data.horizons},
            pred_class={h: cls for h in data.horizons},
        )

    def hyperparameters(self) -> dict:
        return {"gamma": self.gamma}

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as fh:
            pickle.dump({"gamma": self.gamma, "version": self.version}, fh)

    @classmethod
    def load(cls, path: Path) -> "ImbalanceRuleModel":
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        return cls(gamma=state["gamma"])
