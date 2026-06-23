"""The model interface every model implements, plus a small registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .data import ModelData


class ForecastModel(ABC):
    """Base class for all the models. They all do fit / predict / save / load."""

    name: str = "base"
    version: str = "1.0"
    # tabular models read scaled features; the TCN reads sequence windows
    requires_sequences: bool = False

    @abstractmethod
    def fit(self, train: ModelData, validation: ModelData, config) -> None:
        """Fit on the training data. Validation is only for picking thresholds."""

    @abstractmethod
    def predict(self, data: ModelData, config, run_id: str = "") -> pd.DataFrame:
        """Predict, returning rows in the common prediction format."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the fitted model."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "ForecastModel":
        """Load a model saved with save()."""

    def hyperparameters(self) -> dict:
        """Hyperparameters to put in the training log. Empty by default."""
        return {}


_REGISTRY: dict[str, type[ForecastModel]] = {}


def register_model(cls: type[ForecastModel]) -> type[ForecastModel]:
    """Decorator that adds a model class to the registry under its name."""
    _REGISTRY[cls.name] = cls
    return cls


def get_model_class(name: str) -> type[ForecastModel]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered_models() -> list[str]:
    return sorted(_REGISTRY)
