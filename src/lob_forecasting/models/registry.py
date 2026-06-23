"""Build a model by name, reading its hyperparameters from configs/model/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import get_model_class

# model name -> yaml file stem under configs/model/
MODEL_CONFIG_FILES: dict[str, str] = {
    "logistic_regression": "logistic_regression",
    "ridge_regression": "ridge",
    "lightgbm": "lightgbm",
    "tcn_small": "tcn_small",
    "tcn_exec_multitask": "tcn_exec_multitask",
}


def load_model_params(name: str, model_config_dir: str | Path | None) -> dict[str, Any]:
    """Read a model's hyperparameters from its yaml file, or {} if there isn't one."""
    if model_config_dir is None:
        return {}
    stem = MODEL_CONFIG_FILES.get(name)
    if stem is None:
        return {}
    path = Path(model_config_dir) / f"{stem}.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_model(
    name: str,
    model_config_dir: str | Path | None = "configs/model",
    overrides: dict[str, Any] | None = None,
):
    """Make a model. yaml params first, then anything in overrides on top."""
    params = load_model_params(name, model_config_dir)
    if overrides:
        params.update(overrides)
    cls = get_model_class(name)
    return cls(**params)
