"""The models. Importing this package registers all of them."""

from .base import ForecastModel, get_model_class, register_model, registered_models
from .baselines import ImbalanceRuleModel, NoChangeModel
from .data import ModelData, load_model_data
from .gbm import LightGBMModel
from .linear import LogisticRegressionModel, RidgeRegressionModel
from .prediction import (
    PREDICTION_COLUMNS,
    build_predictions,
    enforce_prediction_schema,
    merge_ridge_sidecar,
)
from .deep import TCNExecMultiTaskModel
from .registry import build_model, load_model_params
from .tcn import TCNModel
from .variants import ModelVariant, experiment_matrix, return_head_variants, ssm_variants

__all__ = [
    # Interface + registry
    "ForecastModel",
    "register_model",
    "get_model_class",
    "registered_models",
    "build_model",
    "load_model_params",
    # Data
    "ModelData",
    "load_model_data",
    # Prediction schema
    "PREDICTION_COLUMNS",
    "build_predictions",
    "enforce_prediction_schema",
    "merge_ridge_sidecar",
    # Variants
    "ModelVariant",
    "experiment_matrix",
    "return_head_variants",
    "ssm_variants",
    # Models
    "NoChangeModel",
    "ImbalanceRuleModel",
    "LogisticRegressionModel",
    "RidgeRegressionModel",
    "LightGBMModel",
    "TCNModel",
    "TCNExecMultiTaskModel",
]
