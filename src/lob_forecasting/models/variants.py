"""Execution-aware multi-task TCN variants.

The experiment matrix runs several variants of the one `tcn_exec_multitask`
class that differ only in how the point-return head is wired and whether the
latent state-space context is appended. Rather than duplicate a near-identical
YAML per variant, each variant is a set of overrides layered on top of
`configs/model/tcn_exec_multitask.yaml`; the driver trains each with its own
`output_name` so the prediction files stay separate.

Each ModelVariant carries:
  - output_name: the prediction-file / model label;
  - overrides: params merged over the base model yaml;
  - needs_ssm: whether the latent state-space context must be appended;
  - return_source: how pred_return is produced (mirrors the override, surfaced
    for the driver's ridge-sidecar merge).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# execution-head loss weights shared by every variant; only the return head's
# wiring changes between variants.
_BASE_HEADS = {
    "direction_head": {"enabled": True, "loss_weight": 1.0},
    "quantile_head": {"enabled": True, "loss_weight": 1.5},
    "markout_head": {"enabled": True, "loss_weight": 0.75},
    "adverse_head": {"enabled": True, "loss_weight": 0.50},
}

# the two-phase schedule applies to all multi-task variants
_TWO_PHASE = {
    "enabled": True,
    "phase_a": {"epochs": 30, "min_epochs": 12, "patience": 8,
                "learning_rate": 5e-4, "weight_decay": 5e-4},
    "phase_b": {"epochs": 8, "learning_rate": 1e-4, "weight_decay": 1e-5,
                "weights": {"direction": 0.0, "quantile": 2.0, "markout": 1.0, "adverse": 1.0}},
}


def _heads(return_head: dict) -> dict:
    return {"return_head": return_head, **_BASE_HEADS}


@dataclass(frozen=True)
class ModelVariant:
    output_name: str
    base_model: str = "tcn_exec_multitask"
    overrides: dict = field(default_factory=dict)
    needs_ssm: bool = False
    return_source: str = "neural_head"
    ridge_model_name: str = "ridge_regression"


def _common(two_phase: bool) -> dict:
    return {"selection_metric": "composite", "two_phase": dict(_TWO_PHASE) if two_phase else {"enabled": False}}


def return_head_variants(*, two_phase: bool = True, suffix: str = "") -> list[ModelVariant]:
    """The four return-head ablations of the multi-task model."""
    def name(stem: str) -> str:
        return f"{stem}{suffix}"

    variants = [
        ModelVariant(
            output_name=name("tcn_exec_base"),
            overrides={**_common(two_phase),
                       "execution_heads": _heads({"enabled": True, "loss_weight": 0.10,
                                                  "detach_from_encoder": False,
                                                  "prediction_source": "neural_head"})},
            return_source="neural_head",
        ),
        ModelVariant(
            output_name=name("tcn_exec_ret0"),
            overrides={**_common(two_phase),
                       "execution_heads": _heads({"enabled": False, "loss_weight": 0.0,
                                                  "detach_from_encoder": False,
                                                  "prediction_source": "none"})},
            return_source="none",
        ),
        ModelVariant(
            output_name=name("tcn_exec_ret_detached"),
            overrides={**_common(two_phase),
                       "execution_heads": _heads({"enabled": True, "loss_weight": 0.10,
                                                  "detach_from_encoder": True,
                                                  "prediction_source": "neural_head"})},
            return_source="neural_head",
        ),
        ModelVariant(
            output_name=name("tcn_exec_ret0_ridge_sidecar"),
            overrides={**_common(two_phase),
                       "execution_heads": _heads({"enabled": False, "loss_weight": 0.0,
                                                  "detach_from_encoder": False,
                                                  "prediction_source": "ridge_sidecar",
                                                  "ridge_model_name": "ridge_regression"})},
            return_source="ridge_sidecar",
        ),
    ]
    return variants


def ssm_variants(*, two_phase: bool = True, suffix: str = "") -> list[ModelVariant]:
    """Latent-SSM-context variants paired with their no-SSM counterparts."""
    def name(stem: str) -> str:
        return f"{stem}{suffix}"

    return [
        ModelVariant(
            output_name=name("tcn_exec_ret0_ssm"),
            overrides={**_common(two_phase),
                       "execution_heads": _heads({"enabled": False, "loss_weight": 0.0,
                                                  "detach_from_encoder": False,
                                                  "prediction_source": "none"})},
            needs_ssm=True,
            return_source="none",
        ),
        ModelVariant(
            output_name=name("tcn_exec_ret0_ssm_ridge_sidecar"),
            overrides={**_common(two_phase),
                       "execution_heads": _heads({"enabled": False, "loss_weight": 0.0,
                                                  "detach_from_encoder": False,
                                                  "prediction_source": "ridge_sidecar",
                                                  "ridge_model_name": "ridge_regression"})},
            needs_ssm=True,
            return_source="ridge_sidecar",
        ),
    ]


def experiment_matrix(*, include_ssm: bool, two_phase: bool = True) -> list[ModelVariant]:
    """Full multi-task variant list for the 12-month experiment."""
    variants = return_head_variants(two_phase=two_phase)
    if include_ssm:
        variants += ssm_variants(two_phase=two_phase)
    return variants
