"""Acceptance and regression tests for the configuration module."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lob_forecasting.config import (
    ExperimentConfig,
    load_config,
    parse_cli_overrides,
    save_resolved_config,
)

# Minimal valid config dict used across multiple tests

VALID_CONFIG: dict = {
    "data": {
        "venue": "binance",
        "symbols": ["BTCUSDT"],
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "top_k": 10,
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "artefact_dir": "artefacts",
    },
    "sampling": {
        "mode": "event_time",
        "horizons_events": [10, 50, 200],
        "feature_lookbacks_events": [10, 50, 200],
    },
    "labels": {
        "direction_threshold_mode": "train_median_relative_spread",
        "direction_threshold_alpha": 0.5,
    },
    "splits": {
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "embargo_events": 200,
    },
    "features": {
        "include_basic_microstructure": True,
        "include_best_level_ofi": True,
        "include_multilevel_imbalance": True,
        "realised_vol_lookbacks_events": [50, 200],
    },
    "models": {
        "run": [
            "no_change",
            "imbalance_rule",
            "logistic_regression",
            "ridge_regression",
            "lightgbm",
            "tcn_small",
        ]
    },
    "backtest": {
        "horizon": 50,
        "threshold_grid": [0.0, 0.00001, 0.00002, 0.00005, 0.0001],
        "fee_bps": 5.0,
        "latency_events": 1,
        "max_position": 1.0,
        "trade_size": 1.0,
    },
    "random_seed": 42,
}


def _make_config(**overrides) -> dict:
    """Deep-copy VALID_CONFIG and apply top-level key overrides."""
    import copy
    cfg = copy.deepcopy(VALID_CONFIG)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg




class TestValidConfig:
    def test_valid_config_loads(self):
        cfg = ExperimentConfig.model_validate(VALID_CONFIG)
        assert cfg.data.venue == "binance"
        assert cfg.data.top_k == 10
        assert cfg.random_seed == 42

    def test_valid_config_round_trips_to_yaml(self):
        cfg = ExperimentConfig.model_validate(VALID_CONFIG)
        dumped = yaml.dump(cfg.model_dump(mode="json"), default_flow_style=False)
        reloaded = ExperimentConfig.model_validate(yaml.safe_load(dumped))
        assert reloaded == cfg

    def test_mvp_yaml_file_loads(self):
        """The shipped configs/experiment/mvp.yaml must be a valid config."""
        mvp_path = Path(__file__).parent.parent / "configs" / "experiment" / "mvp.yaml"
        cfg, run_id = load_config(mvp_path)
        assert cfg.data.symbols == ["BTCUSDT"]
        assert run_id  # non-empty string




class TestInvalidHorizon:
    def test_zero_horizon_fails(self):
        cfg = _make_config(sampling={"horizons_events": [0, 50, 200], "feature_lookbacks_events": [10, 50, 200]})
        with pytest.raises(ValidationError, match="positive integers"):
            ExperimentConfig.model_validate(cfg)

    def test_negative_horizon_fails(self):
        cfg = _make_config(sampling={"horizons_events": [-10, 50], "feature_lookbacks_events": [10, 50]})
        with pytest.raises(ValidationError, match="positive integers"):
            ExperimentConfig.model_validate(cfg)

    def test_empty_horizons_fails(self):
        cfg = _make_config(sampling={"horizons_events": [], "feature_lookbacks_events": [10]})
        with pytest.raises(ValidationError, match="non-empty"):
            ExperimentConfig.model_validate(cfg)


class TestInvalidSplitFractions:
    def test_fractions_not_summing_to_one_fails(self):
        cfg = _make_config(splits={
            "train_fraction": 0.5,
            "validation_fraction": 0.3,
            "test_fraction": 0.3,
            "embargo_events": 200,
        })
        with pytest.raises(ValidationError, match="equal 1.0"):
            ExperimentConfig.model_validate(cfg)

    def test_fractions_summing_to_one_with_rounding_passes(self):
        # Exact floating-point representation; should still pass tolerance check.
        cfg = _make_config(splits={
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "embargo_events": 200,
        })
        ExperimentConfig.model_validate(cfg)  # must not raise


class TestEmbargoTooSmall:
    def test_embargo_less_than_max_horizon_fails(self):
        cfg = _make_config(splits={
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "embargo_events": 100,  # max horizon is 200
        })
        with pytest.raises(ValidationError, match="embargo_events"):
            ExperimentConfig.model_validate(cfg)

    def test_embargo_equal_to_max_horizon_passes(self):
        cfg = _make_config(splits={
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "embargo_events": 200,
        })
        ExperimentConfig.model_validate(cfg)  # must not raise


# DataConfig invariants


class TestDataConfig:
    def test_top_k_zero_fails(self):
        cfg = _make_config(data={**VALID_CONFIG["data"], "top_k": 0})
        with pytest.raises(ValidationError, match="top_k"):
            ExperimentConfig.model_validate(cfg)

    def test_top_k_negative_fails(self):
        cfg = _make_config(data={**VALID_CONFIG["data"], "top_k": -1})
        with pytest.raises(ValidationError, match="top_k"):
            ExperimentConfig.model_validate(cfg)

    def test_start_after_end_fails(self):
        cfg = _make_config(data={**VALID_CONFIG["data"], "start_date": "2024-06-01", "end_date": "2024-01-01"})
        with pytest.raises(ValidationError, match="start_date"):
            ExperimentConfig.model_validate(cfg)

    def test_absolute_raw_dir_fails(self):
        import sys
        abs_path = "C:/some/path" if sys.platform == "win32" else "/some/path"
        cfg = _make_config(data={**VALID_CONFIG["data"], "raw_dir": abs_path})
        with pytest.raises(ValidationError, match="absolute"):
            ExperimentConfig.model_validate(cfg)

    def test_empty_symbols_fails(self):
        cfg = _make_config(data={**VALID_CONFIG["data"], "symbols": []})
        with pytest.raises(ValidationError, match="symbols"):
            ExperimentConfig.model_validate(cfg)


# Model registry invariants


class TestModelRegistry:
    def test_unknown_model_fails(self):
        cfg = _make_config(models={"run": ["no_change", "magic_model_v99"]})
        with pytest.raises(ValidationError, match="Unknown model"):
            ExperimentConfig.model_validate(cfg)

    def test_all_registered_models_pass(self):
        cfg = ExperimentConfig.model_validate(VALID_CONFIG)
        assert "tcn_small" in cfg.models.run


# Backtest invariants


class TestBacktestConfig:
    def test_empty_threshold_grid_fails(self):
        cfg = _make_config(backtest={**VALID_CONFIG["backtest"], "threshold_grid": []})
        with pytest.raises(ValidationError, match="threshold_grid"):
            ExperimentConfig.model_validate(cfg)

    def test_negative_latency_fails(self):
        cfg = _make_config(backtest={**VALID_CONFIG["backtest"], "latency_events": -1})
        with pytest.raises(ValidationError, match="latency_events"):
            ExperimentConfig.model_validate(cfg)


# load_config and save_resolved_config


class TestLoader:
    def test_load_config_returns_config_and_run_id(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml.dump(VALID_CONFIG), encoding="utf-8")
        cfg, run_id = load_config(yaml_file)
        assert isinstance(cfg, ExperimentConfig)
        assert run_id  # non-empty

    def test_load_config_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")

    def test_load_config_with_dict_override(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml.dump(VALID_CONFIG), encoding="utf-8")
        cfg, _ = load_config(yaml_file, overrides={"random_seed": 99})
        assert cfg.random_seed == 99

    def test_save_resolved_config_writes_file(self, tmp_path):
        cfg = ExperimentConfig.model_validate(VALID_CONFIG)
        run_id = "20240101T000000_abc123"
        out_path = save_resolved_config(cfg, run_id, project_root=tmp_path)
        assert out_path.exists()
        with out_path.open(encoding="utf-8") as fh:
            saved = yaml.safe_load(fh)
        assert saved["_meta"]["run_id"] == run_id
        assert "created_at_utc" in saved["_meta"]

    def test_save_resolved_config_is_valid_yaml(self, tmp_path):
        cfg = ExperimentConfig.model_validate(VALID_CONFIG)
        out_path = save_resolved_config(cfg, "test_run", project_root=tmp_path)
        with out_path.open(encoding="utf-8") as fh:
            saved = yaml.safe_load(fh)
        # Verify the data section round-trips cleanly
        assert saved["data"]["venue"] == "binance"
        assert saved["random_seed"] == 42

    def test_each_load_generates_unique_run_id(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml.dump(VALID_CONFIG), encoding="utf-8")
        _, run_id_1 = load_config(yaml_file)
        _, run_id_2 = load_config(yaml_file)
        assert run_id_1 != run_id_2


# parse_cli_overrides


class TestParseCLIOverrides:
    def test_integer_override(self):
        result = parse_cli_overrides(["data.top_k=5"])
        assert result == {"data": {"top_k": 5}}

    def test_nested_override(self):
        result = parse_cli_overrides(["sampling.horizons_events=[10,50]"])
        assert result == {"sampling": {"horizons_events": [10, 50]}}

    def test_multiple_overrides(self):
        result = parse_cli_overrides(["random_seed=99", "data.top_k=20"])
        assert result["random_seed"] == 99
        assert result["data"]["top_k"] == 20

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid override"):
            parse_cli_overrides(["no_equals_sign"])

    def test_float_override(self):
        result = parse_cli_overrides(["labels.direction_threshold_alpha=0.25"])
        assert result["labels"]["direction_threshold_alpha"] == pytest.approx(0.25)

    def test_override_applied_via_load_config(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml.dump(VALID_CONFIG), encoding="utf-8")
        overrides = parse_cli_overrides(["random_seed=7", "data.top_k=5"])
        cfg, _ = load_config(yaml_file, overrides=overrides)
        assert cfg.random_seed == 7
        assert cfg.data.top_k == 5
