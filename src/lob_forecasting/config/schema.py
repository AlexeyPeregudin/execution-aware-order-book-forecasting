"""Config objects for an experiment run, validated with pydantic."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# the models we know how to build; ModelRunConfig checks against this
REGISTERED_MODELS: frozenset[str] = frozenset(
    {
        "no_change",
        "imbalance_rule",
        "logistic_regression",
        "ridge_regression",
        "lightgbm",
        "tcn_small",
        "tcn_exec_multitask",
    }
)


def _is_first_of_month(d: date) -> bool:
    return d.day == 1


class MonthlySnapshotConfig(BaseModel):
    """First-day-of-month snapshot scope (the monthly distribution-shift benchmark).

    When enabled, the pipeline runs only on the listed first-of-month dates and
    treats each date as an independent calendar regime.
    """

    enabled: bool = False
    day_of_month: int = 1
    dates: list[date] = Field(default_factory=list)

    @field_validator("dates", mode="before")
    @classmethod
    def parse_dates(cls, v: object) -> object:
        # placeholder "YYYY-MM-01" template entries are dropped so a skeleton
        # config still validates; real dates are kept
        if isinstance(v, list):
            return [x for x in v if not (isinstance(x, str) and x.upper().startswith("YYYY"))]
        return v

    @field_validator("dates")
    @classmethod
    def dates_first_of_month(cls, v: list[date]) -> list[date]:
        bad = [d.isoformat() for d in v if not _is_first_of_month(d)]
        if bad:
            raise ValueError(
                f"monthly_snapshot dates must be the first day of their month; got {bad}"
            )
        return sorted(v)

    @model_validator(mode="after")
    def day_of_month_is_one(self) -> MonthlySnapshotConfig:
        if self.enabled and self.day_of_month != 1:
            raise ValueError("monthly_snapshot.day_of_month must be 1 in this extension")
        return self


class DataConfig(BaseModel):
    venue: str
    symbols: list[str]
    start_date: date | None = None
    end_date: date | None = None
    top_k: int = 10
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    artefact_dir: Path = Path("artefacts")
    monthly_snapshot: MonthlySnapshotConfig = Field(default_factory=MonthlySnapshotConfig)

    @field_validator("top_k")
    @classmethod
    def top_k_at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"top_k must be >= 1, got {v}")
        return v

    @field_validator("symbols")
    @classmethod
    def symbols_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("symbols must contain at least one entry")
        return v

    @field_validator("raw_dir", "processed_dir", "artefact_dir", mode="before")
    @classmethod
    def reject_absolute_paths(cls, v: object) -> object:
        # output dirs must stay inside the project, so no absolute paths
        p = Path(str(v))
        if p.is_absolute():
            raise ValueError(f"output path must be relative, got absolute path: {v}")
        return v

    @model_validator(mode="after")
    def resolve_date_range(self) -> DataConfig:
        # in monthly mode the date range is derived from the listed snapshots
        if self.monthly_snapshot.enabled and self.monthly_snapshot.dates:
            self.start_date = self.monthly_snapshot.dates[0]
            self.end_date = self.monthly_snapshot.dates[-1]
        if self.start_date is None or self.end_date is None:
            if not self.monthly_snapshot.enabled:
                raise ValueError("start_date and end_date are required unless monthly_snapshot is used")
            return self
        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be <= end_date ({self.end_date})"
            )
        return self

    @property
    def monthly_dates(self) -> list[date]:
        return list(self.monthly_snapshot.dates)


class SamplingConfig(BaseModel):
    mode: Literal["event_time"] = "event_time"
    horizons_events: list[int] = [10, 50, 200]
    feature_lookbacks_events: list[int] = [10, 50, 200]
    # keep every Nth book snapshot before features (book_snapshot data is highly
    # redundant at sub-second cadence; 1 = use every event)
    event_stride: int = 1

    @field_validator("event_stride")
    @classmethod
    def stride_at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"event_stride must be >= 1, got {v}")
        return v

    @field_validator("horizons_events", "feature_lookbacks_events")
    @classmethod
    def all_positive_integers(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("list must be non-empty")
        for h in v:
            if h <= 0:
                raise ValueError(f"all values must be positive integers, got {h}")
        return v


class LabelConfig(BaseModel):
    direction_threshold_mode: Literal["train_median_relative_spread"] = (
        "train_median_relative_spread"
    )
    direction_threshold_alpha: float = 0.5
    # execution-aware label extensions (used by the multi-task TCN)
    quantiles: list[float] = Field(default_factory=list)
    include_markout_targets: bool = False
    include_adverse_selection_targets: bool = False

    @field_validator("quantiles")
    @classmethod
    def quantiles_in_unit_interval(cls, v: list[float]) -> list[float]:
        for q in v:
            if not (0.0 < q < 1.0):
                raise ValueError(f"quantiles must be in (0, 1), got {q}")
        return v


class SplitConfig(BaseModel):
    # 'fraction' is the original time-fraction split; the monthly modes split by
    # first-of-month snapshot date and produce one or more folds.
    mode: Literal["fraction", "fixed_monthly_snapshot", "expanding_monthly_snapshot"] = "fraction"
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    embargo_events: int = 200
    # monthly-mode knobs
    min_train_months: int = 3
    validation_months: int = 1
    test_months: int = 1
    step_months: int = 1  # how many months the train window grows per expanding fold
    aggregate_fold_metrics: bool = True
    # In monthly mode the canonical (frozen) training rows for fitting global
    # direction/regime thresholds are the first fold's training months. These
    # flags make that behaviour explicit; setting either to False would fit the
    # transform per fold instead (not recommended -- it weakens causality).
    freeze_global_label_thresholds_on_first_train: bool = True
    freeze_global_regime_thresholds_on_first_train: bool = True
    # explicit date lists for fixed_monthly_snapshot (ISO date strings)
    train_dates: list[date] = Field(default_factory=list)
    validation_dates: list[date] = Field(default_factory=list)
    test_dates: list[date] = Field(default_factory=list)

    @field_validator("step_months")
    @classmethod
    def step_months_at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"step_months must be >= 1, got {v}")
        return v

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> SplitConfig:
        if self.mode != "fraction":
            return self  # fractions are only used in fraction mode
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must equal 1.0, got {total:.10f}")
        return self

    @property
    def is_monthly(self) -> bool:
        return self.mode in ("fixed_monthly_snapshot", "expanding_monthly_snapshot")


class FeatureConfig(BaseModel):
    include_basic_microstructure: bool = True
    include_best_level_ofi: bool = True
    include_multilevel_imbalance: bool = True
    realised_vol_lookbacks_events: list[int] = [50, 200]
    # also keep raw top-K prices/sizes in the feature table (the TCN wants them)
    include_raw_levels: bool = False
    # regime descriptors (volatility / spread / liquidity buckets + time-of-day)
    include_regime_features: bool = False
    regime_windows_events: dict[str, int] = Field(
        default_factory=lambda: {"volatility": 1000, "spread": 1000, "liquidity": 1000}
    )
    regime_quantiles: dict[str, float] = Field(
        default_factory=lambda: {"low": 0.33, "high": 0.67}
    )

    @field_validator("realised_vol_lookbacks_events")
    @classmethod
    def vol_lookbacks_positive(cls, v: list[int]) -> list[int]:
        for lb in v:
            if lb <= 0:
                raise ValueError(f"realised_vol_lookbacks_events must be positive, got {lb}")
        return v

    @field_validator("regime_quantiles")
    @classmethod
    def regime_quantiles_valid(cls, v: dict[str, float]) -> dict[str, float]:
        low = v.get("low", 0.33)
        high = v.get("high", 0.67)
        if not (0.0 < low < high < 1.0):
            raise ValueError(f"regime_quantiles must satisfy 0 < low < high < 1, got {v}")
        return v


class ReturnHeadConfig(BaseModel):
    """Point-return head wiring for the multi-task TCN."""

    enabled: bool = True
    loss_weight: float = 0.10
    detach_from_encoder: bool = False
    prediction_source: Literal["neural_head", "none", "ridge_sidecar"] = "neural_head"
    ridge_model_name: str = "ridge_regression"


class _HeadConfig(BaseModel):
    enabled: bool = True
    loss_weight: float = 1.0


class ExecutionHeadsConfig(BaseModel):
    """Per-head enable/weight wiring for the multi-task TCN.

    The model reads this (as a plain dict from its yaml) to switch heads on/off,
    set per-head loss weights, and choose how the point return is produced.
    """

    return_head: ReturnHeadConfig = Field(default_factory=ReturnHeadConfig)
    direction_head: _HeadConfig = Field(default_factory=lambda: _HeadConfig(loss_weight=1.0))
    quantile_head: _HeadConfig = Field(default_factory=lambda: _HeadConfig(loss_weight=1.0))
    markout_head: _HeadConfig = Field(default_factory=lambda: _HeadConfig(loss_weight=0.5))
    adverse_head: _HeadConfig = Field(default_factory=lambda: _HeadConfig(loss_weight=0.25))


class ModelRunConfig(BaseModel):
    run: list[str]

    @field_validator("run")
    @classmethod
    def models_are_registered(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - REGISTERED_MODELS)
        if unknown:
            raise ValueError(
                f"Unknown model(s): {unknown}. known: {sorted(REGISTERED_MODELS)}"
            )
        return v


class BacktestConfig(BaseModel):
    horizon: int = 50
    threshold_grid: list[float] = Field(
        default_factory=lambda: [0.0, 0.00001, 0.00002, 0.00005, 0.0001]
    )
    fee_bps: float = 5.0
    latency_events: int = 1
    max_position: float = 1.0
    trade_size: float = 1.0
    # extension switches / aliases
    run_taker_sanity: bool = True
    run_market_making: bool = False
    fee_bps_taker: float | None = None  # falls back to fee_bps if unset
    fee_bps_maker: float = 0.0
    max_inventory: float | None = None  # falls back to max_position if unset

    @field_validator("threshold_grid")
    @classmethod
    def grid_is_non_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("threshold_grid must contain at least one value")
        return v

    @field_validator("latency_events")
    @classmethod
    def latency_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"latency_events must be >= 0, got {v}")
        return v

    @property
    def taker_fee_bps(self) -> float:
        return self.fee_bps_taker if self.fee_bps_taker is not None else self.fee_bps


class QueueFillConfig(BaseModel):
    """Queue-aware partial-fill model settings."""

    quote_levels: list[int] = Field(default_factory=lambda: [1, 2])
    queue_position: Literal["front", "middle", "back"] = "back"
    depletion_fill_fraction_grid: list[float] = Field(default_factory=lambda: [0.25, 0.50, 0.75])
    select_depletion_fraction_on_validation: bool = True
    aggregate_partial_fills: bool = True
    full_cross_fill: bool = True

    @field_validator("quote_levels")
    @classmethod
    def quote_levels_valid(cls, v: list[int]) -> list[int]:
        if not v or any(lv not in (1, 2) for lv in v):
            raise ValueError(f"queue_fill.quote_levels must be a non-empty subset of [1, 2], got {v}")
        return v

    @field_validator("depletion_fill_fraction_grid")
    @classmethod
    def kappa_in_unit_interval(cls, v: list[float]) -> list[float]:
        if not v or any(not (0.0 <= k <= 1.0) for k in v):
            raise ValueError(f"depletion_fill_fraction_grid values must be in [0, 1], got {v}")
        return v


class ControlPolicyConfig(BaseModel):
    """Control-style quote optimiser settings."""

    enabled: bool = False
    action_levels: list[int] = Field(default_factory=lambda: [1, 2])
    lambda_inv_grid: list[float] = Field(default_factory=lambda: [0.005, 0.01, 0.02])
    lambda_turn_grid: list[float] = Field(default_factory=lambda: [0.0005, 0.001])
    lambda_adv_grid: list[float] = Field(default_factory=lambda: [0.10, 0.25, 0.50])
    lambda_unc_grid: list[float] = Field(default_factory=lambda: [0.0, 0.01, 0.05])
    lambda_act_grid: list[float] = Field(default_factory=lambda: [0.0, 0.001])
    select_params_on_validation: bool = True
    min_fills_per_test_month: int = 25
    fill_probability_model: Literal["lightgbm", "logistic_regression"] = "lightgbm"


class MarketMakingConfig(BaseModel):
    """Passive market-making simulator settings."""

    enabled: bool = False
    action_space: Literal["quote_sides_only", "quote_sides_and_distance"] = "quote_sides_only"
    quote_distance_ticks: list[int] = Field(default_factory=lambda: [0, 1])
    quote_size: float = 1.0
    fill_model: Literal[
        "conservative_touch_or_mid_cross", "touch_through", "queue_aware_partial"
    ] = "conservative_touch_or_mid_cross"
    queue_fill: QueueFillConfig = Field(default_factory=QueueFillConfig)
    control: ControlPolicyConfig = Field(default_factory=ControlPolicyConfig)
    # only act every Nth event (a market maker requotes periodically, not on every
    # micro-update); keeps the replay tractable on high-frequency real data
    decision_interval: int = 1
    inventory_penalties: list[float] = Field(default_factory=lambda: [0.0, 0.001, 0.01, 0.05])
    uncertainty_threshold_grid: list[float] = Field(
        default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0]
    )
    policies: list[str] = Field(
        default_factory=lambda: [
            "naive_symmetric_mm",
            "inventory_skewed_mm",
            "forecast_aware_mm",
            "uncertainty_aware_mm",
            "contextual_bandit_mm",
        ]
    )
    # reward weights (selected on validation in practice; these are defaults)
    lambda_inv: float = 0.01
    lambda_turn: float = 0.001
    lambda_dd: float = 0.0
    lambda_adv: float = 0.25
    inventory_soft_limit_grid: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75])
    return_threshold_grid: list[float] = Field(
        default_factory=lambda: [0.0, 0.00001, 0.00005, 0.0001]
    )


class BlockBootstrapConfig(BaseModel):
    enabled: bool = False
    n_bootstrap: int = 500
    block_size_events: int = 1000
    confidence_level: float = 0.95

    @field_validator("confidence_level")
    @classmethod
    def confidence_in_unit_interval(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1), got {v}")
        return v


class RobustnessConfig(BaseModel):
    block_bootstrap: BlockBootstrapConfig = Field(default_factory=BlockBootstrapConfig)


class SyntheticConfig(BaseModel):
    """Settings for the fake-data generator (no external files needed)."""

    num_days: int = 2
    rows_per_day: int = 2000
    seed: int = 7
    base_price: float | None = None  # if None we pick a price per symbol

    @field_validator("num_days", "rows_per_day")
    @classmethod
    def positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"value must be a positive integer, got {v}")
        return v


class IngestionConfig(BaseModel):
    """Where raw data comes from.

    mode picks the source: 'synthetic' generates fake data and needs nothing,
    'local_archive'/'fixture' read files under source_root, and 'url_archive'
    downloads from base_url using url_template.
    """

    mode: Literal[
        "synthetic", "local_archive", "url_archive", "fixture", "tardis_archive"
    ] = "synthetic"
    file_format: str = "csv"
    schema_version: str = "1"
    source_root: Path | None = None  # for local_archive / fixture
    base_url: str | None = None  # for url_archive
    url_template: str = "{base_url}/{symbol}/{symbol}-{date}.{ext}"
    overwrite: bool = False
    manifest_path: Path = Path("data_manifest/sources.yml")
    checksums_path: Path = Path("data_manifest/checksums.yml")
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)

    @model_validator(mode="after")
    def check_mode_requirements(self) -> IngestionConfig:
        if self.mode in ("local_archive", "fixture", "tardis_archive") and self.source_root is None:
            raise ValueError(f"mode '{self.mode}' requires 'source_root' to be set")
        if self.mode == "url_archive" and not self.base_url:
            raise ValueError("mode 'url_archive' requires 'base_url' to be set")
        return self


class NormalisationConfig(BaseModel):
    """Settings for turning raw messages into the event table."""

    # what to do with event types we don't recognise
    unknown_event_type_policy: Literal["flag", "reject"] = "flag"
    # force a specific venue adapter; None means pick by data.venue
    venue_adapter: str | None = None


class OrderBookConfig(BaseModel):
    """Settings for rebuilding the order book."""

    # 'auto' decides snapshot vs replay per file; the others force it
    mode: Literal["auto", "snapshot", "replay"] = "auto"
    # whether to attach trades to the previous book state (off for now)
    align_trades: bool = False


class DatasetConfig(BaseModel):
    """Settings for building train/val/test datasets."""

    sequence_length: int = 100  # window length for the TCN
    exclude_crossed: bool = True  # drop crossed-book rows from the datasets
    drop_incomplete_feature_rows: bool = True  # drop rows with missing features

    @field_validator("sequence_length")
    @classmethod
    def sequence_length_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"sequence_length must be >= 1, got {v}")
        return v


class LatentStateConfig(BaseModel):
    """Linear-Gaussian latent state-space context module.

    Fitted per fold on training months only; the causal filtered state and its
    variance are appended to the model context branch and the policy state.
    """

    enabled: bool = False
    model_type: Literal["linear_gaussian"] = "linear_gaussian"
    state_dim: int = 4
    observation_columns: list[str] = Field(
        default_factory=lambda: [
            "imbalance_l1",
            "imbalance_lK",
            "ofi_50",
            "return_lag_10",
            "realised_vol_200",
            "relative_spread",
            "regime_depth",
        ]
    )
    fit_scope: Literal["fold_train"] = "fold_train"
    reset_each_monthly_day: bool = True
    append_to_model_context: bool = True
    append_to_policy_state: bool = True
    max_em_iterations: int = 25
    loglik_tol: float = 1e-4

    @field_validator("state_dim")
    @classmethod
    def state_dim_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"latent_state.state_dim must be >= 1, got {v}")
        return v

    @field_validator("observation_columns")
    @classmethod
    def observations_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("latent_state.observation_columns must be non-empty")
        return v


class ExperimentConfig(BaseModel):
    data: DataConfig
    sampling: SamplingConfig
    labels: LabelConfig
    splits: SplitConfig
    features: FeatureConfig
    models: ModelRunConfig
    backtest: BacktestConfig
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    normalisation: NormalisationConfig = Field(default_factory=NormalisationConfig)
    orderbook: OrderBookConfig = Field(default_factory=OrderBookConfig)
    datasets: DatasetConfig = Field(default_factory=DatasetConfig)
    market_making: MarketMakingConfig = Field(default_factory=MarketMakingConfig)
    latent_state: LatentStateConfig = Field(default_factory=LatentStateConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    random_seed: int = 42

    @model_validator(mode="after")
    def embargo_covers_max_horizon(self) -> ExperimentConfig:
        # the embargo gap has to be at least as long as the longest label horizon,
        # otherwise a training row's label could peek into the next block
        max_h = max(self.sampling.horizons_events)
        if self.splits.embargo_events < max_h:
            raise ValueError(
                f"embargo_events ({self.splits.embargo_events}) must be >= "
                f"max(horizons_events) ({max_h})"
            )
        return self

    @model_validator(mode="after")
    def monthly_extension_invariants(self) -> ExperimentConfig:
        """Hard constraints for the monthly top-5 BTCUSDT extension."""
        monthly = self.data.monthly_snapshot.enabled or self.splits.is_monthly
        if not monthly:
            return self

        if self.data.symbols != ["BTCUSDT"]:
            raise ValueError("monthly extension requires data.symbols == ['BTCUSDT']")
        if self.data.top_k != 5:
            raise ValueError("monthly extension requires data.top_k == 5")
        if self.sampling.mode != "event_time":
            raise ValueError("monthly extension requires sampling.mode == 'event_time'")

        n_dates = len(self.data.monthly_snapshot.dates)
        need = (
            self.splits.min_train_months
            + self.splits.validation_months
            + self.splits.test_months
        )
        if n_dates > 0 and n_dates < need:
            raise ValueError(
                f"monthly_snapshot needs at least {need} dates "
                f"(min_train+validation+test), got {n_dates}"
            )

        if self.market_making.enabled and not self.labels.include_markout_targets:
            raise ValueError(
                "market_making.enabled requires labels.include_markout_targets == true"
            )
        if (
            "contextual_bandit_mm" in self.market_making.policies
            and self.market_making.enabled
            and self.splits.validation_months < 1
            and self.splits.mode != "fraction"
        ):
            raise ValueError("contextual_bandit_mm requires at least one validation fold")
        return self

    def config_warnings(self) -> list[str]:
        """Soft warnings: non-fatal data-adequacy notes."""
        warns: list[str] = []
        if self.data.monthly_snapshot.enabled:
            n = len(self.data.monthly_snapshot.dates)
            if 0 < n < 6:
                warns.append(f"fewer than 6 monthly snapshots available (got {n})")
        return warns
