"""Orchestrate the passive market-making simulator over one monthly fold.

For each policy: select parameters on the fold's validation day(s) only, freeze
them, then evaluate on validation (for reference) and the held-out test month.
The learned contextual bandit is trained on the fold's training days and never
sees test rewards. Outputs follow the order/fill/inventory/metrics schemas.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ...config import ExperimentConfig
from ...datasets.monthly_splits import MonthlyFold
from ...features.regimes import encode_context_one_hot

from .control import ControlQuoteOptimizer
from .environment import ACTIONS, MarketArrays, MarketMakingEnv, simulate_day
from .environment import _RewardWeights
from .fill_probability import fit_fill_probability
from .fills import infer_tick_size
from .learned_policies import ContextualBanditMM, build_training_examples
from .metrics import compute_policy_metrics, metrics_long
from .policies import build_policy

ORDER_COLUMNS = (
    "run_id", "fold_id", "monthly_date", "model_name", "policy_name", "event_id",
    "timestamp_exchange_ns", "action", "quote_bid", "quote_ask", "bid_quote_price",
    "ask_quote_price", "bid_distance_ticks", "ask_distance_ticks", "inventory_before",
    "inventory_after_action", "reward", "policy_params_json",
)
FILL_COLUMNS = (
    "run_id", "fold_id", "monthly_date", "model_name", "policy_name", "decision_event_id",
    "fill_event_id", "side", "quote_price", "fill_price", "fill_occurred",
    "fill_latency_events", "mark_mid_horizon", "gross_pnl", "fee", "net_pnl",
    "adverse_selection_cost", "fill_model", "fill_assumption_version",
    # queue-aware partial-fill extensions (nullable for the legacy fill models)
    "quote_level", "queue_position", "queue_ahead", "cum_effective_depletion",
    "fill_fraction", "fill_reason", "fill_model_name", "fill_model_params_hash",
)
INVENTORY_COLUMNS = (
    "run_id", "fold_id", "monthly_date", "model_name", "policy_name", "event_id",
    "timestamp_exchange_ns", "cash", "inventory", "mid", "wealth", "drawdown",
)
POLICY_METRIC_COLUMNS = (
    "run_id", "fold_id", "model_name", "policy_name", "split", "monthly_date",
    "metric_name", "metric_value", "created_at_utc",
)

_STATE_SIGNALS = (
    "imbalance_l1", "imbalance_lK", "ofi_10", "ofi_50", "ofi_200",
    "relative_spread", "realised_vol_200",
    "pred_return", "pred_q05", "pred_q50", "pred_q95", "pred_adverse_bid", "pred_adverse_ask",
)


@dataclass
class MarketMakingResult:
    orders: pd.DataFrame
    fills: pd.DataFrame
    inventory: pd.DataFrame
    policy_metrics: pd.DataFrame
    policy_selection: dict[str, Any]
    run_id: str = ""

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.orders.to_parquet(out_dir / "orders.parquet", index=False)
        self.fills.to_parquet(out_dir / "fills.parquet", index=False)
        self.inventory.to_parquet(out_dir / "inventory.parquet", index=False)
        self.policy_metrics.to_parquet(out_dir / "policy_metrics.parquet", index=False)
        with (out_dir / "policy_selection.yaml").open("w", encoding="utf-8") as fh:
            yaml.dump(self.policy_selection, fh, default_flow_style=False, sort_keys=False)
        return out_dir


def _day_arrays(
    fl_day: pd.DataFrame, pred_day: pd.DataFrame, config: ExperimentConfig, monthly_date: str
) -> MarketArrays:
    """Build the per-day market arrays + signals + bandit state matrix."""
    mid = fl_day["mid"].to_numpy(dtype="float64")
    bid = fl_day["bid_px_1"].to_numpy(dtype="float64") if "bid_px_1" in fl_day else (mid - fl_day["spread"].to_numpy() / 2)
    ask = fl_day["ask_px_1"].to_numpy(dtype="float64") if "ask_px_1" in fl_day else (mid + fl_day["spread"].to_numpy() / 2)
    tick = infer_tick_size(np.concatenate([bid, ask]))

    # align predictions (at the backtest horizon) to features-labels rows by timestamp
    pred = pred_day.set_index("timestamp_exchange_ns")
    ts = fl_day["timestamp_exchange_ns"].to_numpy()

    def pred_col(col: str) -> np.ndarray:
        if col not in pred.columns:
            return np.zeros(len(ts))
        s = pred[col].reindex(ts)
        return s.to_numpy(dtype="float64")

    signals = {
        "pred_return": pred_col("pred_return"),
        "interval_width": pred_col("pred_interval_width"),
        "uncertainty_score": pred_col("pred_uncertainty_score"),
        "adv_bid": pred_col("pred_adverse_bid"),
        "adv_ask": pred_col("pred_adverse_ask"),
        "markout_bid": pred_col("pred_markout_bid"),
        "markout_ask": pred_col("pred_markout_ask"),
        "pred_q05": pred_col("pred_q05"),
        "pred_q50": pred_col("pred_q50"),
        "pred_q95": pred_col("pred_q95"),
    }
    for name in ("imbalance_l1", "imbalance_lK", "ofi_10", "ofi_50", "ofi_200",
                 "relative_spread", "realised_vol_200"):
        if name in fl_day.columns:
            signals[name] = fl_day[name].to_numpy(dtype="float64")

    # bandit state matrix: signals + regime one-hot (static; inventory appended at runtime)
    cols = []
    for name in _STATE_SIGNALS:
        if name in fl_day.columns:
            cols.append(np.nan_to_num(fl_day[name].to_numpy(dtype="float64")))
        else:
            cols.append(np.nan_to_num(pred_col(name)))
    onehot, names = encode_context_one_hot(fl_day)
    state = np.column_stack([*cols, onehot]).astype("float64")
    state_names = list(_STATE_SIGNALS) + names

    # top-K level prices/sizes for the queue-aware fill model (level 1 = column 0)
    k = config.data.top_k
    bid_px_levels = _level_matrix(fl_day, "bid_px", k)
    bid_qty_levels = _level_matrix(fl_day, "bid_qty", k)
    ask_px_levels = _level_matrix(fl_day, "ask_px", k)
    ask_qty_levels = _level_matrix(fl_day, "ask_qty", k)

    # market-only feature matrix for the fill-probability model (no forecasts):
    # microstructure signals + regime one-hot + latent state columns when present
    fill_cols: list[np.ndarray] = []
    fill_names: list[str] = []
    for name in ("imbalance_l1", "imbalance_lK", "ofi_10", "ofi_50", "ofi_200",
                 "relative_spread", "realised_vol_200"):
        if name in fl_day.columns:
            fill_cols.append(np.nan_to_num(fl_day[name].to_numpy(dtype="float64")))
            fill_names.append(name)
    ssm_cols = [c for c in fl_day.columns if c.startswith("ssm_z_") or c.startswith("ssm_var_")]
    for c in ssm_cols:
        fill_cols.append(np.nan_to_num(fl_day[c].to_numpy(dtype="float64")))
        fill_names.append(c)
    fill_features = np.column_stack([*fill_cols, onehot]).astype("float64") if fill_cols else onehot.astype("float64")
    fill_feature_names = fill_names + names

    return MarketArrays(
        mid=mid, bid=bid, ask=ask,
        event_id=fl_day["event_id"].to_numpy(dtype="int64"),
        timestamp=ts, tick_size=tick, monthly_date=monthly_date,
        signals=signals, state_matrix=state, state_names=state_names,
        bid_px_levels=bid_px_levels, bid_qty_levels=bid_qty_levels,
        ask_px_levels=ask_px_levels, ask_qty_levels=ask_qty_levels,
        fill_features=fill_features, fill_feature_names=fill_feature_names,
    )


def _level_matrix(fl_day: pd.DataFrame, stem: str, k: int) -> np.ndarray | None:
    """Stack `{stem}_1..k` into an (n, k) array, or None if columns are absent."""
    cols = [f"{stem}_{i}" for i in range(1, k + 1)]
    if not all(c in fl_day.columns for c in cols):
        return None
    return fl_day[cols].to_numpy(dtype="float64")


def _select_queue_fill(config: ExperimentConfig, days: dict, val_dates: set[str],
                       fold_id: int, model_name: str) -> tuple[dict, str, list[dict]]:
    """Pick one queue-fill (kappa, queue_position) per fold on validation reward.

    Selection uses a single reference policy (naive_symmetric) and the same fill
    model for every policy, so no policy gets a tuned simulator.
    Returns (queue_kwargs, params_hash, grid_results).
    """
    qf = config.market_making.queue_fill
    if config.market_making.fill_model != "queue_aware_partial":
        return None, "", []
    base = {"queue_position": qf.queue_position, "full_cross_fill": qf.full_cross_fill}
    grid = qf.depletion_fill_fraction_grid if qf.select_depletion_fraction_on_validation else [qf.depletion_fill_fraction_grid[0]]
    ref = build_policy("naive_symmetric_mm")
    results: list[dict] = []
    best_kw, best_reward = {**base, "kappa": grid[0]}, -np.inf
    for kappa in grid:
        kw = {**base, "kappa": float(kappa)}
        rec = _run_days(ref, days, val_dates, config, fold_id, model_name, {}, queue_kwargs=kw)
        total = float(np.sum(rec["rewards"]))
        results.append({"params": kw, "validation_total_reward": total})
        if total > best_reward:
            best_reward, best_kw = total, kw
    return best_kw, _hash_params(best_kw), results


def _hash_params(params: dict) -> str:
    import hashlib
    return hashlib.sha1(str(sorted(params.items())).encode()).hexdigest()[:12]


def _reward_weights(config: ExperimentConfig) -> _RewardWeights:
    mm = config.market_making
    return _RewardWeights(mm.lambda_inv, mm.lambda_turn, mm.lambda_dd, mm.lambda_adv)


def _sim_kwargs(config: ExperimentConfig) -> dict:
    mm = config.market_making
    bt = config.backtest
    return dict(
        horizon=bt.horizon, quote_size=mm.quote_size,
        distance_ticks=int(mm.quote_distance_ticks[0]),
        max_inventory=bt.max_inventory if bt.max_inventory is not None else bt.max_position,
        maker_fee_rate=bt.fee_bps_maker / 1e4, reward_weights=_reward_weights(config),
        fill_model=mm.fill_model, decision_interval=mm.decision_interval,
    )


def _standardise(days: dict, train_dates: set[str]) -> None:
    """Fit state-matrix mean/std on training days only and apply in place."""
    train_states = [d["arrays"].state_matrix for dt, d in days.items() if dt in train_dates]
    if not train_states:
        return
    pooled = np.vstack(train_states)
    mu = pooled.mean(axis=0)
    sd = pooled.std(axis=0)
    sd[sd == 0] = 1.0
    for d in days.values():
        d["arrays"].state_matrix = ((d["arrays"].state_matrix - mu) / sd).astype("float64")


def _policy_param_grid(name: str, config: ExperimentConfig) -> list[dict]:
    mm = config.market_making
    max_inv = config.backtest.max_inventory if config.backtest.max_inventory is not None else config.backtest.max_position
    if name in ("naive_symmetric_mm", "no_quote"):
        return [{}]
    if name == "inventory_skewed_mm":
        return [{"rho": r, "max_inventory": max_inv} for r in mm.inventory_soft_limit_grid]
    if name == "forecast_aware_mm":
        return [{"theta": th, "rho": 0.75, "max_inventory": max_inv} for th in mm.return_threshold_grid]
    if name == "uncertainty_aware_mm":
        grid = []
        for u, th in itertools.product(mm.uncertainty_threshold_grid, mm.return_threshold_grid):
            grid.append({"u_thresh": u, "theta": th, "adv_thresh": 1e9, "max_inventory": max_inv})
        return grid
    return [{}]


def _run_days(policy, days: dict, date_set: set[str], config: ExperimentConfig,
              fold_id: int, model_name: str, params: dict,
              queue_kwargs: dict | None = None, fill_hash: str = "") -> dict[str, list]:
    out = {"orders": [], "fills": [], "inventory": [], "rewards": []}
    kw = _sim_kwargs(config)
    for dt, d in days.items():
        if dt not in date_set:
            continue
        policy.reset()
        rec = simulate_day(policy, d["arrays"], fold_id=fold_id, model_name=model_name,
                           policy_name=policy.name, policy_params=params,
                           queue_kwargs=queue_kwargs, fill_model_params_hash=fill_hash, **kw)
        out["orders"].extend(rec["orders"])
        out["fills"].extend(rec["fills"])
        out["inventory"].extend(rec["inventory"])
        out["rewards"].extend(rec["rewards"])
    return out


def run_market_making(
    config: ExperimentConfig,
    fold: MonthlyFold,
    predictions: pd.DataFrame,
    features_labels: pd.DataFrame,
    run_id: str,
    project_root: str | Path | None = None,
) -> MarketMakingResult:
    """Run all configured policies for one fold and one forecasting model."""
    model_name = str(predictions["model_name"].iloc[0]) if len(predictions) else "none"
    created = datetime.now(timezone.utc).isoformat()
    h = config.backtest.horizon
    pred_h = predictions[predictions["horizon"] == h]

    train_dates = {d.isoformat() for d in fold.train_dates}
    val_dates = {d.isoformat() for d in fold.validation_dates}
    test_dates = {d.isoformat() for d in fold.test_dates}
    fold_dates = train_dates | val_dates | test_dates

    # build per-day arrays
    days: dict[str, dict] = {}
    fl = features_labels.copy()
    fl["monthly_date"] = fl["monthly_date"].astype("string")
    for monthly_date, fl_day in fl.groupby("monthly_date"):
        md = str(monthly_date)
        if md not in fold_dates:
            continue
        split = "train" if md in train_dates else "validation" if md in val_dates else "test"
        pred_day = pred_h[pred_h["timestamp_exchange_ns"].isin(fl_day["timestamp_exchange_ns"])]
        arrays = _day_arrays(fl_day.reset_index(drop=True), pred_day, config, md)
        days[md] = {"split": split, "arrays": arrays}

    _standardise(days, train_dates)

    # one queue-fill model per fold, selected on validation, used by every policy
    queue_kw, fill_hash, qf_grid = _select_queue_fill(
        config, days, val_dates, fold.fold_id, model_name)

    all_orders, all_fills, all_inv, metric_rows = [], [], [], []
    selection: dict[str, Any] = {"fold_id": fold.fold_id, "model_name": model_name,
                                 "horizon": h, "objective": "validation_total_reward",
                                 "fill_model": config.market_making.fill_model,
                                 "queue_fill": {"selected_params": queue_kw, "grid": qf_grid},
                                 "policies": {}}

    for policy_name in config.market_making.policies:
        if policy_name == "contextual_bandit_mm":
            policy, params, grid = _fit_bandit(config, days, train_dates)
        elif policy_name == "control_quote_optimizer":
            policy, params, grid = _fit_control(
                config, days, train_dates, val_dates, fold.fold_id, model_name, queue_kw, fill_hash)
        else:
            policy, params, grid = _select_policy(
                policy_name, config, days, val_dates, fold.fold_id, model_name, queue_kw, fill_hash)
        selection["policies"][policy_name] = {"selected_params": params, "grid": grid}

        # frozen evaluation on validation (reference) and test
        for split, dates in (("validation", val_dates), ("test", test_dates)):
            rec = _run_days(policy, days, dates, config, fold.fold_id, model_name, params,
                            queue_kwargs=queue_kw, fill_hash=fill_hash)
            orders = pd.DataFrame(rec["orders"])
            fills = pd.DataFrame(rec["fills"])
            inv = pd.DataFrame(rec["inventory"])
            all_orders.append(orders)
            all_fills.append(fills)
            all_inv.append(inv)

            # overall metrics for this split
            mets = compute_policy_metrics(orders, fills, inv)
            metric_rows.extend(metrics_long(mets, run_id=run_id, fold_id=fold.fold_id,
                model_name=model_name, policy_name=policy_name, split=split,
                monthly_date="all", created_at=created))
            # per-month metrics
            if len(orders):
                for md, og in orders.groupby("monthly_date"):
                    fg = fills[fills["monthly_date"] == md] if len(fills) else fills
                    ig = inv[inv["monthly_date"] == md] if len(inv) else inv
                    m = compute_policy_metrics(og, fg, ig)
                    metric_rows.extend(metrics_long(m, run_id=run_id, fold_id=fold.fold_id,
                        model_name=model_name, policy_name=policy_name, split=split,
                        monthly_date=str(md), created_at=created))

    def _concat(parts, cols):
        parts = [p for p in parts if len(p)]
        return pd.concat(parts, ignore_index=True)[list(cols)] if parts else pd.DataFrame(columns=list(cols))

    for p in all_orders:
        p["run_id"] = run_id
    for p in all_fills:
        p["run_id"] = run_id
    for p in all_inv:
        p["run_id"] = run_id

    return MarketMakingResult(
        orders=_concat(all_orders, ORDER_COLUMNS),
        fills=_concat(all_fills, FILL_COLUMNS),
        inventory=_concat(all_inv, INVENTORY_COLUMNS),
        policy_metrics=pd.DataFrame(metric_rows, columns=list(POLICY_METRIC_COLUMNS)),
        policy_selection=selection,
        run_id=run_id,
    )


def _select_policy(policy_name, config, days, val_dates, fold_id, model_name,
                   queue_kwargs=None, fill_hash=""):
    """Grid-search a deterministic policy's params on validation total reward."""
    grid_results = []
    best_params, best_reward = {}, -np.inf
    for params in _policy_param_grid(policy_name, config):
        policy = build_policy(policy_name, **params)
        rec = _run_days(policy, days, val_dates, config, fold_id, model_name, params,
                        queue_kwargs=queue_kwargs, fill_hash=fill_hash)
        total = float(np.sum(rec["rewards"]))
        grid_results.append({"params": params, "validation_total_reward": total})
        if total > best_reward:
            best_reward, best_params = total, params
    return build_policy(policy_name, **best_params), best_params, grid_results


def _fit_control(config, days, train_dates, val_dates, fold_id, model_name, queue_kw, fill_hash):
    """Fit the ex-ante fill-probability model and select control lambdas on validation.

    The fill model is trained on training days only; the control risk weights are
    grid-searched on validation total reward and frozen for test.
    """
    ctrl = config.market_making.control
    bt = config.backtest
    mm = config.market_making
    max_inv = bt.max_inventory if bt.max_inventory is not None else bt.max_position
    train_arrays = [d["arrays"] for dt, d in days.items() if dt in train_dates]
    val_arrays = [d["arrays"] for dt, d in days.items() if dt in val_dates]

    fill_model = fit_fill_probability(
        train_arrays, val_arrays, horizon=bt.horizon, quote_size=mm.quote_size,
        levels=list(ctrl.action_levels), queue_kwargs=queue_kw or {},
        decision_interval=mm.decision_interval, classifier_kind=ctrl.fill_probability_model,
    )

    base = dict(quote_size=mm.quote_size, max_inventory=max_inv, horizon=bt.horizon,
                maker_fee=bt.fee_bps_maker / 1e4, action_levels=list(ctrl.action_levels))
    grids = [ctrl.lambda_inv_grid, ctrl.lambda_turn_grid, ctrl.lambda_adv_grid,
             ctrl.lambda_unc_grid, ctrl.lambda_act_grid]
    combos = list(itertools.product(*grids)) if ctrl.select_params_on_validation else [
        (grids[0][0], grids[1][0], grids[2][0], grids[3][0], grids[4][0])]

    grid_results: list[dict] = []
    best_params, best_reward = None, -np.inf
    for li, lt, la, lu, lact in combos:
        params = {"lambda_inv": li, "lambda_turn": lt, "lambda_adv": la,
                  "lambda_unc": lu, "lambda_act": lact}
        policy = ControlQuoteOptimizer(fill_model=fill_model, **base, **params)
        rec = _run_days(policy, days, val_dates, config, fold_id, model_name, params,
                        queue_kwargs=queue_kw, fill_hash=fill_hash)
        total = float(np.sum(rec["rewards"]))
        grid_results.append({"params": params, "validation_total_reward": total})
        if total > best_reward:
            best_reward, best_params = total, params
    best_params = best_params or {}
    policy = ControlQuoteOptimizer(fill_model=fill_model, **base, **best_params)
    selected = {**best_params, "fill_probability_model": fill_model.classifier_kind,
                "fill_prob_calibrated": fill_model.calibrator is not None}
    return policy, selected, grid_results


def _fit_bandit(config, days, train_dates):
    """Train the contextual bandit on training days only (no test leakage)."""
    kw = _sim_kwargs(config)
    train_days = [d for dt, d in days.items() if dt in train_dates]
    envs = [MarketMakingEnv(d["arrays"], **kw) for d in train_days]
    states = [d["arrays"].state_matrix for d in train_days]
    examples = build_training_examples(envs, states, config.market_making.decision_interval)
    bandit = ContextualBanditMM().fit(examples)
    return bandit, {"function_class": "logistic_regression", "trained_on": "train_days"}, []
