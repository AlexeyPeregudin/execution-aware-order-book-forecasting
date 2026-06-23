"""Tests for the execution-aware taker backtest module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lob_forecasting.backtesting import BookSeq, run_backtest, simulate
from lob_forecasting.config import ExperimentConfig
from lob_forecasting.datasets import build_datasets
from lob_forecasting.features import build_features
from lob_forecasting.ingestion import ingest_raw_data
from lob_forecasting.labels import build_labels
from lob_forecasting.normalisation import normalise_events
from lob_forecasting.orderbook import build_order_books
from lob_forecasting.training import train_and_predict


def _seq() -> BookSeq:
    """5-event book: bid below mid below ask at every level."""
    return BookSeq(
        ts_to_pos={10: 0, 20: 1, 30: 2, 40: 3, 50: 4},
        event_id=np.array([0, 1, 2, 3, 4]),
        bid=np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
        ask=np.array([101.0, 102.0, 103.0, 104.0, 105.0]),
        mid=np.array([100.5, 101.5, 102.5, 103.5, 104.5]),
    )


def _pred(rows: list[tuple[int, float]]) -> pd.DataFrame:
    """Build a prediction frame from (timestamp, pred_return) pairs."""
    return pd.DataFrame({
        "venue": "binance", "symbol": "BTCUSDT",
        "event_id": list(range(len(rows))),
        "timestamp_exchange_ns": [t for t, _ in rows],
        "pred_return": [r for _, r in rows],
    })


def _sim(pred, **kw):
    defaults = dict(threshold=0.0, horizon=2, latency_events=1, fee_bps=0.0,
                    trade_size=1.0, max_position=10.0)
    defaults.update(kw)
    return simulate(pred, _seq(), **defaults)




def test_buy_executes_at_ask():
    trades, _, _ = _sim(_pred([(10, 0.5)]))  # buy at decision pos 0
    assert len(trades) == 1
    t = trades[0]
    assert t["side"] == "buy"
    assert t["exec_price"] == 102.0  # ask at exec pos 1 (latency 1)
    assert t["mark_price"] == 102.5  # mid at mark pos 2 (horizon 2)


def test_sell_executes_at_bid():
    trades, _, _ = _sim(_pred([(10, -0.5)]))  # sell at decision pos 0
    assert len(trades) == 1
    t = trades[0]
    assert t["side"] == "sell"
    assert t["exec_price"] == 101.0  # bid at exec pos 1




def test_latency_shifts_execution_event():
    t0 = _sim(_pred([(10, 0.5)]), latency_events=0)[0][0]
    t1 = _sim(_pred([(10, 0.5)]), latency_events=1)[0][0]
    assert t0["exec_price"] == 101.0  # ask at pos 0
    assert t1["exec_price"] == 102.0  # ask at pos 1
    assert t0["exec_event_id"] != t1["exec_event_id"]




def test_fees_reduce_net_pnl():
    no_fee = _sim(_pred([(10, 0.5)]), fee_bps=0.0)[0][0]
    with_fee = _sim(_pred([(10, 0.5)]), fee_bps=5.0)[0][0]
    assert with_fee["fees"] > 0
    assert with_fee["net_pnl"] < with_fee["gross_pnl"]
    assert no_fee["net_pnl"] == no_fee["gross_pnl"]
    # Net = gross - fees exactly.
    assert with_fee["net_pnl"] == pytest.approx(with_fee["gross_pnl"] - with_fee["fees"])




def test_position_limit_blocks_overlapping_trades():
    # Two buys one event apart; horizon 3 keeps the first open when the second fires.
    pred = _pred([(10, 0.5), (20, 0.5)])
    trades, _, skipped = _sim(pred, horizon=3, latency_events=0, max_position=1.0)
    assert len(trades) == 1
    assert skipped == 1


def test_higher_position_limit_allows_both():
    pred = _pred([(10, 0.5), (20, 0.5)])
    trades, _, skipped = _sim(pred, horizon=3, latency_events=0, max_position=2.0)
    assert len(trades) == 2
    assert skipped == 0


# Exclusion: no future mark/exec price


def test_excluded_when_no_future_mark():
    # Decision at last position -> mark pos out of range.
    trades, excluded, _ = _sim(_pred([(50, 0.5)]), horizon=2, latency_events=1)
    assert len(trades) == 0
    assert excluded == 1


def test_hold_when_signal_within_threshold():
    trades, _, _ = _sim(_pred([(10, 0.000001)]), threshold=0.001)
    assert len(trades) == 0  # |r| <= theta -> hold




def _config() -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "data": {"venue": "binance", "symbols": ["BTCUSDT"], "start_date": "2024-01-01", "end_date": "2024-03-31", "top_k": 2},
        "sampling": {"horizons_events": [1, 2], "feature_lookbacks_events": [1, 2]},
        "labels": {"direction_threshold_alpha": 0.5},
        "splits": {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "embargo_events": 2},
        "features": {"realised_vol_lookbacks_events": [2]},
        "models": {"run": ["no_change", "ridge_regression"]},
        "backtest": {"horizon": 2, "threshold_grid": [0.0, 0.00001, 0.00005], "fee_bps": 5.0,
                     "latency_events": 1, "max_position": 1.0, "trade_size": 1.0},
        "ingestion": {"mode": "synthetic", "synthetic": {"num_days": 1, "rows_per_day": 200, "seed": 7}},
        "datasets": {"sequence_length": 5},
        "random_seed": 42,
    })


@pytest.fixture(scope="module")
def backtest_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("backtest")
    config = _config()
    run_id = "run_bt"
    manifest = ingest_raw_data(config, project_root=root)
    events = normalise_events(config, manifest, project_root=root)
    books = build_order_books(config, events, project_root=root)
    features = build_features(config, books, project_root=root)
    labelled = build_labels(config, features, run_id, project_root=root)
    ds_index = build_datasets(config, labelled, run_id, project_root=root)
    preds = [
        train_and_predict(config, ds_index, name, project_root=root, model_config_dir=None)
        for name in ("no_change", "ridge_regression")
    ]
    result = run_backtest(config, preds, books, project_root=root, write=True)
    return config, result, root, run_id


def test_validation_threshold_frozen_for_test(backtest_run):
    config, result, root, run_id = backtest_run
    selected = result.threshold_selection["models"]["ridge_regression"]["selected_threshold"]
    assert selected in config.backtest.threshold_grid

    # Every test trade for the model uses the frozen, validation-selected threshold.
    test_trades = result.trades[(result.trades["model_name"] == "ridge_regression") & (result.trades["split"] == "test")]
    if len(test_trades) > 0:
        assert (test_trades["threshold"] == selected).all()

    # The selection recorded a validation grid search.
    grid = result.threshold_selection["models"]["ridge_regression"]["grid"]
    assert len(grid) == len(config.backtest.threshold_grid)
    assert all("validation_net_pnl" in row for row in grid)


def test_backtest_outputs_written(backtest_run):
    config, result, root, run_id = backtest_run
    bt = root / "artefacts" / "runs" / run_id / "backtests"
    assert (bt / "taker_trades.parquet").exists()
    assert (bt / "taker_metrics.parquet").exists()
    assert (bt / "threshold_selection.yaml").exists()


def test_required_metrics_present(backtest_run):
    config, result, root, run_id = backtest_run
    names = set(result.metrics["metric_name"])
    for required in ("n_trades", "turnover", "gross_pnl", "net_pnl", "mean_pnl_per_trade",
                     "hit_rate", "max_drawdown", "sharpe_like"):
        assert required in names


def test_fees_reduce_pnl_in_real_trades(backtest_run):
    config, result, root, run_id = backtest_run
    trades = result.trades
    if len(trades) > 0:
        # Every trade pays a non-negative fee and net never exceeds gross.
        assert (trades["fees"] >= 0).all()
        assert (trades["net_pnl"] <= trades["gross_pnl"] + 1e-12).all()


def test_classification_only_model_skipped(backtest_run):
    # Only return-signal models (ridge, no_change) are backtested.
    config, result, root, run_id = backtest_run
    assert set(result.threshold_selection["models"]) <= {"ridge_regression", "no_change"}
