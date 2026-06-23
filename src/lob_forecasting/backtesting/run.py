"""Run the backtest for every model that predicts a return.

For each such model we pick the threshold on validation, freeze it, and use it
on test. Classification-only models (no predicted return) are skipped, since the
policy needs a return to threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..config import ExperimentConfig
from .engine import TRADE_COLUMNS, build_book_sequences, compute_metrics, simulate, simulate_split

METRIC_COLUMNS = (
    "run_id", "model_name", "split", "venue", "symbol", "horizon",
    "metric_name", "metric_value", "n_observations", "created_at_utc",
)


class BacktestError(RuntimeError):
    """The backtest couldn't run (e.g. no predictions given)."""


@dataclass
class BacktestResult:
    """The trades, metrics, and which threshold was chosen for each model."""

    trades: pd.DataFrame
    metrics: pd.DataFrame
    threshold_selection: dict[str, Any]
    run_id: str = ""

    def test_metrics(self) -> pd.DataFrame:
        return self.metrics[self.metrics["split"] == "test"]

    def save(self, config: ExperimentConfig, project_root: str | Path | None = None) -> Path:
        root = Path(project_root) if project_root is not None else Path.cwd()
        out = root / config.data.artefact_dir / "runs" / self.run_id / "backtests"
        out.mkdir(parents=True, exist_ok=True)
        self.trades.to_parquet(out / "taker_trades.parquet", engine="pyarrow", index=False)
        self.metrics.to_parquet(out / "taker_metrics.parquet", engine="pyarrow", index=False)
        with (out / "threshold_selection.yaml").open("w", encoding="utf-8") as fh:
            yaml.dump(self.threshold_selection, fh, default_flow_style=False, sort_keys=False)
        return out


def _select_threshold(val_df: pd.DataFrame, seqs, cfg, grid: list[float]) -> tuple[float, list[dict]]:
    """Try each threshold on validation, keep the one with the best net PnL."""
    results: list[dict] = []
    best_theta = grid[0]
    best_net = -np.inf
    for theta in grid:
        trades, _, _ = simulate_split(val_df, seqs, float(theta), cfg)
        net = float(sum(t["net_pnl"] for t in trades))
        results.append({"threshold": float(theta), "validation_net_pnl": net, "n_trades": len(trades)})
        if net > best_net + 1e-15:
            best_net = net
            best_theta = float(theta)
    return best_theta, results


def _trades_frame(trades: list[dict], run_id, model, split) -> pd.DataFrame:
    keep = (
        "venue", "symbol", "horizon", "decision_event_id",
        "decision_timestamp_exchange_ns", "side", "exec_event_id",
        "exec_price", "mark_price", "trade_size", "gross_pnl", "fees",
        "net_pnl", "threshold",
    )
    rows = []
    for t in trades:
        row = {"run_id": run_id, "model_name": model, "split": split}
        for k in keep:
            row[k] = t[k]
        rows.append(row)
    return pd.DataFrame(rows, columns=list(TRADE_COLUMNS))


def run_backtest(
    config: ExperimentConfig,
    predictions: list[pd.DataFrame],
    books,
    project_root: str | Path | None = None,
    write: bool = True,
) -> BacktestResult:
    """Run the taker backtest for every return-predicting model."""
    if not predictions:
        raise BacktestError("No prediction frames supplied.")

    root = Path(project_root) if project_root is not None else Path.cwd()
    cfg = config.backtest
    h = cfg.horizon
    grid = list(cfg.threshold_grid)
    created = datetime.now(timezone.utc).isoformat()
    run_id = str(predictions[0]["run_id"].iloc[0])

    seqs = build_book_sequences(books, root)

    all_trades: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    selection: dict[str, Any] = {
        "horizon": h, "fee_bps": cfg.fee_bps, "latency_events": cfg.latency_events,
        "max_position": cfg.max_position, "trade_size": cfg.trade_size,
        "threshold_grid": grid, "objective": "validation_net_pnl", "models": {},
    }

    for frame in predictions:
        model = str(frame["model_name"].iloc[0])
        sub = frame[frame["horizon"] == h]
        if sub["pred_return"].notna().sum() == 0:
            continue  # no return predicted, can't trade on it

        val_df = sub[sub["split"] == "validation"]
        theta, grid_results = _select_threshold(val_df, seqs, cfg, grid)
        selection["models"][model] = {"selected_threshold": theta, "grid": grid_results}

        # now use the frozen threshold on validation (for reference) and test
        for split in ("validation", "test"):
            split_df = sub[sub["split"] == split]
            for (venue, symbol), g in split_df.groupby(["venue", "symbol"]):
                seq = seqs.get((venue, symbol))
                if seq is None:
                    continue
                trades, n_exc, n_lim = simulate(
                    g, seq, threshold=theta, horizon=h, latency_events=cfg.latency_events,
                    fee_bps=cfg.fee_bps, trade_size=cfg.trade_size, max_position=cfg.max_position,
                )
                all_trades.append(_trades_frame(trades, run_id, model, split))

                # range of event positions in this split, for the Sharpe buckets
                pos = g["timestamp_exchange_ns"].map(lambda t: seq.ts_to_pos.get(int(t), -1))
                pos = pos[pos >= 0]
                pos_min = int(pos.min()) if len(pos) else 0
                pos_max = int(pos.max()) if len(pos) else 0
                mets = compute_metrics(trades, pos_min, pos_max, n_exc, n_lim, theta)
                for name, value in mets.items():
                    metric_rows.append({
                        "run_id": run_id, "model_name": model, "split": split,
                        "venue": venue, "symbol": symbol, "horizon": h,
                        "metric_name": name, "metric_value": float(value),
                        "n_observations": len(trades), "created_at_utc": created,
                    })

    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
    else:
        trades_df = pd.DataFrame(columns=list(TRADE_COLUMNS))
    metrics_df = pd.DataFrame(metric_rows, columns=list(METRIC_COLUMNS))
    result = BacktestResult(trades_df, metrics_df, selection, run_id=run_id)
    if write:
        result.save(config, root)
    return result
