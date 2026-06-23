"""Run the taker backtest.

    python scripts/run_backtest.py --config configs/experiment/mvp.yaml

Reads the predictions and the books, picks each model's threshold on validation,
uses it on test, and writes the trades, metrics and threshold file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# so we can run this without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from lob_forecasting.backtesting import run_backtest
from lob_forecasting.config import load_config, parse_cli_overrides
from lob_forecasting.orderbook import BookTableIndex
from lob_forecasting.utils import read_current_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the taker backtest.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--run-id", help="Run id (defaults to the active run).")
    parser.add_argument("overrides", nargs="*", help="Extra key=value config overrides.")
    args = parser.parse_args(argv)

    overrides = parse_cli_overrides(args.overrides)
    config, _ = load_config(args.config, overrides=overrides)

    project_root = Path.cwd()
    run_id = args.run_id or read_current_run(project_root)
    if run_id is None:
        print("[backtest] no active run; run build-datasets/train first (or pass --run-id).")
        return 1

    pred_dir = project_root / config.data.artefact_dir / "runs" / run_id / "predictions"
    pred_files = sorted(pred_dir.glob("*.parquet"))
    if not pred_files:
        print(f"[backtest] no prediction files in {pred_dir}; train models first.")
        return 1
    predictions = [pd.read_parquet(p) for p in pred_files]

    books_index = project_root / config.data.processed_dir / "books" / "books_index.yaml"
    books = BookTableIndex.load(books_index)
    if len(books) == 0:
        print(f"[backtest] no book partitions at {books_index}; run build-orderbooks first.")
        return 1

    print(f"[backtest] run_id={run_id} horizon={config.backtest.horizon} fee_bps={config.backtest.fee_bps}")
    result = run_backtest(config, predictions, books, project_root, write=True)

    test = result.test_metrics()
    for model, sel in result.threshold_selection["models"].items():
        g = test[(test["model_name"] == model)]
        def m(name):
            r = g[g["metric_name"] == name]["metric_value"]
            return float(r.iloc[0]) if len(r) else float("nan")
        print(f"  {model:<22} theta={sel['selected_threshold']:.2e} "
              f"trades={int(m('n_trades'))} gross={m('gross_pnl'):.4g} net={m('net_pnl'):.4g} "
              f"hit_rate={m('hit_rate'):.3f}")
    print(f"[backtest] artefacts -> artefacts/runs/{run_id}/backtests/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
