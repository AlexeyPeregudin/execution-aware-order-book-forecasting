"""Run the passive market-making simulator for an existing monthly run.

    python scripts/run_market_making.py --config configs/experiment/btcusdt_top5_monthly.yaml --run-id RUN

Loads the per-fold multi-task predictions written by run_monthly_robustness.py
and replays every configured policy (validation-only selection) on each fold,
writing orders / fills / inventory / policy metrics under the run's
market_making folders. The full pipeline already runs this; the script lets you
re-run the simulator without retraining.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from lob_forecasting.backtesting import run_market_making
from lob_forecasting.config import load_config, parse_cli_overrides
from lob_forecasting.datasets import generate_folds
from lob_forecasting.labels import LabelledTableIndex
from lob_forecasting.utils import read_current_run

MULTITASK = "tcn_exec_multitask"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Passive market-making simulator.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", help="Run id (defaults to the active run).")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    config, _ = load_config(args.config, overrides=parse_cli_overrides(args.overrides))
    root = Path.cwd()
    run_id = args.run_id or read_current_run(root)
    if run_id is None:
        print("[mm] no run id given and no active run found.")
        return 1
    if not config.market_making.enabled:
        print("[mm] market_making.enabled is false in this config.")
        return 1

    labelled = LabelledTableIndex.load(root / config.data.processed_dir.parent / "features" / "labelled_index.yaml")
    fl = pd.concat([pd.read_parquet(root / lp.file_path) for lp in labelled.partitions], ignore_index=True)
    run_root = root / config.data.artefact_dir / "runs" / run_id

    for fold in generate_folds(config):
        pred_path = run_root / "folds" / fold.name / "predictions" / f"{MULTITASK}.parquet"
        if not pred_path.exists():
            print(f"[mm] {fold.name}: no {MULTITASK} predictions at {pred_path}; skipping.")
            continue
        preds = pd.read_parquet(pred_path)
        res = run_market_making(config, fold, preds, fl, run_id, root)
        out = run_root / "folds" / fold.name / "market_making"
        res.save(out)
        test = res.policy_metrics[(res.policy_metrics["split"] == "test")
                                  & (res.policy_metrics["monthly_date"] == "all")
                                  & (res.policy_metrics["metric_name"] == "net_pnl")]
        summary = ", ".join(f"{r.policy_name}={r.metric_value:.2f}" for r in test.itertuples())
        print(f"[mm] {fold.name} test net_pnl: {summary}")
        print(f"[mm] {fold.name} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
