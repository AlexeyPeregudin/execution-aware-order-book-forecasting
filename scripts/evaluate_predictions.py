"""Compute the metrics from the prediction files.

    python scripts/evaluate_predictions.py --config configs/experiment/mvp.yaml

Reads all the prediction parquets for the run and writes the metrics, confusion
and calibration tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# so we can run this without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from lob_forecasting.config import load_config, parse_cli_overrides
from lob_forecasting.evaluation import evaluate_predictions
from lob_forecasting.utils import read_current_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--run-id", help="Run id (defaults to the active run).")
    parser.add_argument("overrides", nargs="*", help="Extra key=value config overrides.")
    args = parser.parse_args(argv)

    overrides = parse_cli_overrides(args.overrides)
    config, _ = load_config(args.config, overrides=overrides)

    project_root = Path.cwd()
    run_id = args.run_id or read_current_run(project_root)
    if run_id is None:
        print("[evaluate] no active run; run build-datasets/train first (or pass --run-id).")
        return 1

    pred_dir = project_root / config.data.artefact_dir / "runs" / run_id / "predictions"
    pred_files = sorted(pred_dir.glob("*.parquet"))
    if not pred_files:
        print(f"[evaluate] no prediction files in {pred_dir}; train models first.")
        return 1

    predictions = [pd.read_parquet(p) for p in pred_files]
    print(f"[evaluate] run_id={run_id} evaluating {len(predictions)} model(s)")
    result = evaluate_predictions(config, predictions, project_root, write=True)

    # print a quick leaderboard for the main metrics on the test split
    test = result.test_metrics()
    for metric in ("accuracy", "r2_oos"):
        sub = test[test["metric_name"] == metric]
        if len(sub) == 0:
            continue
        print(f"  test {metric} (by model, mean over horizons):")
        for model, g in sub.groupby("model_name"):
            print(f"    {model:<22} {g['metric_value'].mean():.4f}")
    print(f"[evaluate] metrics -> artefacts/runs/{run_id}/metrics/predictive_metrics.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
