"""Train one model and write its predictions.

    python scripts/train_model.py --config configs/experiment/mvp.yaml --model lightgbm

The run id defaults to the active run (the one build_datasets recorded). You can
also pass --run-id.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# so we can run this without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_forecasting.config import load_config, parse_cli_overrides
from lob_forecasting.datasets import DatasetIndex
from lob_forecasting.training import train_and_predict
from lob_forecasting.utils import read_current_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a model and write predictions.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--model", required=True, help="Registered model name to train.")
    parser.add_argument("--run-id", help="Dataset run id (defaults to the active run).")
    parser.add_argument("overrides", nargs="*", help="Extra key=value config overrides.")
    args = parser.parse_args(argv)

    overrides = parse_cli_overrides(args.overrides)
    config, _ = load_config(args.config, overrides=overrides)

    project_root = Path.cwd()
    run_id = args.run_id or read_current_run(project_root)
    if run_id is None:
        print("[train] no active run; run build-datasets first (or pass --run-id).")
        return 1

    meta_path = project_root / config.data.artefact_dir / "runs" / run_id / "dataset_metadata.yaml"
    if not meta_path.exists():
        print(f"[train] dataset metadata not found for run {run_id} at {meta_path}.")
        return 1
    dataset_index = DatasetIndex.load(meta_path)

    print(f"[train] run_id={run_id} training model={args.model}")
    preds = train_and_predict(config, dataset_index, args.model, project_root)

    n_test = int((preds["split"] == "test").sum())
    print(f"[train] wrote {len(preds)} prediction rows ({n_test} for test)")
    print(f"[train] artefacts under artefacts/runs/{run_id}/ (models, predictions, logs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
