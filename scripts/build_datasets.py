"""Build the train/validation/test datasets.

    python scripts/build_datasets.py --config configs/experiment/mvp.yaml

Reads features_labels and writes data/datasets/{run_id}/ plus the scaler and the
dataset metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# so we can run this without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_forecasting.config import load_config, parse_cli_overrides, save_resolved_config
from lob_forecasting.datasets import build_datasets
from lob_forecasting.labels import LabelledTableIndex
from lob_forecasting.utils import write_current_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build train/validation/test datasets.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("overrides", nargs="*", help="Extra key=value config overrides.")
    args = parser.parse_args(argv)

    overrides = parse_cli_overrides(args.overrides)
    config, run_id = load_config(args.config, overrides=overrides)
    save_resolved_config(config, run_id)

    project_root = Path.cwd()
    labelled_index_path = project_root / config.data.processed_dir.parent / "features" / "labelled_index.yaml"
    labelled = LabelledTableIndex.load(labelled_index_path)
    if len(labelled) == 0:
        print(f"[datasets] no features-labels at {labelled_index_path}; run build-features-labels first.")
        return 1

    print(f"[datasets] run_id={run_id} building datasets (L={config.datasets.sequence_length})")
    index = build_datasets(config, labelled, run_id, project_root)

    # remember this run id so the later steps use the same one
    write_current_run(run_id, project_root)

    print(f"[datasets] scaler -> {index.scaler_path}")
    for s in index.stats:
        print(f"  - {s.split}: {s.n_tabular} tabular rows, {s.n_sequence} sequence windows")
    print(f"[datasets] metadata -> artefacts/runs/{run_id}/dataset_metadata.yaml")
    print(f"[datasets] active run recorded: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
