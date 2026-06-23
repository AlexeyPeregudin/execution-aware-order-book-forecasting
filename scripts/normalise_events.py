"""Normalise the raw data into event tables.

    python scripts/normalise_events.py --config configs/experiment/mvp.yaml

Reads the manifest from download_sample_data.py and writes the events.parquet
files under data/processed/events/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# so we can run this without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_forecasting.config import load_config, parse_cli_overrides, save_resolved_config
from lob_forecasting.ingestion import RawDataManifest
from lob_forecasting.normalisation import normalise_events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalise raw data into event tables.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("overrides", nargs="*", help="Extra key=value config overrides.")
    args = parser.parse_args(argv)

    overrides = parse_cli_overrides(args.overrides)
    config, run_id = load_config(args.config, overrides=overrides)
    save_resolved_config(config, run_id)

    project_root = Path.cwd()
    manifest_path = project_root / config.ingestion.manifest_path
    manifest = RawDataManifest.load(manifest_path)
    if len(manifest) == 0:
        print(f"[normalise] no raw files in {manifest_path}; run download-sample-data first.")
        return 1

    print(f"[normalise] run_id={run_id} normalising {len(manifest)} raw file(s)")
    index = normalise_events(config, manifest, project_root)

    print(f"[normalise] wrote {len(index)} partition(s), {index.total_rows()} event rows total")
    for p in index.partitions:
        flags = ", ".join(f"{k}={v}" for k, v in p.flag_counts.items()) or "clean"
        print(f"  - {p.symbol} {p.date}: {p.row_count} rows [{flags}] -> {p.file_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
