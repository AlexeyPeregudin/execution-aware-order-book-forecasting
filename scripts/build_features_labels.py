"""Build the features and labels from the book tables.

    python scripts/build_features_labels.py --config configs/experiment/mvp.yaml

Writes features.parquet, then adds the labels and writes features_labels.parquet
plus the fitted threshold file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# so we can run this without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_forecasting.config import load_config, parse_cli_overrides, save_resolved_config
from lob_forecasting.features import build_features
from lob_forecasting.labels import build_labels
from lob_forecasting.orderbook import BookTableIndex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build causal features from book tables.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("overrides", nargs="*", help="Extra key=value config overrides.")
    args = parser.parse_args(argv)

    overrides = parse_cli_overrides(args.overrides)
    config, run_id = load_config(args.config, overrides=overrides)
    save_resolved_config(config, run_id)

    project_root = Path.cwd()
    books_index_path = project_root / config.data.processed_dir / "books" / "books_index.yaml"
    books = BookTableIndex.load(books_index_path)
    if len(books) == 0:
        print(f"[features] no book partitions at {books_index_path}; run build-orderbooks first.")
        return 1

    print(f"[features] run_id={run_id} building features for {len(books)} book partition(s)")
    features = build_features(config, books, project_root)
    print(f"[features] wrote {len(features)} feature partition(s), {features.total_rows()} rows total")
    for p in features.partitions:
        flags = ", ".join(f"{k}={v}" for k, v in p.flag_counts.items()) or "clean"
        print(
            f"  - {p.symbol} [{p.n_dates} date(s)]: {p.row_count} rows, "
            f"{len(p.feature_columns)} features [{flags}] -> {p.file_path}"
        )

    print("[labels] fitting thresholds on training rows and appending labels")
    labelled = build_labels(config, features, run_id, project_root)
    thr = ", ".join(f"{k}={v:.3g}" for k, v in labelled.thresholds.items())
    print(f"[labels] thresholds ({labelled.threshold_source}, alpha={labelled.alpha}): {thr}")
    for p in labelled.partitions:
        dist = "; ".join(
            f"{h}: up/neu/down={d['up']}/{d['neutral']}/{d['down']} (avail={d['available']})"
            for h, d in p.label_distribution.items()
        )
        print(f"  - {p.symbol}: {p.row_count} rows -> {p.file_path}")
        print(f"      {dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
