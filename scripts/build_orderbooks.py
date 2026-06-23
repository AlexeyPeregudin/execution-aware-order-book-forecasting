"""Rebuild the order books from the event tables.

    python scripts/build_orderbooks.py --config configs/experiment/mvp.yaml

Reads the event index from normalise_events.py and writes the book_top{K}.parquet
files under data/processed/books/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# so we can run this without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_forecasting.config import load_config, parse_cli_overrides, save_resolved_config
from lob_forecasting.normalisation import EventTableIndex
from lob_forecasting.orderbook import build_order_books


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconstruct top-K order books.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("overrides", nargs="*", help="Extra key=value config overrides.")
    args = parser.parse_args(argv)

    overrides = parse_cli_overrides(args.overrides)
    config, run_id = load_config(args.config, overrides=overrides)
    save_resolved_config(config, run_id)

    project_root = Path.cwd()
    events_index_path = project_root / config.data.processed_dir / "events" / "events_index.yaml"
    events = EventTableIndex.load(events_index_path)
    if len(events) == 0:
        print(f"[orderbook] no event partitions at {events_index_path}; run normalise-events first.")
        return 1

    print(f"[orderbook] run_id={run_id} reconstructing top-{config.data.top_k} books")
    index = build_order_books(config, events, project_root)

    print(f"[orderbook] wrote {len(index)} partition(s), {index.total_rows()} book rows total")
    for p in index.partitions:
        print(
            f"  - {p.symbol} {p.date}: {p.row_count} rows "
            f"(clean={p.n_clean}, crossed={p.n_crossed}, missing={p.n_missing_levels}, "
            f"gap={p.n_update_gap}) -> {p.file_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
