"""Run only the data-prep stages (ingest -> normalise -> order books -> features
-> labels) for a monthly config, caching the features-labels table so a later
pipeline run reuses it. Useful for staging a long run: do the shared, cheap data
prep once and time it before committing to model training.

    python scripts/prep_features.py --config configs/experiment/<cfg>.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_forecasting.config import load_config
from lob_forecasting.features import build_features
from lob_forecasting.ingestion import RawDataManifest, ingest_raw_data
from lob_forecasting.labels import build_labels
from lob_forecasting.normalisation import EventTableIndex, normalise_events
from lob_forecasting.orderbook import BookTableIndex, build_order_books


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default="PREP")
    args = ap.parse_args(argv)

    root = Path.cwd()
    cfg, _ = load_config(args.config)
    t0 = time.time()

    def stamp(msg: str) -> None:
        print(f"[prep +{time.time()-t0:5.0f}s] {msg}", flush=True)

    stamp("ingesting raw data")
    ingest_raw_data(cfg, root)
    manifest = RawDataManifest.load(root / cfg.ingestion.manifest_path)

    stamp("normalising events")
    normalise_events(cfg, manifest, root)
    events = EventTableIndex.load(root / cfg.data.processed_dir / "events" / "events_index.yaml")

    stamp("reconstructing order books (top-5)")
    build_order_books(cfg, events, root)
    books = BookTableIndex.load(root / cfg.data.processed_dir / "books" / "books_index.yaml")

    stamp("building features")
    features = build_features(cfg, books, root)

    stamp("building labels")
    labelled = build_labels(cfg, features, args.run_id, root)

    stamp(f"DONE: {labelled.total_rows():,} feature-label rows across "
          f"{len(cfg.data.monthly_dates)} monthly days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
