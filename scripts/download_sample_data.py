"""Get the raw data and write the manifest.

Run it like:
    python scripts/download_sample_data.py --config configs/experiment/mvp.yaml

You can also pass --mode, --overwrite, --verify-only, and key=value overrides
like ingestion.synthetic.num_days=1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# so we can run this without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_forecasting.config import load_config, parse_cli_overrides, save_resolved_config
from lob_forecasting.ingestion import RawDataManifest, ingest_raw_data, verify_raw_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest raw market data and build the manifest.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--mode", help="Override ingestion mode (synthetic|local_archive|url_archive|fixture).")
    parser.add_argument("--overwrite", action="store_true", help="Re-acquire files even if they already exist.")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing manifest files instead of ingesting.")
    parser.add_argument("overrides", nargs="*", help="Extra key=value config overrides.")
    args = parser.parse_args(argv)

    overrides = parse_cli_overrides(args.overrides)
    ing_overrides = overrides.setdefault("ingestion", {})
    if args.mode:
        ing_overrides["mode"] = args.mode
    if args.overwrite:
        ing_overrides["overwrite"] = True

    config, run_id = load_config(args.config, overrides=overrides)
    save_resolved_config(config, run_id)

    project_root = Path.cwd()
    manifest_path = project_root / config.ingestion.manifest_path

    if args.verify_only:
        manifest = RawDataManifest.load(manifest_path)
        verify_raw_files(manifest, project_root)
        print(f"[ingestion] verified {len(manifest)} raw file(s) against {manifest_path}")
        return 0

    print(f"[ingestion] run_id={run_id} mode={config.ingestion.mode}")
    manifest = ingest_raw_data(config, project_root)

    total_rows = sum(e.row_count_or_null or 0 for e in manifest.entries)
    print(f"[ingestion] {len(manifest)} raw file(s), ~{total_rows} rows total")
    print(f"[ingestion] manifest written to {manifest_path}")
    for e in sorted(manifest.entries, key=lambda x: (x.symbol, x.date)):
        print(f"  - {e.symbol} {e.date} {e.file_path} sha256={e.checksum_sha256[:12]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
