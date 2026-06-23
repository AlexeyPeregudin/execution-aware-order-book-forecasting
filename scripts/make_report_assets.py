"""Build the report: tables, figures and reports/mvp_results.md.

    python scripts/make_report_assets.py --config configs/experiment/mvp.yaml

Reads everything the earlier steps produced and puts together the report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# so we can run this without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_forecasting.config import load_config, parse_cli_overrides
from lob_forecasting.diagnostics import build_report
from lob_forecasting.utils import read_current_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the MVP report assets.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--run-id", help="Run id (defaults to the active run).")
    parser.add_argument("overrides", nargs="*", help="Extra key=value config overrides.")
    args = parser.parse_args(argv)

    overrides = parse_cli_overrides(args.overrides)
    config, _ = load_config(args.config, overrides=overrides)

    project_root = Path.cwd()
    run_id = args.run_id or read_current_run(project_root)
    if run_id is None:
        print("[report] no active run; run the pipeline first (or pass --run-id).")
        return 1

    print(f"[report] run_id={run_id} building report assets")
    result = build_report(config, run_id, project_root)

    print(f"[report] tables: {len(result.tables)} | figures: {len(result.figures)}")
    print(f"[report] conclusion: {result.conclusion}")
    print(f"[report] report -> {result.markdown_path}")
    print(f"[report] self-contained copy -> {result.report_assets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
