"""Put the report together: write the tables, draw the figures, build the markdown."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from .collect import ReportData, load_report_data
from .figures import build_figures
from .tables import build_tables, label_distribution_table


@dataclass
class ReportResult:
    markdown_path: Path
    tables: dict[str, Path] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)
    report_assets_dir: Path | None = None
    conclusion: str = ""
    best_model: str | None = None


def _fmt(v) -> str:
    """Format a value for a markdown table cell."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float):
        # use scientific notation for very small/large numbers
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            return f"{v:.3e}"
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def _df_to_md(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a DataFrame as a markdown table."""
    if df is None or df.empty:
        return "_(no data)_"
    df = df.head(max_rows)
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


@dataclass
class Analysis:
    n_test: int | None = None
    best_acc_model: str | None = None
    best_acc: float = float("nan")
    baseline_acc: float = float("nan")
    best_r2_model: str | None = None
    best_r2: float = float("nan")
    backtest_model: str | None = None
    backtest_test_net: float = float("nan")
    signal: bool = False
    survived: bool = False
    conclusion: str = ""


def _majority_class_accuracy(rd: ReportData) -> float:
    """The accuracy you'd get by always guessing the most common test class.

    This is the honest baseline to beat. Computed from the confusion matrix.
    """
    if rd.confusion is None:
        return float("nan")
    c = rd.confusion[rd.confusion["split"] == "test"]
    if len(c) == 0:
        return float("nan")
    # the true labels are the same for every model, so just use the first one
    c = c[c["model_name"] == c["model_name"].iloc[0]]
    fracs = []
    for _, g in c.groupby("horizon"):
        total = g["count"].sum()
        if total == 0:
            continue
        by_true = g.groupby("true_class")["count"].sum()
        fracs.append(float(by_true.max()) / float(total))
    return float(np.mean(fracs)) if fracs else float("nan")


def analyse(rd: ReportData) -> Analysis:
    """Work out the headline numbers and the one-line conclusion."""
    a = Analysis()
    if rd.datasets is not None and rd.datasets.stat("test"):
        a.n_test = rd.datasets.stat("test").n_tabular

    a.baseline_acc = _majority_class_accuracy(rd)

    pm = rd.predictive_metrics
    if pm is not None:
        test = pm[pm["split"] == "test"]
        acc = test[test["metric_name"] == "accuracy"]
        if len(acc):
            by_model = acc.groupby("model_name")["metric_value"].mean()
            a.best_acc_model = str(by_model.idxmax())
            a.best_acc = float(by_model.max())
            # fall back to no_change's accuracy if we couldn't get a majority baseline
            if np.isnan(a.baseline_acc) and "no_change" in by_model.index:
                a.baseline_acc = float(by_model["no_change"])
        r2 = test[test["metric_name"] == "r2_oos"]
        if len(r2):
            by_model = r2.groupby("model_name")["metric_value"].mean()
            a.best_r2_model = str(by_model.idxmax())
            a.best_r2 = float(by_model.max())

    # the backtest model is whichever did best on validation net PnL
    bt = rd.backtest_metrics
    if bt is not None:
        val_net = bt[(bt["split"] == "validation") & (bt["metric_name"] == "net_pnl")]
        if len(val_net):
            by_model = val_net.groupby("model_name")["metric_value"].sum()
            a.backtest_model = str(by_model.idxmax())
            test_net = bt[(bt["split"] == "test") & (bt["metric_name"] == "net_pnl") & (bt["model_name"] == a.backtest_model)]
            a.backtest_test_net = float(test_net["metric_value"].sum()) if len(test_net) else float("nan")

    # do we beat the baseline, and does it survive the backtest?
    baseline = a.baseline_acc if not np.isnan(a.baseline_acc) else 1.0 / 3.0
    acc_signal = (not np.isnan(a.best_acc)) and a.best_acc > baseline + 0.02
    r2_signal = (not np.isnan(a.best_r2)) and a.best_r2 > 0.0
    a.signal = bool(acc_signal or r2_signal)
    a.survived = bool(a.signal and not np.isnan(a.backtest_test_net) and a.backtest_test_net > 0)

    small = a.n_test is not None and a.n_test < 300
    if not a.signal:
        a.conclusion = "There is **no reliable predictive signal** in the MVP sample."
    elif not a.survived:
        a.conclusion = "There is **predictive signal but it does not survive** the simple transaction-cost and latency model."
    else:
        a.conclusion = "There is **predictive signal and it survives** the simple cost model."
    if small:
        a.conclusion += (
            f" Note: the test split has only {a.n_test} usable rows, so this sample is likely "
            "**too small to make a stable claim**; treat the result as indicative only."
        )
    return a


def _render_markdown(rd: ReportData, tables: dict[str, pd.DataFrame], figures: dict[str, str], a: Analysis) -> str:
    c = rd.config
    lines: list[str] = []

    def tbl(key: str) -> str:
        return _df_to_md(tables.get(key, pd.DataFrame()))

    def fig(key: str, caption: str) -> str:
        if key in figures:
            return f"\n![{caption}](figures/{figures[key]})\n"
        return ""

    lines.append("# MVP Results - Execution-Aware Multi-Horizon LOB Forecasting\n")
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()} for run `{rd.run_id}`._\n")

    lines.append("## 1. Research question and conclusion\n")
    lines.append("> Does a simple order-book forecasting pipeline produce statistically measurable "
                 "short-horizon signals, and do those signals survive a basic transaction-cost and "
                 "latency sanity check?\n")
    lines.append(a.conclusion + "\n")
    if a.best_acc_model:
        lines.append(f"- Best directional accuracy (test): **{a.best_acc_model}** at {_fmt(a.best_acc)} "
                     f"(majority-class baseline {_fmt(a.baseline_acc)}).")
    if a.best_r2_model:
        lines.append(f"- Best out-of-sample R2 (test): **{a.best_r2_model}** at {_fmt(a.best_r2)}.")
    if a.backtest_model:
        lines.append(f"- Validation-selected backtest model **{a.backtest_model}** had test net PnL "
                     f"{_fmt(a.backtest_test_net)}.")
    lines.append("")

    lines.append("## 2. Data used\n")
    lines.append(f"- Venue: `{c.data.venue}` | Symbols: `{', '.join(c.data.symbols)}` | "
                 f"Date range: `{c.data.start_date} to {c.data.end_date}` | top-K: {c.data.top_k} | "
                 f"Ingestion mode: `{c.ingestion.mode}`\n")
    lines.append("**Row counts by pipeline stage**\n")
    lines.append(tbl("row_counts") + "\n")
    lines.append("**Data-quality flag counts**\n")
    if "quality_flags" in tables:
        lines.append(tbl("quality_flags") + "\n")
    else:
        lines.append("_No quality flags raised (clean sample)._\n")

    lines.append("## 3. Labels predicted\n")
    lines.append(f"- Horizons (events): {c.sampling.horizons_events} | "
                 f"Threshold mode: `{c.labels.direction_threshold_mode}` (alpha={c.labels.direction_threshold_alpha})\n")
    lines.append("**Label distribution by horizon**\n")
    lines.append(tbl("label_distribution"))
    lines.append(fig("labels", "Label distribution by horizon"))

    lines.append("## 4. Feature diagnostics\n")
    lines.append("**Feature summary statistics**\n")
    lines.append(tbl("feature_summary") + "\n")
    lines.append(fig("spread", "Relative spread distribution"))
    lines.append(fig("imbalance", "Best-level imbalance distribution"))
    lines.append(fig("ofi", "OFI distribution"))

    lines.append("\n## 5. Temporal splits\n")
    lines.append(f"**Train / validation / test ranges** (embargo {c.splits.embargo_events} events)\n")
    lines.append(tbl("split_time_ranges") + "\n")

    lines.append("## 6. Models compared and predictive performance\n")
    lines.append(f"- Models run: {', '.join(c.models.run)}\n")
    lines.append("**Predictive metrics (test split)**\n")
    lines.append(tbl("predictive_metrics_test") + "\n")
    if "confusion_matrices_test" in tables and a.best_acc_model:
        conf = tables["confusion_matrices_test"]
        conf = conf[conf["model_name"] == a.best_acc_model]
        lines.append(f"**Confusion matrix (test) - {a.best_acc_model}**\n")
        lines.append(_df_to_md(conf) + "\n")

    lines.append("## 7. Execution-aware backtest\n")
    if rd.threshold_selection is not None:
        ts = rd.threshold_selection
        lines.append(f"- Selected horizon: {ts.get('horizon')} | fee_bps: {ts.get('fee_bps')} | "
                     f"latency_events: {ts.get('latency_events')} | max_position: {ts.get('max_position')}\n")
    lines.append("**Taker backtest metrics (test split)**\n")
    if "backtest_metrics_test" in tables:
        lines.append(tbl("backtest_metrics_test") + "\n")
    else:
        lines.append("_No backtest results._\n")
    lines.append(fig("pnl", "Cumulative net PnL on test"))

    lines.append("\n## 8. Limitations\n")
    lines.append("- The MVP does not claim real-world profitability; the backtest is a simplified "
                 "taker sanity check (single fill at the touch, fixed fee, integer-event latency).")
    if c.ingestion.mode == "synthetic":
        lines.append("- Data is **synthetic** (a seeded random walk), so by construction there is no "
                     "genuine micro-structure signal to recover; replace with real venue archives for "
                     "a meaningful study.")
    if a.n_test is not None and a.n_test < 300:
        lines.append(f"- The test split is small ({a.n_test} rows); metrics are high-variance.")
    lines.append("- Single venue / symbol scope; no passive fills, queue position or adverse selection.")
    lines.append("")
    return "\n".join(lines)


def build_report(
    config: ExperimentConfig, run_id: str, project_root: str | Path | None = None
) -> ReportResult:
    """Build the report for a run: tables, figures, mvp_results.md, and a copy under the run."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    rd = load_report_data(config, run_id, root)

    reports_dir = root / "reports"
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # write the tables
    tables = build_tables(rd)
    table_paths: dict[str, Path] = {}
    for name, df in tables.items():
        p = tables_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        table_paths[name] = p

    # work out the conclusion and draw the figures
    a = analyse(rd)
    if config.sampling.feature_lookbacks_events:
        ofi_col = f"ofi_{config.sampling.feature_lookbacks_events[0]}"
    else:
        ofi_col = None
    figures = build_figures(rd, figures_dir, a.backtest_model, label_distribution_table(rd), ofi_col)

    # write the markdown
    markdown = _render_markdown(rd, tables, figures, a)
    md_path = reports_dir / "mvp_results.md"
    md_path.write_text(markdown, encoding="utf-8")

    # keep a self-contained copy under the run folder
    assets_dir = root / config.data.artefact_dir / "runs" / run_id / "report_assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    (assets_dir / "tables").mkdir(parents=True, exist_ok=True)
    (assets_dir / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy2(md_path, assets_dir / "mvp_results.md")
    for p in table_paths.values():
        shutil.copy2(p, assets_dir / "tables" / p.name)
    for fname in figures.values():
        shutil.copy2(figures_dir / fname, assets_dir / "figures" / fname)

    return ReportResult(
        markdown_path=md_path, tables=table_paths, figures=figures,
        report_assets_dir=assets_dir, conclusion=a.conclusion, best_model=a.backtest_model,
    )
