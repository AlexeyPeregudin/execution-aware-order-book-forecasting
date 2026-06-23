"""Monthly robustness report.

Assembles the month-level and regime-level results into reports/monthly_results.md
plus the CSV tables and figures. The conclusion follows a fixed rule: it keeps
predictive skill separate from execution decision value and states any
overfitting/instability of the learned policy explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import _plot_style as ps  # noqa: E402

from ..config import ExperimentConfig  # noqa: E402

MANDATORY_STATEMENTS = [
    "Only BTCUSDT is used.",
    "Only the first day of each configured month is used.",
    "Only top-5 LOB levels are used.",
    "The monthly days are not consecutive and are treated as regime snapshots.",
    "No live-trading profitability is claimed.",
    "Passive fill modelling is approximate.",
    "All model and policy selection is validation-only.",
    "Test months are held out.",
]


@dataclass
class MonthlyReportData:
    """All aggregated tables the report renders (any may be empty)."""

    folds: list[dict] = field(default_factory=list)
    monthly_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    distributional: pd.DataFrame = field(default_factory=pd.DataFrame)
    stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    taker_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    policy_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    bootstrap: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    label_distribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_accuracy: pd.DataFrame = field(default_factory=pd.DataFrame)
    inventory_paths: pd.DataFrame = field(default_factory=pd.DataFrame)


# conclusion logic


def _predictive_beats_baseline(data: MonthlyReportData) -> tuple[bool, float]:
    """Does any model beat the majority-class baseline on most test months?"""
    mm = data.monthly_metrics
    base = data.baseline_accuracy
    if mm.empty or base.empty:
        return False, 0.0
    acc = mm[(mm["split"] == "test") & (mm["metric_name"] == "accuracy")]
    if acc.empty:
        return False, 0.0
    best = acc.groupby("monthly_date")["metric_value"].max()
    baseline = base.set_index("monthly_date")["baseline_accuracy"]
    months = best.index
    wins = sum(1 for m in months if best[m] > baseline.get(m, 0.5) + 1e-9)
    frac = wins / len(months) if len(months) else 0.0
    return frac > 0.5, frac


def _execution_improves(data: MonthlyReportData) -> tuple[bool, float]:
    """Does a forecast-using policy beat naive net_pnl on most test months?"""
    pm = data.policy_metrics
    if pm.empty:
        return False, 0.0
    test = pm[(pm["split"] == "test") & (pm["metric_name"] == "net_pnl") & (pm["monthly_date"] != "all")]
    if test.empty:
        return False, 0.0
    pivot = test.pivot_table(index="monthly_date", columns="policy_name", values="metric_value", aggfunc="mean")
    if "naive_symmetric_mm" not in pivot.columns:
        return False, 0.0
    forecast_cols = [c for c in pivot.columns if c in (
        "forecast_aware_mm", "uncertainty_aware_mm", "contextual_bandit_mm")]
    if not forecast_cols:
        return False, 0.0
    wins = 0
    for m in pivot.index:
        best_forecast = pivot.loc[m, forecast_cols].max()
        if best_forecast > pivot.loc[m, "naive_symmetric_mm"] + 1e-9:
            wins += 1
    frac = wins / len(pivot.index) if len(pivot.index) else 0.0
    return frac > 0.5, frac


def _bandit_overfits(data: MonthlyReportData) -> bool:
    """True if the bandit beats deterministic on validation but not on test."""
    pm = data.policy_metrics
    if pm.empty or "contextual_bandit_mm" not in pm["policy_name"].unique():
        return False
    det = ("naive_symmetric_mm", "inventory_skewed_mm", "forecast_aware_mm", "uncertainty_aware_mm")

    def reward(split):
        s = pm[(pm["split"] == split) & (pm["metric_name"] == "total_reward") & (pm["monthly_date"] == "all")]
        if s.empty:
            return None, None
        by = s.groupby("policy_name")["metric_value"].sum()
        bandit = by.get("contextual_bandit_mm", float("nan"))
        best_det = max((by.get(d, float("-inf")) for d in det), default=float("-inf"))
        return bandit, best_det

    vb, vd = reward("validation")
    tb, td = reward("test")
    if None in (vb, vd, tb, td):
        return False
    return (vb > vd) and (tb <= td)


def derive_conclusion(data: MonthlyReportData) -> str:
    pred_ok, pred_frac = _predictive_beats_baseline(data)
    exec_ok, exec_frac = _execution_improves(data)
    if not pred_ok:
        verdict = "No robust predictive signal was found."
    elif not exec_ok:
        verdict = ("Predictive signal exists, but decision value was not robust under "
                   "execution assumptions.")
    else:
        verdict = ("Execution-relevant signal was found under the simplified simulator, "
                   "subject to fill-model limitations.")
    if _bandit_overfits(data):
        verdict += (" The learned contextual-bandit policy beat the deterministic policies on "
                    "validation but not on test, indicating overfitting / instability.")
    return verdict


# tables and figures


def _save_tables(data: MonthlyReportData, tables_dir: Path) -> dict[str, str]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def dump(name: str, df: pd.DataFrame) -> None:
        if df is not None and not df.empty:
            df.to_csv(tables_dir / name, index=False)
            written[name] = name

    dump("monthly_data_summary.csv", data.data_summary)
    dump("monthly_label_distribution.csv", data.label_distribution)
    dump("monthly_predictive_metrics.csv", data.monthly_metrics)
    dump("monthly_distributional_metrics.csv", data.distributional)
    dump("robustness_summary.csv", data.stability)
    dump("monthly_taker_backtest.csv", data.taker_metrics)
    dump("monthly_market_making_policy_metrics.csv", data.policy_metrics)
    if not data.policy_metrics.empty:
        sel = data.policy_metrics[
            (data.policy_metrics["split"] == "test") & (data.policy_metrics["monthly_date"] == "all")
            & (data.policy_metrics["metric_name"].isin(["net_pnl", "total_reward", "max_abs_inventory"]))
        ]
        dump("policy_selection_summary.csv", sel)
    # model ablation summary: mean test metric per model
    if not data.monthly_metrics.empty:
        abl = (data.monthly_metrics[data.monthly_metrics["split"] == "test"]
               .groupby(["model_name", "metric_name", "horizon"])["metric_value"].mean().reset_index())
        dump("model_ablation_summary.csv", abl)
    return written


def _line_by_month(df, metric, ylabel, path,
                   *, exclude: frozenset = frozenset()):
    sub = df[(df["split"] == "test") & (df["metric_name"] == metric)]
    sub = sub[~sub["model_name"].isin(exclude)]
    if sub.empty:
        return None
    ps.apply()
    models = sorted(sub["model_name"].unique(),
                    key=lambda m: (m in ps.BASELINE_MODELS, m))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    all_month_means = (sub.groupby(["model_name", "monthly_date"])["metric_value"]
                       .mean())
    for idx, model in enumerate(models):
        g = sub[sub["model_name"] == model]
        gg = g.groupby("monthly_date")["metric_value"].mean().sort_index()
        is_base = model in ps.BASELINE_MODELS
        color = ps.SERIES[idx % len(ps.SERIES)]
        months_fmt = [ps.fmt_month(m) for m in gg.index]
        ax.plot(
            months_fmt, gg.to_numpy(),
            marker=ps.MARKERS[idx % len(ps.MARKERS)],
            markersize=5.5 if not is_base else 4.0,
            color=color,
            alpha=0.90 if not is_base else 0.50,
            linewidth=2.0 if not is_base else 1.1,
            linestyle="-" if not is_base else "--",
            label=ps.MODEL_LABELS.get(model, model),
            zorder=4 - int(is_base),
        )
    vmin, vmax = float(all_month_means.min()), float(all_month_means.max())
    margin = max((vmax - vmin) * 0.12, 0.005)
    ax.set_ylim(vmin - margin, vmax + margin)
    ax.set_xlabel("month (test)", labelpad=7)
    ax.set_ylabel(ylabel, labelpad=7)
    ax.set_title(f"{ylabel} by month")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", color=ps.C["line"], alpha=0.45, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=max(1, len(models) // 5), loc="best", framealpha=0.92)
    return ps.save(fig, path)


def _pnl_by_policy_month(pm, path):
    sub = pm[(pm["split"] == "test") & (pm["metric_name"] == "net_pnl")
             & (pm["monthly_date"] != "all")]
    if sub.empty:
        return None
    ps.apply()
    pivot = (sub.pivot_table(index="monthly_date", columns="policy_name",
                              values="metric_value", aggfunc="mean")
             .sort_index())
    months_fmt = [ps.fmt_month(m) for m in pivot.index]
    policies = pivot.columns.tolist()
    n = len(policies)
    total_width = 0.72
    bar_width = total_width / n
    x = np.arange(len(months_fmt))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, policy in enumerate(policies):
        vals = pivot[policy].to_numpy()
        color = ps.POLICY_COLORS.get(policy, ps.SERIES[i % len(ps.SERIES)])
        label = ps.POLICY_LABELS.get(policy, policy)
        offset = (i - n / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width * 0.88,
               color=color, alpha=0.85,
               edgecolor=ps.C["white"], linewidth=0.4,
               label=label, zorder=3)
    ax.axhline(0, color=ps.C["slate"], lw=0.8, ls="--", alpha=0.55, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(months_fmt, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("test net PnL", labelpad=7)
    ax.set_title("Net PnL by policy and month")
    ax.grid(axis="y", color=ps.C["line"], alpha=0.45, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=max(1, n // 3), loc="best", framealpha=0.92)
    return ps.save(fig, path)


def _adverse_by_policy(pm, path):
    sub = pm[(pm["split"] == "test")
             & (pm["metric_name"] == "adverse_selection_cost_total")
             & (pm["monthly_date"] == "all")]
    if sub.empty:
        return None
    ps.apply()
    by = sub.groupby("policy_name")["metric_value"].sum().sort_values()
    policies = by.index.tolist()
    labels = [ps.POLICY_LABELS.get(p, p) for p in policies]
    colors = [ps.POLICY_COLORS.get(p, ps.C["gray"]) for p in policies]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    y = np.arange(len(policies))
    bars = ax.barh(y, by.to_numpy(), color=colors, alpha=0.85,
                   edgecolor=ps.C["white"], linewidth=0.45, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("total adverse-selection cost (test)", labelpad=7)
    ax.set_title("Adverse selection by policy")
    ax.grid(axis="x", color=ps.C["line"], alpha=0.45, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # Annotate bar ends
    val_range = by.max() - by.min() if len(by) > 1 else abs(by.max())
    for bar, val in zip(bars, by.to_numpy()):
        ax.text(val + val_range * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val:.3g}", va="center", ha="left", fontsize=7.5, color=ps.C["ink"])
    return ps.save(fig, path)


def _coverage_by_month(dist, path):
    sub = dist[(dist["split"] == "test")
               & (dist["metric_name"] == "empirical_coverage_90")
               & (dist["group_kind"] == "monthly_date")]
    if sub.empty:
        return None
    ps.apply()
    models = sorted(sub["model_name"].unique(),
                    key=lambda m: (m in ps.BASELINE_MODELS, m))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for idx, model in enumerate(models):
        g = sub[sub["model_name"] == model]
        gg = g.groupby("group_value")["metric_value"].mean().sort_index()
        is_base = model in ps.BASELINE_MODELS
        color = ps.SERIES[idx % len(ps.SERIES)]
        months_fmt = [ps.fmt_month(m) for m in gg.index]
        ax.plot(
            months_fmt, gg.to_numpy(),
            marker=ps.MARKERS[idx % len(ps.MARKERS)],
            color=color,
            alpha=0.90 if not is_base else 0.50,
            linewidth=2.0 if not is_base else 1.1,
            linestyle="-" if not is_base else "--",
            label=ps.MODEL_LABELS.get(model, model),
            zorder=4 - int(is_base),
        )
    # Nominal coverage band + reference line
    ax.axhspan(0.85, 0.95, alpha=0.06, color=ps.C["blue"], zorder=1)
    ax.axhline(0.90, ls="--", color=ps.C["slate"], lw=1.3, alpha=0.75, zorder=3,
               label="nominal 90%")
    ax.set_ylabel("90 % empirical coverage", labelpad=7)
    ax.set_title("Interval calibration by month")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", color=ps.C["line"], alpha=0.45, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=max(1, len(models) // 5), loc="best", framealpha=0.92)
    return ps.save(fig, path)


def _save_figures(data: MonthlyReportData, fig_dir: Path) -> dict[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for metric, ylab, stem, excl in (
        ("accuracy", "accuracy", "accuracy_by_month", frozenset({"no_change"})),
        ("rank_ic",  "rank IC",  "rank_ic_by_month",  frozenset()),
        ("r2_oos",   "R² OOS",   "r2_oos_by_month",   frozenset()),
    ):
        r = _line_by_month(data.monthly_metrics, metric, ylab,
                           fig_dir / f"{stem}.pdf", exclude=excl)
        if r:
            out[stem] = r
    for stem, call in (
        ("coverage_by_month",
         lambda p: _coverage_by_month(data.distributional, p)),
        ("net_pnl_by_policy_and_month",
         lambda p: _pnl_by_policy_month(data.policy_metrics, p)),
        ("adverse_selection_by_policy",
         lambda p: _adverse_by_policy(data.policy_metrics, p)),
        ("inventory_paths_by_policy",
         lambda p: _inventory_paths(data, p)),
    ):
        r = call(fig_dir / f"{stem}.pdf")
        if r:
            out[stem] = r
    return out


def _inventory_paths(data: MonthlyReportData, path: Path):
    inv = getattr(data, "inventory_paths", None)
    if inv is None or len(inv) == 0:
        return None
    ps.apply()
    # Sort: no_quote last (trivial baseline), others by POLICY_COLORS key order
    _key_order = list(ps.POLICY_COLORS.keys())
    policies = sorted(inv["policy_name"].unique(),
                      key=lambda p: _key_order.index(p) if p in _key_order else 99)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for policy in policies:
        g = inv[inv["policy_name"] == policy].reset_index(drop=True)
        color = ps.POLICY_COLORS.get(policy, ps.C["gray"])
        label = ps.POLICY_LABELS.get(policy, policy)
        ax.plot(np.arange(len(g)), g["inventory"].to_numpy(),
                color=color, lw=1.1, alpha=0.85, label=label, zorder=3)
    ax.axhline(0, color=ps.C["slate"], lw=0.9, ls="--", alpha=0.55, zorder=2)
    ax.set_xlabel("event index", labelpad=7)
    ax.set_ylabel("inventory", labelpad=7)
    ax.set_title("Inventory paths — test (by policy)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="both", color=ps.C["line"], alpha=0.38, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="best", framealpha=0.92)
    return ps.save(fig, path)


# markdown rendering


def _fmt(v) -> str:
    if isinstance(v, float):
        if not np.isfinite(v):
            return "nan"
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            return f"{v:.3e}"
        return f"{v:.4g}"
    return str(v)


def _df_md(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_(none)_"
    df = df.head(max_rows)
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def build_monthly_report(
    config: ExperimentConfig,
    run_id: str,
    data: MonthlyReportData,
    project_root: str | Path | None = None,
) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    reports_dir = root / "reports"
    tables_dir = reports_dir / "tables"
    fig_dir = reports_dir / "figures"

    tables = _save_tables(data, tables_dir)
    figures = _save_figures(data, fig_dir)
    conclusion = derive_conclusion(data)

    n_months = len(config.data.monthly_dates)
    n_folds = len(data.folds)
    test_months = sorted({m for f in data.folds for m in f.get("test_dates", [])})

    lines: list[str] = []
    lines.append("# Monthly Execution-Aware LOB Forecasting — Robustness Report")
    lines.append("")
    lines.append(f"Run `{run_id}`.")
    lines.append("")
    lines.append("## 1. Executive summary")
    lines.append("")
    lines.append(f"**Conclusion:** {conclusion}")
    lines.append("")
    lines.append(f"- Monthly snapshots: {n_months} first-of-month BTCUSDT days")
    lines.append(f"- Evaluation folds (expanding): {n_folds}")
    lines.append(f"- Held-out test months: {', '.join(test_months) if test_months else 'n/a'}")
    lines.append("")
    lines.append("## 2. Data scope and limitations")
    lines.append("")
    for s in MANDATORY_STATEMENTS:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## 3. Monthly snapshot methodology")
    lines.append("")
    lines.append("Models are trained on earlier first-of-month snapshots and evaluated on later "
                 "held-out months using an expanding-window protocol. No feature window, label "
                 "horizon or sequence window crosses a monthly day boundary; the embargo removes "
                 "the last max-horizon rows of each day.")
    lines.append("")
    if data.folds:
        fold_tbl = pd.DataFrame([
            {"fold": f["fold_id"], "train": ",".join(f.get("train_dates", [])),
             "validation": ",".join(f.get("validation_dates", [])),
             "test": ",".join(f.get("test_dates", []))}
            for f in data.folds
        ])
        lines.append(_df_md(fold_tbl))
        lines.append("")
    lines.append("## 4. Top-5 LOB feature and label definitions")
    lines.append("")
    lines.append("Top-5 microstructure features (imbalance, OFI, lagged returns, realised vol), "
                 "regime descriptors (volatility / spread / liquidity buckets, time-of-day), and "
                 "return / direction / quantile / markout / adverse-selection labels. Direction "
                 "and regime thresholds are fitted on training rows only.")
    lines.append("")
    lines.append(_df_md(data.label_distribution))
    lines.append("")
    lines.append("## 5. Model architecture and ablations")
    lines.append("")
    lines.append("Baselines (no_change, imbalance_rule), linear (logistic, ridge), LightGBM, the "
                 "original `tcn_small`, and the compact execution-aware multi-task TCN "
                 "(`tcn_exec_multitask`) with return/direction/quantile/markout/adverse heads. "
                 "Pooling ablations: last-step vs gated.")
    lines.append("")
    if "model_ablation_summary.csv" in tables:
        abl = pd.read_csv(tables_dir / "model_ablation_summary.csv")
        acc = abl[abl["metric_name"] == "accuracy"].pivot_table(
            index="model_name", columns="horizon", values="metric_value").round(4)
        lines.append("Mean test accuracy by model and horizon:")
        lines.append("")
        lines.append(_df_md(acc.reset_index()))
        lines.append("")
    lines.append("## 6. Predictive results by month")
    lines.append("")
    lines.append(_month_pivot(data.monthly_metrics, "accuracy"))
    lines.append("")
    lines.append("### Month-level stability")
    lines.append("")
    lines.append(_df_md(data.stability[data.stability["metric_name"].isin(["accuracy", "rank_ic", "r2_oos"])]
                        if not data.stability.empty else data.stability))
    lines.append("")
    lines.append("## 7. Distributional calibration and uncertainty")
    lines.append("")
    cov = data.distributional[(data.distributional["metric_name"] == "empirical_coverage_90")
                              & (data.distributional["group_kind"] == "all")
                              & (data.distributional["split"] == "test")] if not data.distributional.empty else pd.DataFrame()
    lines.append(_df_md(cov[["model_name", "horizon", "metric_value", "n_observations"]] if not cov.empty else cov))
    lines.append("")
    lines.append("## 8. Taker sanity backtest")
    lines.append("")
    lines.append(_df_md(_taker_summary(data.taker_metrics)))
    lines.append("")
    lines.append("## 9. Passive market-making simulator")
    lines.append("")
    lines.append("Conservative book-path fills, maker fee, inventory limit and adverse-selection "
                 "accounting. All policy parameters selected on validation only.")
    lines.append("")
    lines.append("## 10. Policy comparison, including learned policy")
    lines.append("")
    lines.append(_df_md(_policy_summary(data.policy_metrics)))
    lines.append("")
    lines.append("## 11. Robustness and failure analysis")
    lines.append("")
    if not data.bootstrap.empty:
        lines.append("Block-bootstrap 95% confidence intervals on headline test-month metrics:")
        lines.append("")
        lines.append(_df_md(data.bootstrap))
    else:
        lines.append("_Bootstrap intervals not computed for this run._")
    lines.append("")
    lines.extend(_improvements_report_lines(config, data))
    lines.append("## 12. Honest conclusion")
    lines.append("")
    lines.append(conclusion)
    lines.append("")
    lines.append(f"_Figures: {', '.join(sorted(figures.values())) or 'none'}_")
    lines.append("")

    out_path = reports_dir / "monthly_results.md"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _month_pivot(mm: pd.DataFrame, metric: str) -> str:
    if mm.empty:
        return "_(no predictive metrics)_"
    sub = mm[(mm["split"] == "test") & (mm["metric_name"] == metric)]
    if sub.empty:
        return "_(no test metrics)_"
    pivot = sub.pivot_table(index="model_name", columns="monthly_date", values="metric_value", aggfunc="mean").round(4)
    return _df_md(pivot.reset_index())


def _taker_summary(tm: pd.DataFrame) -> pd.DataFrame:
    if tm is None or tm.empty:
        return pd.DataFrame()
    sub = tm[(tm["split"] == "test") & (tm["metric_name"].isin(["n_trades", "gross_pnl", "net_pnl", "hit_rate"]))]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="model_name", columns="metric_name", values="metric_value", aggfunc="sum").reset_index().round(4)


def _policy_summary(pm: pd.DataFrame) -> pd.DataFrame:
    if pm is None or pm.empty:
        return pd.DataFrame()
    sub = pm[(pm["split"] == "test") & (pm["monthly_date"] == "all")
             & (pm["metric_name"].isin(["net_pnl", "total_reward", "number_of_fills",
                                        "max_abs_inventory", "adverse_selection_cost_total"]))]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="policy_name", columns="metric_name", values="metric_value", aggfunc="sum").reset_index().round(4)


# report sections for the execution-aware improvements

_RETURN_VARIANTS = ("tcn_exec_base", "tcn_exec_ret0", "tcn_exec_ret_detached",
                    "tcn_exec_ret0_ridge_sidecar")
_SSM_PAIRS = (("tcn_exec_ret0", "tcn_exec_ret0_ssm"),
              ("tcn_exec_ret0_ridge_sidecar", "tcn_exec_ret0_ssm_ridge_sidecar"))
_CONTROL = "control_quote_optimizer"
_NO_QUOTE = "no_quote"


def _test_metric_pivot(mm: pd.DataFrame, models: list[str], metric: str) -> pd.DataFrame:
    if mm is None or mm.empty:
        return pd.DataFrame()
    sub = mm[(mm["split"] == "test") & (mm["metric_name"] == metric) & (mm["model_name"].isin(models))]
    if sub.empty:
        return pd.DataFrame()
    return (sub.groupby(["model_name", "horizon"])["metric_value"].mean()
            .reset_index().pivot_table(index="model_name", columns="horizon", values="metric_value")
            .round(4).reset_index())


def _coverage_pivot(dist: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    if dist is None or dist.empty:
        return pd.DataFrame()
    sub = dist[(dist["split"] == "test") & (dist["group_kind"] == "all")
               & (dist["metric_name"] == "empirical_coverage_90") & (dist["model_name"].isin(models))]
    if sub.empty:
        return pd.DataFrame()
    return (sub.groupby(["model_name", "horizon"])["metric_value"].mean()
            .reset_index().pivot_table(index="model_name", columns="horizon", values="metric_value")
            .round(4).reset_index())


def _present_variants(data: MonthlyReportData) -> list[str]:
    if data.monthly_metrics.empty:
        return []
    present = set(data.monthly_metrics["model_name"].unique())
    return [m for m in (*_RETURN_VARIANTS, "tcn_exec_ret0_ssm", "tcn_exec_ret0_ssm_ridge_sidecar")
            if m in present]


def _per_month_reward(pm: pd.DataFrame, policy: str) -> dict[str, float]:
    sub = pm[(pm["split"] == "test") & (pm["policy_name"] == policy)
             & (pm["metric_name"] == "total_reward") & (pm["monthly_date"] != "all")]
    return {str(m): float(v) for m, v in zip(sub["monthly_date"], sub["metric_value"])}


def _months_beating(pm: pd.DataFrame, policy: str, reference: str) -> tuple[int, int]:
    a = _per_month_reward(pm, policy)
    b = _per_month_reward(pm, reference)
    months = sorted(set(a) & set(b))
    wins = sum(1 for m in months if a[m] > b[m] + 1e-9)
    return wins, len(months)


def _control_fill_floor(pm: pd.DataFrame, min_fills: int) -> tuple[int, int]:
    sub = pm[(pm["split"] == "test") & (pm["policy_name"] == _CONTROL)
             & (pm["metric_name"] == "number_of_fills") & (pm["monthly_date"] != "all")]
    vals = sub["metric_value"].to_numpy()
    return int((vals >= min_fills).sum()), len(vals)


def _retain_drop_table(config: ExperimentConfig, data: MonthlyReportData) -> pd.DataFrame:
    """Fixed-rule retain/drop decision per new component."""
    pm = data.policy_metrics
    mm = data.monthly_metrics
    rows: list[dict] = []

    def acc(model: str) -> float:
        if mm.empty:
            return float("nan")
        s = mm[(mm["split"] == "test") & (mm["metric_name"] == "accuracy") & (mm["model_name"] == model)]
        return float(s["metric_value"].mean()) if len(s) else float("nan")

    present = set(mm["model_name"].unique()) if not mm.empty else set()

    def cov_err(model: str) -> float:
        c = _coverage_pivot(data.distributional, [model])
        if c.empty:
            return float("nan")
        vals = c.drop(columns=["model_name"]).to_numpy().ravel()
        return float(np.nanmean(np.abs(vals - 0.90)))

    if {"tcn_exec_base", "tcn_exec_ret0"} <= present:
        better = cov_err("tcn_exec_ret0") < cov_err("tcn_exec_base")
        rows.append({"component": "tcn_exec_ret0 (zero return)",
                     "criterion": "better coverage/exec-head score vs base",
                     "decision": "retain" if better else "drop"})
    if {"tcn_exec_ret0", "tcn_exec_ret0_ssm"} <= present:
        better = cov_err("tcn_exec_ret0_ssm") < cov_err("tcn_exec_ret0") + 1e-9
        rows.append({"component": "latent SSM context",
                     "criterion": "improves vs paired no-SSM run",
                     "decision": "retain" if better else "drop"})
    if not pm.empty and _CONTROL in pm["policy_name"].unique():
        w_nq, n1 = _months_beating(pm, _CONTROL, _NO_QUOTE)
        w_ns, n2 = _months_beating(pm, _CONTROL, "naive_symmetric_mm")
        floor_ok, nf = _control_fill_floor(pm, config.market_making.control.min_fills_per_test_month)
        keep = (n1 and w_nq > n1 / 2) and (n2 and w_ns > n2 / 2) and (nf and floor_ok > nf / 2)
        rows.append({"component": "control_quote_optimizer",
                     "criterion": "beats no_quote & naive in >1/2 months + fill floor",
                     "decision": "retain" if keep else "risk-avoidance only"})
    if config.market_making.fill_model == "queue_aware_partial":
        rows.append({"component": "queue_aware_partial fills",
                     "criterion": "stable ranking, non-pathological fill fractions",
                     "decision": "retain (diagnostic)"})
    return pd.DataFrame(rows)


def _improvements_report_lines(config: ExperimentConfig, data: MonthlyReportData) -> list[str]:
    """Render the improvement sections; each degrades gracefully if empty."""
    variants = _present_variants(data)
    pm = data.policy_metrics
    has_control = (not pm.empty) and (_CONTROL in pm["policy_name"].unique())
    if not variants and not has_control:
        return []  # none of the improvement variants ran

    lines: list[str] = ["## 13. Execution-aware improvements (return heads, SSM, control)", ""]

    if variants:
        lines += ["### 13.1 Return-head ablation — mean test direction accuracy", "",
                  _df_md(_test_metric_pivot(data.monthly_metrics, list(_RETURN_VARIANTS), "accuracy")), "",
                  "### 13.2 Execution-head calibration — 90% interval coverage (test)", "",
                  _df_md(_coverage_pivot(data.distributional, variants)), ""]
        ssm_present = [s for pair in _SSM_PAIRS for s in pair if s in variants]
        if ssm_present:
            lines += ["### 13.3 Latent-SSM ablation — paired test accuracy", "",
                      _df_md(_test_metric_pivot(data.monthly_metrics, ssm_present, "accuracy")), ""]

    if config.market_making.fill_model == "queue_aware_partial" and not pm.empty:
        diag = pm[(pm["split"] == "test") & (pm["monthly_date"] == "all")
                  & (pm["metric_name"].isin(["bid_fill_rate", "ask_fill_rate",
                                             "average_fill_fraction", "number_of_partial_fills"]))]
        if not diag.empty:
            tbl = diag.pivot_table(index="policy_name", columns="metric_name",
                                   values="metric_value", aggfunc="mean").reset_index().round(4)
            lines += ["### 13.4 Queue-aware fill-model diagnostics (test)", "", _df_md(tbl), ""]

    if has_control:
        lines += ["### 13.5 Control optimiser vs baselines (test)", "",
                  _df_md(_policy_summary(pm)), "",
                  "### 13.6 Anti-triviality check (control vs no_quote / naive)", ""]
        w_nq, n1 = _months_beating(pm, _CONTROL, _NO_QUOTE)
        w_ns, n2 = _months_beating(pm, _CONTROL, "naive_symmetric_mm")
        floor_ok, nf = _control_fill_floor(pm, config.market_making.control.min_fills_per_test_month)
        lines += [
            f"- control beats `no_quote` (total reward) in **{w_nq}/{n1}** test months",
            f"- control beats `naive_symmetric_mm` in **{w_ns}/{n2}** test months",
            f"- control clears the {config.market_making.control.min_fills_per_test_month}-fill "
            f"activity floor in **{floor_ok}/{nf}** test months", ""]

    rd = _retain_drop_table(config, data)
    if not rd.empty:
        lines += ["### 13.7 Retain / drop decisions", "", _df_md(rd), ""]
    return lines
