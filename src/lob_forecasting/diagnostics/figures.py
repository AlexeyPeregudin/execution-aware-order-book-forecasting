"""Diagnostic figures, written out as vector PDFs."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from . import _plot_style as ps
from .collect import ReportData

try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _to_array(s):
    return s.to_numpy() if hasattr(s, "to_numpy") else np.asarray(s)


def _hist(series, title: str, xlabel: str, path: Path) -> str | None:
    s = series.dropna()
    if len(s) == 0:
        return None
    arr = _to_array(s)
    ps.apply()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    n_bins = min(65, max(15, len(arr) // max(len(arr) // 500, 1)))
    _, edges, _ = ax.hist(
        arr, bins=n_bins,
        color=ps.C["blue"], alpha=0.78,
        edgecolor=ps.C["white"], linewidth=0.45,
        zorder=3,
    )

    # KDE overlay on a twin y-axis so density and count scales don't fight
    if _HAS_SCIPY and len(arr) > 10:
        try:
            kde = gaussian_kde(arr, bw_method="scott")
            xs = np.linspace(arr.min(), arr.max(), 500)
            ys = kde(xs)
            ax2 = ax.twinx()
            ax2.plot(xs, ys, color=ps.C["navy"], lw=2.1, zorder=5,
                     solid_capstyle="round")
            ax2.fill_between(xs, 0, ys, color=ps.C["blue_light"], alpha=0.55, zorder=4)
            ax2.set_ylim(bottom=0)
            ax2.set_yticks([])
            for sp in ax2.spines.values():
                sp.set_visible(False)
        except Exception:
            pass

    # IQR band + median reference
    med = float(np.median(arr))
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    ax.axvspan(q25, q75, alpha=0.09, color=ps.C["blue"], zorder=1,
               label=f"IQR [{q25:.3g}, {q75:.3g}]")
    ax.axvline(med, color=ps.C["navy"], lw=1.5, ls="--", alpha=0.85, zorder=4,
               label=f"median = {med:.3g}")

    ax.set_xlabel(xlabel, labelpad=7)
    ax.set_ylabel("count", labelpad=7)
    ax.set_title(title)
    ax.set_xlim(edges[0], edges[-1])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    ax.grid(axis="y", color=ps.C["line"], alpha=0.45, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.92)

    return ps.save(fig, path)


def _label_bar(label_df, path: Path) -> str | None:
    if label_df is None or label_df.empty:
        return None
    ps.apply()
    horizons = label_df["horizon"].tolist()
    x = np.arange(len(horizons))
    width = 0.26

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for i, (key, col) in enumerate(
        [("down", ps.C["red"]), ("neutral", ps.C["gray"]), ("up", ps.C["blue"])]
    ):
        vals = label_df[key].to_numpy()
        ax.bar(x + (i - 1) * width, vals, width,
               color=col, alpha=0.85,
               edgecolor=ps.C["white"], linewidth=0.45,
               label=key, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_title("Label distribution by horizon")
    ax.set_xlabel("horizon", labelpad=7)
    ax.set_ylabel("count", labelpad=7)
    ax.set_xlim(-0.55, len(horizons) - 0.45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", color=ps.C["line"], alpha=0.45, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(framealpha=0.92)

    return ps.save(fig, path)


def _pnl_curve(trades, model: str | None, path: Path) -> str | None:
    if trades is None or model is None:
        return None
    t = trades[(trades["model_name"] == model) & (trades["split"] == "test")]
    t = t.sort_values("decision_timestamp_exchange_ns")
    if len(t) == 0:
        return None
    ps.apply()
    cum = t["net_pnl"].cumsum().to_numpy()
    xs = np.arange(1, len(cum) + 1)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(xs, cum, color=ps.C["blue"], lw=2.0, zorder=4, label=model,
            solid_capstyle="round")
    ax.fill_between(xs, 0, cum, where=(cum >= 0),
                    alpha=0.14, color=ps.C["blue"], zorder=2)
    ax.fill_between(xs, 0, cum, where=(cum < 0),
                    alpha=0.18, color=ps.C["red"], zorder=2)
    ax.axhline(0.0, color=ps.C["slate"], lw=0.9, ls="--", alpha=0.7, zorder=3)

    ax.set_xlabel("trade #", labelpad=7)
    ax.set_ylabel("cumulative net PnL", labelpad=7)
    ax.set_title(f"Cumulative net PnL (test) — {model}")
    ax.set_xlim(1, len(cum))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="both", color=ps.C["line"], alpha=0.38, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handlelength=1.6)

    return ps.save(fig, path)


def build_figures(
    rd: ReportData, figdir: Path, best_model: str | None, label_df, ofi_col: str | None
) -> dict[str, str]:
    """Draw all figures into figdir as PDFs; return {key: filename}."""
    figdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    fl = rd.features_labels

    if fl is not None:
        if "relative_spread" in fl.columns:
            name = _hist(fl["relative_spread"],
                         "Relative spread distribution",
                         "relative spread (normalised)",
                         figdir / "spread_distribution.pdf")
            if name:
                out["spread"] = name
        if "imbalance_l1" in fl.columns:
            name = _hist(fl["imbalance_l1"],
                         "Best-level order-book imbalance",
                         "imbalance L1 (normalised)",
                         figdir / "imbalance_distribution.pdf")
            if name:
                out["imbalance"] = name
        if ofi_col and ofi_col in fl.columns:
            name = _hist(fl[ofi_col],
                         f"Order-flow imbalance — {ofi_col}",
                         f"{ofi_col} (normalised)",
                         figdir / "ofi_distribution.pdf")
            if name:
                out["ofi"] = name

    name = _label_bar(label_df, figdir / "label_distribution.pdf")
    if name:
        out["labels"] = name
    name = _pnl_curve(rd.backtest_trades, best_model, figdir / "pnl_curve.pdf")
    if name:
        out["pnl"] = name
    return out
