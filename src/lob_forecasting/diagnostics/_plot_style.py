"""
Shared visual identity for all diagnostic figures.
Colour scheme matches the LaTeX report palette.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

# colour palette
C: dict[str, str] = {
    "blue":         "#0080FF",
    "red":          "#FF6666",
    "navy":         "#23395B",
    "purple":       "#6F5CE6",
    "ink":          "#253041",
    "gray":         "#667085",
    "slate":        "#5F6B7A",
    "blue_light":   "#EAF4FF",
    "red_light":    "#FFF1F1",
    "panel":        "#F0F3FA",
    "purple_light": "#F3F0FF",
    "paper":        "#F7F9FC",
    "ice":          "#EEF5FF",
    "white":        "#FBFCFF",
    "line":         "#B8C2D1",
    "line_light":   "#E2EAF4",
    "teal":         "#00B4A0",
    "amber":        "#E87C2A",
}

# Ordered palette for multi-series plots (non-baseline first, then baselines)
SERIES: list[str] = [
    C["blue"], C["purple"], C["navy"], C["teal"],
    C["amber"], C["red"], C["gray"], C["slate"],
]

BASELINE_MODELS: frozenset[str] = frozenset({"no_change", "no_change_up", "imbalance_rule"})

POLICY_COLORS: dict[str, str] = {
    "no_quote":                C["red"],
    "naive_symmetric_mm":      C["gray"],
    "inventory_skewed_mm":     C["slate"],
    "forecast_aware_mm":       C["blue"],
    "uncertainty_aware_mm":    C["purple"],
    "control_quote_optimizer": C["teal"],
}

POLICY_LABELS: dict[str, str] = {
    "no_quote":                "No Quote",
    "naive_symmetric_mm":      "Naive Symm.",
    "inventory_skewed_mm":     "Inv. Skewed",
    "forecast_aware_mm":       "Forecast Aware",
    "uncertainty_aware_mm":    "Uncert. Aware",
    "control_quote_optimizer": "Control Opt.",
}

MODEL_LABELS: dict[str, str] = {
    "no_change":            "No Change",
    "no_change_up":         "No Change ↑",
    "imbalance_rule":       "Imbalance Rule",
    "logistic_regression":  "Logistic",
    "ridge_regression":     "Ridge",
    "lightgbm":             "LightGBM",
    "tcn_small":            "TCN Small",
    "tcn_exec_base":        "TCN Exec",
    "tcn_exec_ret0":        "TCN Ret0",
    "tcn_exec_ret0_ssm":    "TCN SSM",
    "tcn_exec_multitask":   "TCN Multitask",
}

MARKERS: list[str] = ["o", "s", "D", "^", "v", "P", "X", "h"]


def apply() -> None:
    """Apply the report's matplotlib style."""
    plt.rcParams.update({
        # typography
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Inter", "Helvetica Neue", "Helvetica",
                                 "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size":            10,
        "axes.titlesize":       11,
        "axes.titleweight":     "600",
        "axes.titlepad":        10,
        "axes.labelsize":       9.5,
        "axes.labelpad":        6,
        "axes.labelcolor":      C["ink"],
        # canvas
        "figure.facecolor":     C["white"],
        "axes.facecolor":       C["paper"],
        "axes.edgecolor":       C["line"],
        # spines
        "axes.linewidth":       0.8,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        # lines
        "lines.linewidth":      2.0,
        "lines.markersize":     5.5,
        "patch.linewidth":      0.6,
        # grid
        "axes.grid":            True,
        "axes.grid.axis":       "y",
        "grid.color":           C["line"],
        "grid.alpha":           0.45,
        "grid.linewidth":       0.55,
        # ticks
        "xtick.color":          C["slate"],
        "ytick.color":          C["slate"],
        "xtick.labelsize":      8.5,
        "ytick.labelsize":      8.5,
        "xtick.major.size":     3.5,
        "ytick.major.size":     3.5,
        "xtick.major.width":    0.7,
        "ytick.major.width":    0.7,
        # legend
        "legend.fontsize":      8.5,
        "legend.framealpha":    0.92,
        "legend.edgecolor":     C["line"],
        "legend.fancybox":      False,
        "legend.borderpad":     0.6,
        "legend.labelspacing":  0.4,
        "legend.handlelength":  1.8,
        # output
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.15,
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
    })


def fmt_month(date_str: str) -> str:
    """Format a date like '2026-04-01' as \"Apr '26\"."""
    try:
        dt = datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d")
        return dt.strftime("%b '%y")
    except Exception:
        return str(date_str)


def save(fig, path: Path) -> str:
    """Save the figure as a PDF and close it. Always writes .pdf, whatever the suffix."""
    fig.tight_layout()
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, format="pdf")
    plt.close(fig)
    return pdf.name
