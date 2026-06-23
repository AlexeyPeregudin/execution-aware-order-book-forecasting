"""Build the CSV tables that go into the report."""

from __future__ import annotations

import pandas as pd

from .collect import ReportData


def _ns_to_iso(ns: int | None) -> str:
    if ns is None:
        return ""
    return pd.Timestamp(int(ns), unit="ns", tz="UTC").isoformat()


def _horizon_num(h: str) -> int:
    # turn "h50" into 50 so horizons sort numerically
    return int(str(h).lstrip("h"))


def row_counts_table(rd: ReportData) -> pd.DataFrame:
    rows: list[tuple[str, int]] = []
    if rd.manifest is not None:
        rows.append(("raw_files", len(rd.manifest)))
        rows.append(("raw_rows", sum(e.row_count_or_null or 0 for e in rd.manifest.entries)))
    if rd.events is not None:
        rows.append(("normalised_events", rd.events.total_rows()))
    if rd.books is not None:
        rows.append(("book_snapshots", rd.books.total_rows()))
    if rd.labelled is not None:
        rows.append(("features_labels", rd.labelled.total_rows()))
    if rd.datasets is not None:
        for s in rd.datasets.stats:
            rows.append((f"dataset_{s.split}_tabular", s.n_tabular))
            rows.append((f"dataset_{s.split}_sequence", s.n_sequence))
    return pd.DataFrame(rows, columns=["stage", "count"])


def quality_flags_table(rd: ReportData) -> pd.DataFrame:
    rows: list[dict] = []
    for stage, index in (("events", rd.events), ("books", rd.books), ("features", rd.features)):
        if index is None:
            continue
        for part in index.partitions:
            for flag, count in (part.flag_counts or {}).items():
                rows.append({"stage": stage, "flag": flag, "count": int(count)})
    if not rows:
        return pd.DataFrame(columns=["stage", "flag", "count"])
    return pd.DataFrame(rows).groupby(["stage", "flag"], as_index=False)["count"].sum()


def feature_summary_table(rd: ReportData) -> pd.DataFrame:
    if rd.features_labels is None:
        return pd.DataFrame()
    df = rd.features_labels
    cols = []
    for c in ("relative_spread", "spread", "imbalance_l1", "imbalance_lK"):
        if c in df.columns:
            cols.append(c)
    for c in df.columns:
        if c.startswith("ofi_"):
            cols.append(c)
    if not cols:
        return pd.DataFrame()
    desc = df[cols].describe().T  # count, mean, std, min, quartiles, max
    desc.insert(0, "feature", desc.index)
    return desc.reset_index(drop=True)


def label_distribution_table(rd: ReportData) -> pd.DataFrame:
    if rd.labelled is None or not len(rd.labelled):
        return pd.DataFrame()
    # add up the per-horizon counts across all the symbols
    agg: dict[str, dict[str, int]] = {}
    for lp in rd.labelled.partitions:
        for h, d in (lp.label_distribution or {}).items():
            acc = agg.setdefault(h, {"available": 0, "up": 0, "neutral": 0, "down": 0})
            for k in acc:
                acc[k] += int(d.get(k, 0))
    rows = []
    for h in sorted(agg, key=_horizon_num):
        thr = (rd.labelled.thresholds or {}).get(h)
        counts = agg[h]
        rows.append({
            "horizon": h, "threshold": thr,
            "available": counts["available"], "up": counts["up"],
            "neutral": counts["neutral"], "down": counts["down"],
        })
    return pd.DataFrame(rows)


def split_time_ranges_table(rd: ReportData) -> pd.DataFrame:
    if rd.datasets is None:
        return pd.DataFrame()
    rows = []
    for s in rd.datasets.stats:
        rows.append({
            "split": s.split, "n_tabular": s.n_tabular, "n_sequence": s.n_sequence,
            "time_min_utc": _ns_to_iso(s.time_min), "time_max_utc": _ns_to_iso(s.time_max),
        })
    return pd.DataFrame(rows)


def predictive_metrics_table(rd: ReportData, split: str = "test") -> pd.DataFrame:
    if rd.predictive_metrics is None:
        return pd.DataFrame()
    pm = rd.predictive_metrics
    # keep the real metrics, drop the coverage counts
    sub = pm[(pm["split"] == split) & (~pm["metric_name"].isin(["n_predictions", "n_missing"]))]
    if len(sub) == 0:
        return pd.DataFrame()
    pivot = sub.pivot_table(index=["model_name", "horizon"], columns="metric_name", values="metric_value")
    return pivot.reset_index()


def confusion_table(rd: ReportData, split: str = "test") -> pd.DataFrame:
    if rd.confusion is None:
        return pd.DataFrame()
    c = rd.confusion[rd.confusion["split"] == split]
    return c.reset_index(drop=True)


def backtest_metrics_table(rd: ReportData, split: str = "test") -> pd.DataFrame:
    if rd.backtest_metrics is None:
        return pd.DataFrame()
    bt = rd.backtest_metrics[rd.backtest_metrics["split"] == split]
    if len(bt) == 0:
        return pd.DataFrame()
    pivot = bt.pivot_table(index="model_name", columns="metric_name", values="metric_value")
    return pivot.reset_index()


def build_tables(rd: ReportData) -> dict[str, pd.DataFrame]:
    """Build every table; return only the non-empty ones, keyed by file name."""
    candidates = {
        "row_counts": row_counts_table(rd),
        "quality_flags": quality_flags_table(rd),
        "feature_summary": feature_summary_table(rd),
        "label_distribution": label_distribution_table(rd),
        "split_time_ranges": split_time_ranges_table(rd),
        "predictive_metrics_test": predictive_metrics_table(rd, "test"),
        "confusion_matrices_test": confusion_table(rd, "test"),
        "backtest_metrics_test": backtest_metrics_table(rd, "test"),
    }
    out = {}
    for name, table in candidates.items():
        if table is not None and not table.empty:
            out[name] = table
    return out
