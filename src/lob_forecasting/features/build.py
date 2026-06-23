"""Build the feature tables from the book tables.

Books are split per venue/symbol/date, but the feature table is one file per
venue/symbol. So for each symbol we glue its book files together, sort by
timestamp, and compute features over the whole sequence. A rolling window can
reach across a day boundary, but that's fine since it only ever looks backward.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..config import ExperimentConfig
from ..orderbook.book_table import BookTableIndex
from .compute import compute_features
from .feature_table import FeaturePartition, FeatureTableIndex, feature_value_columns, flag_counts


def build_features(
    config: ExperimentConfig,
    books: BookTableIndex,
    project_root: str | Path | None = None,
) -> FeatureTableIndex:
    """Compute features for every venue/symbol and write the parquet files."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    features_root = root / config.data.processed_dir.parent / "features"

    # group the book partitions by (venue, symbol)
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for bp in books.partitions:
        grouped[(bp.venue, bp.symbol)].append(bp)

    # in monthly mode each day is an independent calendar regime: compute features
    # per day so no rolling window crosses a monthly day boundary
    per_day = config.data.monthly_snapshot.enabled or config.features.include_regime_features
    date_index = {d.isoformat(): i for i, d in enumerate(config.data.monthly_dates)}

    partitions: list[FeaturePartition] = []
    for (venue, symbol), parts in grouped.items():
        parts_sorted = sorted(parts, key=lambda p: p.date)
        frames = []
        for bp in parts_sorted:
            path = root / bp.file_path
            if not path.exists():
                raise FileNotFoundError(f"Book partition is missing: {path}")
            frames.append(pd.read_parquet(path))

        if per_day:
            day_feats = [compute_features(f, config) for f in frames]
            feature_df = pd.concat(day_feats, ignore_index=True)
            feature_df = feature_df.sort_values(
                "timestamp_exchange_ns", kind="mergesort"
            ).reset_index(drop=True)
            if "monthly_date" in feature_df.columns and date_index:
                feature_df["month_index"] = (
                    feature_df["monthly_date"].map(date_index).fillna(0).astype("int64")
                )
        else:
            book_df = pd.concat(frames, ignore_index=True)
            feature_df = compute_features(book_df, config)

        out_path = features_root / f"venue={venue}" / f"symbol={symbol}" / "features.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_parquet(out_path, engine="pyarrow", index=False)

        dates = [bp.date for bp in parts_sorted]
        partitions.append(
            FeaturePartition(
                venue=venue,
                symbol=symbol,
                date_range=(dates[0], dates[-1]) if dates else None,
                file_path=out_path.relative_to(root).as_posix(),
                row_count=len(feature_df),
                n_dates=len(parts_sorted),
                feature_columns=feature_value_columns(config),
                flag_counts=flag_counts(feature_df),
            )
        )

    index = FeatureTableIndex(partitions=partitions)
    index.save(features_root / "features_index.yaml")
    return index
