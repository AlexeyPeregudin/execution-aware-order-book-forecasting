"""Finalise parsed events and write the event tables.

finalise_events does the venue-independent cleanup: clamp bad prices/sizes,
handle unknown event types, flag duplicates, sort, and number the events.
normalise_events runs the adapters over the manifest and writes the parquet
files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..ingestion.manifest import RawDataManifest
from .adapters import ParseResult, build_venue_adapter
from .event_table import (
    CANONICAL_EVENT_TYPES,
    EventPartition,
    EventTableIndex,
    enforce_event_schema,
    flag_counts,
)

# two rows are "the same message" if these match
_DEDUP_KEY = ["venue", "symbol", "event_type", "update_id", "trade_id"]


def _add_flags(flags: list[str], mask: np.ndarray, name: str) -> None:
    """Add a flag name to the rows picked out by mask (flags is a list of strings)."""
    for i in np.flatnonzero(np.asarray(mask)):
        if flags[i]:
            flags[i] = flags[i] + "|" + name
        else:
            flags[i] = name


def finalise_events(
    provisional: pd.DataFrame,
    unknown_event_type_policy: str = "flag",
) -> pd.DataFrame:
    """Clean up rough adapter output into a proper event table.

    Bad prices/quantities get nulled and flagged (we don't drop them silently),
    unknown event types are flagged or rejected, duplicate messages are flagged,
    then everything is sorted and given a monotone event_id.
    """
    df = provisional.reset_index(drop=True).copy()

    # make sure the columns we touch exist
    for col in ("price", "quantity", "update_id", "trade_id", "side", "quality_flags"):
        if col not in df.columns:
            df[col] = pd.NA
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    n = len(df)
    flags: list[str] = []
    for x in df["quality_flags"].tolist():
        flags.append("" if pd.isna(x) else str(x))

    # 1. price must be > 0 and quantity >= 0; null + flag the bad ones
    price = df["price"].to_numpy(dtype="float64")
    qty = df["quantity"].to_numpy(dtype="float64")
    bad_price = (~np.isnan(price)) & (price <= 0)
    bad_qty = (~np.isnan(qty)) & (qty < 0)
    _add_flags(flags, bad_price, "nonpositive_price")
    _add_flags(flags, bad_qty, "negative_quantity")
    df.loc[bad_price, "price"] = np.nan
    df.loc[bad_qty, "quantity"] = np.nan

    # 2. unknown event types
    is_unknown = ~df["event_type"].astype("string").isin(CANONICAL_EVENT_TYPES)
    is_unknown_np = is_unknown.to_numpy()
    if unknown_event_type_policy == "reject":
        keep = ~is_unknown_np
        df = df.loc[keep].reset_index(drop=True)
        flags = [f for f, k in zip(flags, keep) if k]
        n = len(df)
    else:
        _add_flags(flags, is_unknown_np, "unknown_event_type")

    # 3. duplicate messages. Only look at rows that actually have an id,
    # otherwise all the id-less rows would look like duplicates of each other.
    has_id = df["update_id"].notna() | df["trade_id"].notna()
    dup = pd.Series(False, index=df.index)
    if has_id.any():
        sub = df.loc[has_id, _DEDUP_KEY]
        dup.loc[has_id] = sub.duplicated(keep=False).to_numpy()
    _add_flags(flags, dup.to_numpy(), "duplicate_update_id")

    df["quality_flags"] = pd.Series(flags, index=df.index, dtype="string")

    # 4. sort and number. parse_order is just the original order, used as a
    # tie-breaker so the sort is deterministic.
    df["_parse_order"] = np.arange(n)
    df = df.sort_values(
        ["timestamp_exchange_ns", "update_id", "_parse_order"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    df = df.drop(columns="_parse_order")
    df["event_id"] = np.arange(len(df), dtype="int64")

    return enforce_event_schema(df)


def _partition_path(events_root: Path, venue: str, symbol: str, date: str) -> Path:
    return events_root / f"venue={venue}" / f"symbol={symbol}" / f"date={date}" / "events.parquet"


def normalise_events(
    config: ExperimentConfig,
    manifest: RawDataManifest,
    project_root: str | Path | None = None,
) -> EventTableIndex:
    """Parse every raw file in the manifest and write the event tables."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    events_root = root / config.data.processed_dir / "events"
    policy = config.normalisation.unknown_event_type_policy

    from ..utils.progress import log

    partitions: list[EventPartition] = []
    for i, entry in enumerate(manifest.entries, 1):
        log(f"[normalise] {i}/{len(manifest.entries)}: {entry.symbol} {entry.date}")
        adapter = build_venue_adapter(entry.venue, config.normalisation.venue_adapter)
        raw_path = root / entry.file_path
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file referenced by manifest is missing: {raw_path}")

        result: ParseResult = adapter.parse(raw_path, entry)
        if len(result.events):
            known = result.events["event_type"].astype("string").isin(CANONICAL_EVENT_TYPES)
            n_unknown = int((~known).sum())
        else:
            n_unknown = 0

        events = finalise_events(result.events, unknown_event_type_policy=policy)

        date_str = entry.date.isoformat()
        out_path = _partition_path(events_root, entry.venue, entry.symbol, date_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(out_path, engine="pyarrow", index=False)

        partitions.append(
            EventPartition(
                venue=entry.venue,
                symbol=entry.symbol,
                date=date_str,
                file_path=out_path.relative_to(root).as_posix(),
                row_count=len(events),
                n_dropped_unparseable=result.n_dropped_unparseable,
                n_unknown_event_type=n_unknown,
                flag_counts=flag_counts(events),
            )
        )

    index = EventTableIndex(partitions=partitions)
    index.save(events_root / "events_index.yaml")
    return index
