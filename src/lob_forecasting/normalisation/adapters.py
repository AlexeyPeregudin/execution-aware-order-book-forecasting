"""Venue adapters. Each one parses a raw file into rough event rows.

The adapter only parses. The shared work (sorting, event ids, flags, dedup)
happens in normalise.py, so adapters stay small. Right now we have one adapter
for the top-K snapshot CSV that the synthetic source produces.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingestion.manifest import ManifestEntry


class NormalisationError(RuntimeError):
    """A raw file couldn't be parsed at all."""


@dataclass
class ParseResult:
    """What an adapter returns for one file."""

    events: pd.DataFrame  # long-format events, not yet finalised
    n_dropped_unparseable: int = 0


class VenueAdapter(ABC):
    """Parses one raw file into rough event rows."""

    name: str

    @abstractmethod
    def parse(self, raw_path: Path, entry: ManifestEntry) -> ParseResult:
        """Parse raw_path into events, tagged with the manifest entry's info."""


_BID_PX_RE = re.compile(r"^bid_px_(\d+)$")


class SnapshotCsvAdapter(VenueAdapter):
    """Turns a top-K snapshot CSV into per-level snapshot events.

    Each CSV row is one book observation:
        timestamp_ms, bid_px_1, bid_qty_1, ask_px_1, ask_qty_1, ... (K levels)

    It becomes 2*K event rows (K bids then K asks) that all share the same
    timestamp. The order-book step later groups them back by timestamp.
    """

    name = "snapshot_csv"

    def parse(self, raw_path: Path, entry: ManifestEntry) -> ParseResult:
        df = pd.read_csv(raw_path)

        if "timestamp_ms" not in df.columns:
            raise NormalisationError(f"{raw_path}: missing required 'timestamp_ms' column")

        # figure out how many levels there are from the bid_px_* columns
        levels = []
        for c in df.columns:
            m = _BID_PX_RE.match(c)
            if m:
                levels.append(int(m.group(1)))
        levels.sort()
        if not levels:
            raise NormalisationError(f"{raw_path}: no 'bid_px_k' columns found; not a snapshot CSV")

        # drop rows with an unparseable timestamp (and count them)
        ts_ms = pd.to_numeric(df["timestamp_ms"], errors="coerce")
        bad_ts = ts_ms.isna()
        n_dropped = int(bad_ts.sum())
        if n_dropped:
            df = df.loc[~bad_ts].reset_index(drop=True)
            ts_ms = ts_ms.loc[~bad_ts].reset_index(drop=True)

        n_rows = len(df)
        if n_rows == 0:
            return ParseResult(events=_empty_provisional(), n_dropped_unparseable=n_dropped)

        ts_ns = ts_ms.to_numpy(dtype="int64") * 1_000_000
        k = len(levels)

        bid_px_cols = [f"bid_px_{i}" for i in levels]
        bid_qty_cols = [f"bid_qty_{i}" for i in levels]
        ask_px_cols = [f"ask_px_{i}" for i in levels]
        ask_qty_cols = [f"ask_qty_{i}" for i in levels]

        # snap_row tells us which original row each level came from, so we can
        # keep one snapshot's levels together later
        snap_row = np.repeat(np.arange(n_rows), k)
        ts_rep = np.repeat(ts_ns, k)

        bid = pd.DataFrame(
            {
                "snap_row": snap_row,
                "timestamp_exchange_ns": ts_rep,
                "side": "bid",
                "price": df[bid_px_cols].to_numpy(dtype="float64").reshape(-1),
                "quantity": df[bid_qty_cols].to_numpy(dtype="float64").reshape(-1),
            }
        )
        ask = pd.DataFrame(
            {
                "snap_row": snap_row,
                "timestamp_exchange_ns": ts_rep,
                "side": "ask",
                "price": df[ask_px_cols].to_numpy(dtype="float64").reshape(-1),
                "quantity": df[ask_qty_cols].to_numpy(dtype="float64").reshape(-1),
            }
        )

        events = pd.concat([bid, ask], ignore_index=True)
        # stable sort on snap_row keeps each snapshot's bids+asks next to each other
        events = events.sort_values("snap_row", kind="mergesort").reset_index(drop=True)
        events = events.drop(columns="snap_row")

        events["event_type"] = "snapshot"
        events["is_snapshot"] = True
        events["timestamp_local_ns"] = pd.NA
        events["update_id"] = pd.NA
        events["trade_id"] = pd.NA
        events["venue"] = entry.venue
        events["symbol"] = entry.symbol
        events["raw_file_id"] = entry.source_id
        events["quality_flags"] = ""

        return ParseResult(events=events, n_dropped_unparseable=n_dropped)


_TARDIS_ASK_PX_RE = re.compile(r"^asks\[(\d+)\]\.price$")


class TardisBookSnapshotAdapter(VenueAdapter):
    """Parses a Tardis `book_snapshot_N` CSV (optionally gzipped) directly.

    Tardis columns are 0-indexed and microsecond-stamped:

        exchange, symbol, timestamp(us), local_timestamp(us),
        asks[0].price, asks[0].amount, bids[0].price, bids[0].amount, ...

    Each row is one book observation; it becomes 2*K snapshot event rows (K bids
    then K asks) sharing the row's timestamp, exactly like the snapshot-CSV
    adapter, so nothing downstream needs to change.
    """

    name = "tardis_book_snapshot"

    def parse(self, raw_path: Path, entry: ManifestEntry) -> ParseResult:
        df = pd.read_csv(raw_path)  # pandas infers gzip from the .gz extension

        if "timestamp" not in df.columns:
            raise NormalisationError(f"{raw_path}: missing Tardis 'timestamp' column")

        levels = []
        for c in df.columns:
            m = _TARDIS_ASK_PX_RE.match(c)
            if m:
                levels.append(int(m.group(1)))
        levels = sorted(levels)
        if not levels:
            raise NormalisationError(f"{raw_path}: no 'asks[k].price' columns; not a Tardis snapshot")

        # Tardis timestamps are microseconds; drop unparseable rows
        ts_us = pd.to_numeric(df["timestamp"], errors="coerce")
        bad_ts = ts_us.isna()
        n_dropped = int(bad_ts.sum())
        if n_dropped:
            df = df.loc[~bad_ts].reset_index(drop=True)
            ts_us = ts_us.loc[~bad_ts].reset_index(drop=True)

        n_rows = len(df)
        if n_rows == 0:
            return ParseResult(events=_empty_provisional(), n_dropped_unparseable=n_dropped)

        ts_ns = ts_us.to_numpy(dtype="int64") * 1_000  # us -> ns
        k = len(levels)

        bid_px_cols = [f"bids[{i}].price" for i in levels]
        bid_qty_cols = [f"bids[{i}].amount" for i in levels]
        ask_px_cols = [f"asks[{i}].price" for i in levels]
        ask_qty_cols = [f"asks[{i}].amount" for i in levels]

        snap_row = np.repeat(np.arange(n_rows), k)
        ts_rep = np.repeat(ts_ns, k)

        bid = pd.DataFrame({
            "snap_row": snap_row, "timestamp_exchange_ns": ts_rep, "side": "bid",
            "price": df[bid_px_cols].to_numpy(dtype="float64").reshape(-1),
            "quantity": df[bid_qty_cols].to_numpy(dtype="float64").reshape(-1),
        })
        ask = pd.DataFrame({
            "snap_row": snap_row, "timestamp_exchange_ns": ts_rep, "side": "ask",
            "price": df[ask_px_cols].to_numpy(dtype="float64").reshape(-1),
            "quantity": df[ask_qty_cols].to_numpy(dtype="float64").reshape(-1),
        })

        events = pd.concat([bid, ask], ignore_index=True)
        events = events.sort_values("snap_row", kind="mergesort").reset_index(drop=True)
        events = events.drop(columns="snap_row")

        events["event_type"] = "snapshot"
        events["is_snapshot"] = True
        events["timestamp_local_ns"] = pd.NA  # not carried downstream
        events["update_id"] = pd.NA
        events["trade_id"] = pd.NA
        events["venue"] = entry.venue
        events["symbol"] = entry.symbol
        events["raw_file_id"] = entry.source_id
        events["quality_flags"] = ""

        return ParseResult(events=events, n_dropped_unparseable=n_dropped)


def _empty_provisional() -> pd.DataFrame:
    cols = [
        "timestamp_exchange_ns", "timestamp_local_ns", "venue", "symbol",
        "event_type", "side", "price", "quantity", "update_id", "trade_id",
        "is_snapshot", "raw_file_id", "quality_flags",
    ]
    return pd.DataFrame({c: [] for c in cols})


# our binance sample is snapshot CSV; a real depth-diff adapter would slot in here
_ADAPTER_REGISTRY: dict[str, type[VenueAdapter]] = {
    "snapshot_csv": SnapshotCsvAdapter,
    "synthetic": SnapshotCsvAdapter,
    "binance": SnapshotCsvAdapter,
    "tardis_book_snapshot": TardisBookSnapshotAdapter,
}


def build_venue_adapter(venue: str, override: str | None = None) -> VenueAdapter:
    """Pick the adapter for a venue (or by an explicit override name)."""
    key = (override or venue).lower()
    cls = _ADAPTER_REGISTRY.get(key)
    if cls is None:
        raise NormalisationError(
            f"No venue adapter for {key!r}. Known: {sorted(_ADAPTER_REGISTRY)}"
        )
    return cls()
