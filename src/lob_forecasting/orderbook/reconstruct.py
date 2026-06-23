"""Rebuild the order book from events and write the book tables.

One engine handles both cases. With snapshot data every timestamp is a full
snapshot, so we clear the book and rebuild it each group. With incremental
depth updates we keep applying changes, remove levels on size 0, and notice
gaps in the update ids. Either way: group events by timestamp, optionally clear
on a snapshot, apply the updates, emit one book row per group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..normalisation.event_table import EventTableIndex
from .book_state import BookState
from .book_table import (
    FLAG_CROSSED,
    FLAG_MISSING_LEVELS,
    FLAG_NO_BEST_ASK,
    FLAG_NO_BEST_BID,
    BookPartition,
    BookTableIndex,
    book_columns,
    compute_derived,
    enforce_book_schema,
    flag_counts,
)

# event types that change the book
_BOOK_EVENT_TYPES = frozenset({"snapshot", "depth_update", "quote"})


def _reconstruct_snapshot_fast(
    df: pd.DataFrame, top_k: int, venue: str | None, symbol: str | None
) -> pd.DataFrame | None:
    """Vectorised reconstruction for clean snapshot data.

    Returns None (so the caller falls back to the general engine) unless the
    events have the exact canonical shape the snapshot adapters produce: every
    timestamp is one full book of 2*K contiguous rows, K bids (best first) then
    K asks (best first), with unique timestamps. On real top-5 snapshot archives
    this turns a multi-minute Python group loop into a few seconds of numpy.
    """
    k = top_k
    n = len(df)
    if n == 0 or n % (2 * k) != 0:
        return None

    side = df["side"].astype("string").to_numpy()
    n_snap = n // (2 * k)
    side_blocks = side.reshape(n_snap, 2 * k)
    if not ((side_blocks[:, :k] == "bid").all() and (side_blocks[:, k:] == "ask").all()):
        return None

    ts = df["timestamp_exchange_ns"].to_numpy()
    ts_blocks = ts.reshape(n_snap, 2 * k)
    if not (ts_blocks == ts_blocks[:, :1]).all():
        return None  # a block spans more than one timestamp -> not canonical
    ts_snap = ts_blocks[:, 0]
    if np.unique(ts_snap).size != n_snap:
        return None  # duplicate timestamps -> let the engine merge them

    price = df["price"].to_numpy(dtype="float64").reshape(n_snap, 2 * k)
    qty = df["quantity"].to_numpy(dtype="float64").reshape(n_snap, 2 * k)
    event_id = df["event_id"].to_numpy().reshape(n_snap, 2 * k)[:, -1]
    bid_px, bid_qty = price[:, :k], qty[:, :k]
    ask_px, ask_qty = price[:, k:], qty[:, k:]

    best_bid, best_ask = bid_px[:, 0], ask_px[:, 0]
    best_bid_qty, best_ask_qty = bid_qty[:, 0], ask_qty[:, 0]
    have_bid = np.isfinite(best_bid)
    have_ask = np.isfinite(best_ask)
    both = have_bid & have_ask

    with np.errstate(invalid="ignore", divide="ignore"):
        mid = np.where(both, (best_ask + best_bid) / 2.0, np.nan)
        spread = np.where(both, best_ask - best_bid, np.nan)
        relative_spread = np.where(both & (mid != 0), spread / mid, np.nan)
        denom = best_ask_qty + best_bid_qty
        microprice = np.where(
            both & (denom > 0),
            (best_ask * best_bid_qty + best_bid * best_ask_qty) / denom,
            np.nan,
        )
    crossed = both & (best_bid > best_ask)
    missing = (~np.isfinite(bid_px).all(axis=1)) | (~np.isfinite(ask_px).all(axis=1))

    out: dict[str, object] = {"event_id": event_id, "timestamp_exchange_ns": ts_snap,
                              "venue": venue, "symbol": symbol}
    for i in range(k):
        out[f"bid_px_{i+1}"] = bid_px[:, i]
        out[f"bid_qty_{i+1}"] = bid_qty[:, i]
        out[f"ask_px_{i+1}"] = ask_px[:, i]
        out[f"ask_qty_{i+1}"] = ask_qty[:, i]
    out.update(mid=mid, spread=spread, relative_spread=relative_spread, microprice=microprice,
               book_is_crossed=crossed, book_has_missing_levels=missing,
               book_update_gap_detected=np.zeros(n_snap, dtype=bool))

    # assemble pipe-joined quality flags vectorially
    flag_cols = [
        (~have_bid, FLAG_NO_BEST_BID),
        (~have_ask, FLAG_NO_BEST_ASK),
        (missing, FLAG_MISSING_LEVELS),
        (crossed, FLAG_CROSSED),
    ]
    flags = np.array([""] * n_snap, dtype=object)
    for mask, name in flag_cols:
        add = np.where(mask, name, "")
        joined = np.where((flags != "") & (add != ""), flags + "|" + add, flags + add)
        flags = joined
    out["quality_flags"] = flags

    book_df = pd.DataFrame(out, columns=book_columns(k))
    return enforce_book_schema(book_df, k)


class OrderBookError(RuntimeError):
    """Reconstruction can't go ahead (e.g. mode doesn't match the data)."""


def _coerce_events(events: pd.DataFrame) -> pd.DataFrame:
    """Add any missing columns and sort by event_id."""
    df = events.copy()
    for col, default in (
        ("event_type", pd.NA),
        ("side", pd.NA),
        ("price", pd.NA),
        ("quantity", pd.NA),
        ("update_id", pd.NA),
        ("is_snapshot", False),
    ):
        if col not in df.columns:
            df[col] = default
    if "event_id" in df.columns:
        df = df.sort_values("event_id", kind="mergesort").reset_index(drop=True)
    return df


def reconstruct_book(
    events: pd.DataFrame,
    top_k: int,
    venue: str | None = None,
    symbol: str | None = None,
    mode: str = "auto",
) -> pd.DataFrame:
    """Replay one file's events into a top-K book table.

    mode is 'auto', 'snapshot' or 'replay'. It only sanity-checks the data; the
    engine below works the same either way.
    """
    df = _coerce_events(events)

    if venue is None and "venue" in df.columns and len(df):
        venue = str(df["venue"].iloc[0])
    if symbol is None and "symbol" in df.columns and len(df):
        symbol = str(df["symbol"].iloc[0])

    all_snapshot = bool(df["is_snapshot"].all()) if len(df) else True
    if mode == "snapshot" and not all_snapshot:
        raise OrderBookError("mode='snapshot' but events contain non-snapshot rows")

    # fast vectorised path for clean, canonically-shaped snapshot data; falls
    # back to the general engine below if the shape isn't exactly as expected
    if all_snapshot and mode in ("auto", "snapshot"):
        fast = _reconstruct_snapshot_fast(df, top_k, venue, symbol)
        if fast is not None:
            return fast

    state = BookState()
    last_update_id: int | None = None
    rows: list[dict] = []

    # timestamps are sorted, so each timestamp group is one book observation
    for ts, group in df.groupby("timestamp_exchange_ns", sort=True):
        is_snapshot_group = bool(group["is_snapshot"].any())
        if is_snapshot_group:
            state.clear()
            last_update_id = None  # snapshot re-syncs, so forget the last id

        group_gap = False
        book_relevant = False
        if "event_id" in group:
            last_event_id = int(group["event_id"].iloc[-1])
        else:
            last_event_id = len(rows)

        for ev in group.itertuples(index=False):
            etype = ev.event_type
            if etype is not None and etype not in _BOOK_EVENT_TYPES:
                continue  # e.g. a trade, doesn't change the book
            side = None if pd.isna(ev.side) else str(ev.side)
            price = None if pd.isna(ev.price) else float(ev.price)
            qty = None if pd.isna(ev.quantity) else float(ev.quantity)

            # look for gaps in update ids (only for incremental updates)
            if not is_snapshot_group and not pd.isna(ev.update_id):
                uid = int(ev.update_id)
                if last_update_id is not None and uid != last_update_id + 1:
                    group_gap = True
                last_update_id = uid

            if state.apply(side, price, qty):
                book_relevant = True

        if not book_relevant:
            continue  # nothing here changed the book

        bids, asks = state.top_k(top_k)
        row = compute_derived(bids, asks, top_k, gap_detected=group_gap)
        row["event_id"] = last_event_id
        row["timestamp_exchange_ns"] = int(ts)
        row["venue"] = venue
        row["symbol"] = symbol
        rows.append(row)

    book_df = pd.DataFrame(rows, columns=book_columns(top_k))
    return enforce_book_schema(book_df, top_k)


def _partition_path(books_root: Path, venue: str, symbol: str, date: str, k: int) -> Path:
    return books_root / f"venue={venue}" / f"symbol={symbol}" / f"date={date}" / f"book_top{k}.parquet"


def build_order_books(
    config: ExperimentConfig,
    events: EventTableIndex,
    project_root: str | Path | None = None,
) -> BookTableIndex:
    """Rebuild books for every event partition and write them out."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    books_root = root / config.data.processed_dir / "books"
    top_k = config.data.top_k
    mode = config.orderbook.mode

    from ..utils.progress import log

    stride = max(1, config.sampling.event_stride)

    partitions: list[BookPartition] = []
    for i, ep in enumerate(events.partitions, 1):
        log(f"[orderbook] {i}/{len(events.partitions)}: {ep.symbol} {ep.date} (reconstructing top-{top_k})")
        events_path = root / ep.file_path
        if not events_path.exists():
            raise FileNotFoundError(f"Event partition is missing: {events_path}")
        events_df = pd.read_parquet(events_path)

        book_df = reconstruct_book(events_df, top_k, venue=ep.venue, symbol=ep.symbol, mode=mode)
        if stride > 1:
            # event-time sampling: keep every Nth book snapshot (the book is
            # time-ordered), so features, backtest and MM share one event stream
            book_df = book_df.iloc[::stride].reset_index(drop=True)

        out_path = _partition_path(books_root, ep.venue, ep.symbol, ep.date, top_k)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        book_df.to_parquet(out_path, engine="pyarrow", index=False)

        counts = flag_counts(book_df)
        n_crossed = int(book_df["book_is_crossed"].sum())
        n_missing = int(book_df["book_has_missing_levels"].sum())
        n_gap = int(book_df["book_update_gap_detected"].sum())
        n_clean = int((book_df["quality_flags"] == "").sum())
        partitions.append(
            BookPartition(
                venue=ep.venue,
                symbol=ep.symbol,
                date=ep.date,
                top_k=top_k,
                file_path=out_path.relative_to(root).as_posix(),
                row_count=len(book_df),
                n_crossed=n_crossed,
                n_missing_levels=n_missing,
                n_update_gap=n_gap,
                n_clean=n_clean,
                flag_counts=counts,
            )
        )

    index = BookTableIndex(partitions=partitions)
    index.save(books_root / "books_index.yaml")
    return index
