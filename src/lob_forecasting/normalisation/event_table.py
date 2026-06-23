"""The normalised event table: column list, dtypes, and the partition index.

This is the stable format that hides the raw, venue-specific data from the rest
of the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field

# event types and side values we recognise
CANONICAL_EVENT_TYPES: frozenset[str] = frozenset({"trade", "depth_update", "snapshot", "quote"})
CANONICAL_SIDES: frozenset[str] = frozenset({"bid", "ask", "buy", "sell"})

# the columns of the event table, in order
EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "timestamp_exchange_ns",
    "timestamp_local_ns",
    "venue",
    "symbol",
    "event_type",
    "side",
    "price",
    "quantity",
    "update_id",
    "trade_id",
    "is_snapshot",
    "raw_file_id",
    "quality_flags",
)

# pandas dtype for each column; the "Int64"/"string" ones are nullable so they
# can hold real nulls and still round-trip through parquet
EVENT_DTYPES: dict[str, str] = {
    "event_id": "int64",
    "timestamp_exchange_ns": "int64",
    "timestamp_local_ns": "Int64",
    "venue": "string",
    "symbol": "string",
    "event_type": "string",
    "side": "string",
    "price": "float64",
    "quantity": "float64",
    "update_id": "Int64",
    "trade_id": "string",
    "is_snapshot": "bool",
    "raw_file_id": "string",
    "quality_flags": "string",
}


def enforce_event_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder/retype df so it matches the event-table schema exactly."""
    out = df.copy()
    for col in EVENT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[list(EVENT_COLUMNS)]
    # quality_flags is never null; empty string means the row is clean
    out["quality_flags"] = out["quality_flags"].fillna("").astype("string")
    for col, dtype in EVENT_DTYPES.items():
        if col == "quality_flags":
            continue
        out[col] = out[col].astype(dtype)
    return out


def flag_counts(df: pd.DataFrame) -> dict[str, int]:
    """How many rows carry each quality flag."""
    counts: dict[str, int] = {}
    for raw in df["quality_flags"].fillna("").astype(str):
        if not raw:
            continue
        for flag in raw.split("|"):
            if flag:
                counts[flag] = counts.get(flag, 0) + 1
    return counts


class EventPartition(BaseModel):
    """One events.parquet file (one venue/symbol/date)."""

    venue: str
    symbol: str
    date: str
    file_path: str  # relative to project root, forward slashes
    row_count: int
    n_dropped_unparseable: int = 0
    n_unknown_event_type: int = 0
    flag_counts: dict[str, int] = Field(default_factory=dict)


class EventTableIndex(BaseModel):
    """List of all the event partitions a run produced."""

    partitions: list[EventPartition] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.partitions)

    def total_rows(self) -> int:
        return sum(p.row_count for p in self.partitions)

    def paths(self) -> list[str]:
        return [p.file_path for p in self.partitions]

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "partitions": [p.model_dump(mode="json") for p in self.partitions]
        }
        with out.open("w", encoding="utf-8") as fh:
            yaml.dump(payload, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "EventTableIndex":
        p = Path(path)
        if not p.exists():
            return cls(partitions=[])
        with p.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        parts = raw.get("partitions") or []
        return cls(partitions=[EventPartition.model_validate(x) for x in parts])
