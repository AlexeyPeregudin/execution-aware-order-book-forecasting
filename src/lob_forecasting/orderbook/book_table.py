"""The top-K book table: columns, dtypes, derived fields, and the index."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field

# quality flags the reconstruction can raise
FLAG_CROSSED = "crossed_book"
FLAG_MISSING_LEVELS = "missing_levels"
FLAG_UPDATE_GAP = "update_gap"
FLAG_NO_BEST_BID = "no_best_bid"
FLAG_NO_BEST_ASK = "no_best_ask"

# the scalar columns after the per-level price/size block
_DERIVED_COLUMNS: tuple[str, ...] = (
    "mid",
    "spread",
    "relative_spread",
    "microprice",
    "book_is_crossed",
    "book_has_missing_levels",
    "book_update_gap_detected",
    "quality_flags",
)


def level_columns(k: int) -> list[str]:
    """The 4*K per-level columns: all bid prices, bid sizes, ask prices, ask sizes."""
    cols: list[str] = []
    cols += [f"bid_px_{i}" for i in range(1, k + 1)]
    cols += [f"bid_qty_{i}" for i in range(1, k + 1)]
    cols += [f"ask_px_{i}" for i in range(1, k + 1)]
    cols += [f"ask_qty_{i}" for i in range(1, k + 1)]
    return cols


def book_columns(k: int) -> list[str]:
    """Full column list for a top-K book table, in order."""
    return [
        "event_id",
        "timestamp_exchange_ns",
        "venue",
        "symbol",
        *level_columns(k),
        *_DERIVED_COLUMNS,
    ]


def book_dtypes(k: int) -> dict[str, str]:
    dtypes: dict[str, str] = {
        "event_id": "int64",
        "timestamp_exchange_ns": "int64",
        "venue": "string",
        "symbol": "string",
        "mid": "float64",
        "spread": "float64",
        "relative_spread": "float64",
        "microprice": "float64",
        "book_is_crossed": "bool",
        "book_has_missing_levels": "bool",
        "book_update_gap_detected": "bool",
    }
    for col in level_columns(k):
        dtypes[col] = "float64"
    return dtypes


def enforce_book_schema(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Reorder/retype df to match the top-K book schema."""
    out = df.copy()
    cols = book_columns(k)
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[cols]
    out["quality_flags"] = out["quality_flags"].fillna("").astype("string")
    for col, dtype in book_dtypes(k).items():
        out[col] = out[col].astype(dtype)
    return out


def compute_derived(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    k: int,
    gap_detected: bool = False,
) -> dict[str, Any]:
    """Build one book row (the level columns plus mid/spread/microprice/flags).

    mid, spread, relative_spread and microprice are null if there's no best bid
    or ask. Crossed books still get priced, they just also get flagged.
    """
    row: dict[str, Any] = {}
    for i in range(k):
        if i < len(bids):
            row[f"bid_px_{i + 1}"], row[f"bid_qty_{i + 1}"] = bids[i]
        else:
            row[f"bid_px_{i + 1}"] = math.nan
            row[f"bid_qty_{i + 1}"] = math.nan
        if i < len(asks):
            row[f"ask_px_{i + 1}"], row[f"ask_qty_{i + 1}"] = asks[i]
        else:
            row[f"ask_px_{i + 1}"] = math.nan
            row[f"ask_qty_{i + 1}"] = math.nan

    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    best_bid_qty = bids[0][1] if bids else None
    best_ask_qty = asks[0][1] if asks else None

    flags: list[str] = []
    if best_bid is None:
        flags.append(FLAG_NO_BEST_BID)
    if best_ask is None:
        flags.append(FLAG_NO_BEST_ASK)

    missing = len(bids) < k or len(asks) < k
    if missing:
        flags.append(FLAG_MISSING_LEVELS)

    crossed = False
    if best_bid is not None and best_ask is not None:
        mid = (best_ask + best_bid) / 2.0
        spread = best_ask - best_bid
        relative_spread = spread / mid if mid != 0 else math.nan
        denom = best_ask_qty + best_bid_qty
        # microprice: weight each side's price by the other side's size
        if denom > 0:
            microprice = (best_ask * best_bid_qty + best_bid * best_ask_qty) / denom
        else:
            microprice = math.nan
        if best_bid > best_ask:
            crossed = True
            flags.append(FLAG_CROSSED)
    else:
        mid = spread = relative_spread = microprice = math.nan

    if gap_detected:
        flags.append(FLAG_UPDATE_GAP)

    row.update(
        {
            "mid": mid,
            "spread": spread,
            "relative_spread": relative_spread,
            "microprice": microprice,
            "book_is_crossed": crossed,
            "book_has_missing_levels": missing,
            "book_update_gap_detected": gap_detected,
            "quality_flags": "|".join(flags),
        }
    )
    return row


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


class BookPartition(BaseModel):
    """One book_top{K}.parquet file plus a few quality counts."""

    venue: str
    symbol: str
    date: str
    top_k: int
    file_path: str
    row_count: int
    n_crossed: int = 0
    n_missing_levels: int = 0
    n_update_gap: int = 0
    n_clean: int = 0
    flag_counts: dict[str, int] = Field(default_factory=dict)


class BookTableIndex(BaseModel):
    """All book partitions from a run (doubles as the quality summary)."""

    partitions: list[BookPartition] = Field(default_factory=list)

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
    def load(cls, path: str | Path) -> "BookTableIndex":
        p = Path(path)
        if not p.exists():
            return cls(partitions=[])
        with p.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        parts = raw.get("partitions") or []
        return cls(partitions=[BookPartition.model_validate(x) for x in parts])
