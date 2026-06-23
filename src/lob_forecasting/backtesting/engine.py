"""The taker backtest engine.

The policy at each decision event: buy if the predicted return is above the
threshold, sell if below minus the threshold, otherwise hold. The trade
executes a few events later, crossing the spread (buy pays the ask, sell hits
the bid), and is marked at t+h using the mid. Fees always come off, and a
position limit caps how much we can hold at once.

The book is indexed the same way features/labels were built (per symbol, sorted
by timestamp), so t+latency and t+horizon are just positional offsets, and the
mark price lines up with the label's forward mid.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_SHARPE_BUCKETS = 20

# columns of one trade row
TRADE_COLUMNS = (
    "run_id", "model_name", "split", "venue", "symbol", "horizon",
    "decision_event_id", "decision_timestamp_exchange_ns", "side",
    "exec_event_id", "exec_price", "mark_price", "trade_size",
    "gross_pnl", "fees", "net_pnl", "threshold",
)


@dataclass
class BookSeq:
    """Best-level book arrays for one symbol, indexed by event position."""

    ts_to_pos: dict[int, int]
    event_id: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    mid: np.ndarray

    @property
    def n(self) -> int:
        return len(self.mid)


def build_book_sequences(books, project_root: str | Path | None = None) -> dict[tuple[str, str], BookSeq]:
    """Read the book files into a BookSeq per (venue, symbol)."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for bp in books.partitions:
        grouped[(bp.venue, bp.symbol)].append(bp)

    seqs: dict[tuple[str, str], BookSeq] = {}
    for (venue, symbol), parts in grouped.items():
        parts = sorted(parts, key=lambda p: p.date)
        frames = []
        for bp in parts:
            cols = ["event_id", "timestamp_exchange_ns", "bid_px_1", "ask_px_1", "mid"]
            frames.append(pd.read_parquet(root / bp.file_path)[cols])
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("timestamp_exchange_ns", kind="mergesort").reset_index(drop=True)
        ts = df["timestamp_exchange_ns"].to_numpy()
        seqs[(venue, symbol)] = BookSeq(
            ts_to_pos={int(t): i for i, t in enumerate(ts)},
            event_id=df["event_id"].to_numpy(),
            bid=df["bid_px_1"].to_numpy(dtype="float64"),
            ask=df["ask_px_1"].to_numpy(dtype="float64"),
            mid=df["mid"].to_numpy(dtype="float64"),
        )
    return seqs


def simulate(
    pred_df: pd.DataFrame,
    book: BookSeq,
    *,
    threshold: float,
    horizon: int,
    latency_events: int,
    fee_bps: float,
    trade_size: float,
    max_position: float,
) -> tuple[list[dict], int, int]:
    """Run the policy over one symbol at one threshold.

    Returns (trades, n_excluded, n_skipped_limit). Each trade dict also has an
    internal decision_pos used later for the Sharpe buckets.
    """
    fee_rate = fee_bps / 1e4
    n = book.n

    rows = pred_df[["venue", "symbol", "event_id", "timestamp_exchange_ns", "pred_return"]].copy()
    rows["posidx"] = rows["timestamp_exchange_ns"].map(lambda t: book.ts_to_pos.get(int(t), -1))
    rows = rows.sort_values("posidx", kind="mergesort")

    inventory = 0.0
    open_heap: list[tuple[int, float]] = []  # (close_pos, signed_size)
    trades: list[dict] = []
    n_excluded = 0
    n_skipped_limit = 0

    for r in rows.itertuples(index=False):
        pos = int(r.posidx)
        rhat = r.pred_return
        if pos < 0 or pd.isna(rhat):
            continue
        if rhat > threshold:
            side = "buy"
            signed = trade_size
        elif rhat < -threshold:
            side = "sell"
            signed = -trade_size
        else:
            continue  # hold

        exec_pos = pos + latency_events
        mark_pos = pos + horizon
        if exec_pos >= n or mark_pos >= n:
            n_excluded += 1  # no future price to execute or mark against
            continue

        # close out any positions that finished at or before now
        while open_heap and open_heap[0][0] <= pos:
            _, sz = heapq.heappop(open_heap)
            inventory -= sz

        if abs(inventory + signed) > max_position + 1e-12:
            n_skipped_limit += 1  # would breach the position limit
            continue

        exec_price = book.ask[exec_pos] if side == "buy" else book.bid[exec_pos]
        mark_price = book.mid[mark_pos]
        if not (np.isfinite(exec_price) and np.isfinite(mark_price)):
            n_excluded += 1
            continue

        if side == "buy":
            gross = (mark_price - exec_price) * trade_size
        else:
            gross = (exec_price - mark_price) * trade_size
        fees = fee_rate * exec_price * trade_size
        trades.append({
            "venue": r.venue, "symbol": r.symbol, "horizon": horizon,
            "decision_event_id": int(r.event_id),
            "decision_timestamp_exchange_ns": int(r.timestamp_exchange_ns),
            "side": side,
            "exec_event_id": int(book.event_id[exec_pos]),
            "exec_price": float(exec_price), "mark_price": float(mark_price),
            "trade_size": float(trade_size),
            "gross_pnl": float(gross), "fees": float(fees), "net_pnl": float(gross - fees),
            "threshold": float(threshold), "decision_pos": pos,
        })
        inventory += signed
        heapq.heappush(open_heap, (mark_pos, signed))

    return trades, n_excluded, n_skipped_limit


def simulate_split(
    split_df: pd.DataFrame, seqs: dict[tuple[str, str], BookSeq], threshold: float, cfg
) -> tuple[list[dict], int, int]:
    """Run simulate() for each symbol in a split and add up the results."""
    trades: list[dict] = []
    n_exc = 0
    n_lim = 0
    for (venue, symbol), g in split_df.groupby(["venue", "symbol"]):
        seq = seqs.get((venue, symbol))
        if seq is None:
            continue
        sym_trades, exc, lim = simulate(
            g, seq, threshold=threshold, horizon=cfg.horizon,
            latency_events=cfg.latency_events, fee_bps=cfg.fee_bps,
            trade_size=cfg.trade_size, max_position=cfg.max_position,
        )
        trades.extend(sym_trades)
        n_exc += exc
        n_lim += lim
    return trades, n_exc, n_lim


def _max_drawdown(net: np.ndarray) -> float:
    if len(net) == 0:
        return 0.0
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum))


def _sharpe_like(trades: list[dict], pos_min: int, pos_max: int, n_buckets: int = N_SHARPE_BUCKETS) -> float:
    """A simple Sharpe: bucket the events by time, sum net PnL per bucket, mean/std.

    Empty (no-trade) buckets count as 0, so flat periods drag the ratio down.
    """
    if len(trades) < 2 or pos_max <= pos_min:
        return float("nan")
    edges = np.linspace(pos_min, pos_max + 1, n_buckets + 1)
    bucket = np.zeros(n_buckets)
    for t in trades:
        b = int(np.clip(np.digitize(t["decision_pos"], edges[1:-1]), 0, n_buckets - 1))
        bucket[b] += t["net_pnl"]
    sd = bucket.std()
    return float(bucket.mean() / sd) if sd > 0 else float("nan")


def compute_metrics(
    trades: list[dict], pos_min: int, pos_max: int, n_excluded: int, n_skipped_limit: int, threshold: float
) -> dict[str, float]:
    """The backtest numbers for one model/split/symbol/horizon."""
    n = len(trades)
    net = np.array([t["net_pnl"] for t in trades], dtype="float64")
    gross = np.array([t["gross_pnl"] for t in trades], dtype="float64")
    notional = np.array([t["exec_price"] * t["trade_size"] for t in trades], dtype="float64")
    return {
        "n_trades": float(n),
        "turnover": float(notional.sum()),
        "gross_pnl": float(gross.sum()),
        "net_pnl": float(net.sum()),
        "mean_pnl_per_trade": float(net.mean()) if n else float("nan"),
        "hit_rate": float((net > 0).mean()) if n else float("nan"),
        "max_drawdown": _max_drawdown(net),
        "sharpe_like": _sharpe_like(trades, pos_min, pos_max),
        "n_excluded_no_mark": float(n_excluded),
        "n_skipped_position_limit": float(n_skipped_limit),
        "selected_threshold": float(threshold),
    }
