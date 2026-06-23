"""Backtesting: a simple taker strategy with fees and latency."""

from .engine import (
    TRADE_COLUMNS,
    BookSeq,
    build_book_sequences,
    compute_metrics,
    simulate,
    simulate_split,
)
from .market_making.run import MarketMakingResult, run_market_making
from .run import (
    METRIC_COLUMNS,
    BacktestError,
    BacktestResult,
    run_backtest,
)

__all__ = [
    "run_backtest",
    "BacktestResult",
    "BacktestError",
    "METRIC_COLUMNS",
    # Market making
    "run_market_making",
    "MarketMakingResult",
    # Engine
    "simulate",
    "simulate_split",
    "compute_metrics",
    "build_book_sequences",
    "BookSeq",
    "TRADE_COLUMNS",
]
