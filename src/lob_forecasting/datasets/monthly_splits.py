"""Monthly snapshot splits for the distribution-shift benchmark.

Monthly days are first-of-month BTCUSDT snapshots and are not consecutive, so
they are treated as independent calendar regimes. A fold assigns whole monthly
days to train / validation / test; nothing crosses a day boundary (the feature
and label stages compute per day, so windows and horizons stay inside a day).

Two modes:
  - fixed_monthly_snapshot:     one fold (explicit date lists, or derived counts)
  - expanding_monthly_snapshot: one fold per feasible step, training window grows
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from ..config import ExperimentConfig
from .splits import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALIDATION

SPLIT_UNUSED = "unused"


@dataclass(frozen=True)
class MonthlyFold:
    """One train/validation/test partition over whole monthly days."""

    fold_id: int
    train_dates: tuple[date, ...]
    validation_dates: tuple[date, ...]
    test_dates: tuple[date, ...]

    @property
    def name(self) -> str:
        return f"fold_{self.fold_id}"

    def split_of(self, d: date) -> str:
        if d in self.train_dates:
            return SPLIT_TRAIN
        if d in self.validation_dates:
            return SPLIT_VALIDATION
        if d in self.test_dates:
            return SPLIT_TEST
        return SPLIT_UNUSED

    def all_dates(self) -> tuple[date, ...]:
        return (*self.train_dates, *self.validation_dates, *self.test_dates)

    def as_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train_dates": [d.isoformat() for d in self.train_dates],
            "validation_dates": [d.isoformat() for d in self.validation_dates],
            "test_dates": [d.isoformat() for d in self.test_dates],
        }


def monthly_dates(config: ExperimentConfig) -> list[date]:
    """The configured first-of-month snapshot dates, sorted, deduplicated."""
    return sorted(set(config.data.monthly_snapshot.dates))


def generate_folds(config: ExperimentConfig) -> list[MonthlyFold]:
    """Build the monthly folds for the configured split mode."""
    splits = config.splits
    if not splits.is_monthly:
        raise ValueError("generate_folds requires a monthly split mode")

    dates = monthly_dates(config)
    m = splits.min_train_months
    v = splits.validation_months
    t = splits.test_months

    if splits.mode == "fixed_monthly_snapshot":
        if splits.train_dates or splits.validation_dates or splits.test_dates:
            return [
                MonthlyFold(
                    fold_id=0,
                    train_dates=tuple(sorted(splits.train_dates)),
                    validation_dates=tuple(sorted(splits.validation_dates)),
                    test_dates=tuple(sorted(splits.test_dates)),
                )
            ]
        # derive a single fold: first m train, next v validation, rest test
        if len(dates) < m + v + 1:
            raise ValueError(
                f"fixed_monthly_snapshot needs >= {m + v + 1} dates, got {len(dates)}"
            )
        return [
            MonthlyFold(
                fold_id=0,
                train_dates=tuple(dates[:m]),
                validation_dates=tuple(dates[m : m + v]),
                test_dates=tuple(dates[m + v :]),
            )
        ]

    # expanding_monthly_snapshot
    step = max(1, int(getattr(splits, "step_months", 1)))
    folds: list[MonthlyFold] = []
    fold_id = 0
    offset = 0
    while True:
        train_end = m + offset
        val_end = train_end + v
        test_end = val_end + t
        if test_end > len(dates):
            break
        folds.append(
            MonthlyFold(
                fold_id=fold_id,
                train_dates=tuple(dates[:train_end]),
                validation_dates=tuple(dates[train_end:val_end]),
                test_dates=tuple(dates[val_end:test_end]),
            )
        )
        fold_id += 1
        offset += step
    if not folds:
        raise ValueError(
            f"expanding_monthly_snapshot produced no folds from {len(dates)} dates "
            f"(need >= {m + v + t})"
        )
    return folds


def assign_monthly_splits(date_values: np.ndarray, fold: MonthlyFold) -> np.ndarray:
    """Label each row train/validation/test/unused by its monthly_date and a fold."""
    out = np.empty(len(date_values), dtype=object)
    lookup = {}
    for d in fold.train_dates:
        lookup[_as_iso(d)] = SPLIT_TRAIN
    for d in fold.validation_dates:
        lookup[_as_iso(d)] = SPLIT_VALIDATION
    for d in fold.test_dates:
        lookup[_as_iso(d)] = SPLIT_TEST
    for i, raw in enumerate(date_values):
        out[i] = lookup.get(_as_iso(raw), SPLIT_UNUSED)
    return out


def _as_iso(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]
