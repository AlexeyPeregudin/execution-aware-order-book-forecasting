"""Tests for the canonical temporal split assignment."""

from __future__ import annotations

import numpy as np

from lob_forecasting.config import SplitConfig
from lob_forecasting.datasets import (
    SPLIT_EMBARGO,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    assign_splits,
    split_boundaries,
    training_mask,
)


def _splits(train=0.6, val=0.2, test=0.2, embargo=10) -> SplitConfig:
    return SplitConfig(
        train_fraction=train,
        validation_fraction=val,
        test_fraction=test,
        embargo_events=embargo,
    )


def test_boundaries_from_fractions():
    b1, b2 = split_boundaries(100, _splits())
    assert b1 == 60
    assert b2 == 80


def test_regions_are_contiguous_and_ordered():
    arr = assign_splits(100, _splits(embargo=0))
    # No embargo: blocks are exactly the fractions.
    assert (arr[:60] == SPLIT_TRAIN).all()
    assert (arr[60:80] == SPLIT_VALIDATION).all()
    assert (arr[80:] == SPLIT_TEST).all()


def test_embargo_carved_from_block_tails():
    e = 10
    arr = assign_splits(100, _splits(embargo=e))
    # Embargo is the last E rows of train and of validation.
    assert (arr[50:60] == SPLIT_EMBARGO).all()   # train tail
    assert (arr[:50] == SPLIT_TRAIN).all()
    assert (arr[70:80] == SPLIT_EMBARGO).all()   # validation tail
    assert (arr[60:70] == SPLIT_VALIDATION).all()
    assert (arr[80:] == SPLIT_TEST).all()


def test_embargo_separates_used_blocks():
    e = 10
    arr = assign_splits(100, _splits(embargo=e))
    train_idx = np.flatnonzero(arr == SPLIT_TRAIN)
    val_idx = np.flatnonzero(arr == SPLIT_VALIDATION)
    test_idx = np.flatnonzero(arr == SPLIT_TEST)
    # At least E rows separate the last used train row and first used val row.
    assert val_idx.min() - train_idx.max() - 1 >= e
    assert test_idx.min() - val_idx.max() - 1 >= e


def test_no_overlap_between_splits():
    arr = assign_splits(137, _splits(embargo=7))
    counts = {name: int((arr == name).sum()) for name in set(arr)}
    assert sum(counts.values()) == 137  # every row assigned exactly once


def test_training_mask_matches_assignment():
    s = _splits(embargo=5)
    mask = training_mask(50, s)
    assert mask.dtype == bool
    assert (mask == (assign_splits(50, s) == SPLIT_TRAIN)).all()


def test_empty_input():
    arr = assign_splits(0, _splits())
    assert len(arr) == 0
