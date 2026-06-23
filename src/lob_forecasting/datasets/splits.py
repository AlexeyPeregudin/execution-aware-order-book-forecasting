"""How rows get split into train / validation / test.

The layout in time order is:

    train | embargo | validation | embargo | test

The boundaries come from the configured fractions. The embargo is cut from the
end of the train and validation blocks: those rows' forward-looking labels
would otherwise reach into the next block and leak. The labels module and the
dataset module both import this so they agree on the split.
"""

from __future__ import annotations

import numpy as np

from ..config import SplitConfig

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLIT_EMBARGO = "embargo"
SPLIT_NAMES: tuple[str, ...] = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST, SPLIT_EMBARGO)


def split_boundaries(n_rows: int, splits: SplitConfig) -> tuple[int, int]:
    """The two boundary indices: train/val at b1, val/test at b2."""
    b1 = int(np.floor(splits.train_fraction * n_rows))
    b2 = int(np.floor((splits.train_fraction + splits.validation_fraction) * n_rows))
    return b1, b2


def assign_splits(n_rows: int, splits: SplitConfig) -> np.ndarray:
    """Label each of n_rows (in time order) as train/validation/test/embargo."""
    arr = np.empty(n_rows, dtype=object)
    if n_rows == 0:
        return arr
    b1, b2 = split_boundaries(n_rows, splits)
    arr[:b1] = SPLIT_TRAIN
    arr[b1:b2] = SPLIT_VALIDATION
    arr[b2:] = SPLIT_TEST

    e = splits.embargo_events
    if e > 0:
        # cut the embargo from the end of train and the end of validation
        arr[max(0, b1 - e):b1] = SPLIT_EMBARGO
        arr[max(b1, b2 - e):b2] = SPLIT_EMBARGO
    return arr


def training_mask(n_rows: int, splits: SplitConfig) -> np.ndarray:
    """Boolean mask for just the training rows."""
    return assign_splits(n_rows, splits) == SPLIT_TRAIN
