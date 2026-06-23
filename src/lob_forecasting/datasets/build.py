"""Build the train/validation/test datasets.

Uses the same assign_splits the labels module used, fits the scaler on the
training rows only, then writes two things per split: a tabular file (scaled
features + labels, for the simple models) and a sequence-index file (one row per
TCN window, which never crosses a split boundary).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..features.feature_table import feature_value_columns
from ..labels.label_schema import LabelledTableIndex, label_columns
from .dataset_schema import (
    SPLITS,
    DatasetIndex,
    DatasetSplitStats,
    context_feature_columns,
    sequence_index_columns,
    tabular_columns,
)
from .monthly_splits import MonthlyFold, assign_monthly_splits
from .scaler import FeatureScaler
from .splits import SPLIT_TRAIN, assign_splits


def _valid_feature_mask(
    df: pd.DataFrame, feature_cols: list[str], exclude_crossed: bool
) -> np.ndarray:
    """Which rows we can use: every feature present, and not a crossed book."""
    mask = df[feature_cols].notna().all(axis=1).to_numpy().copy()
    if exclude_crossed and "quality_flags" in df.columns:
        crossed = df["quality_flags"].fillna("").str.contains("crossed_book").to_numpy()
        mask &= ~crossed
    return mask


def _build_sequences(
    df: pd.DataFrame,
    split_arr: np.ndarray,
    valid: np.ndarray,
    split: str,
    seq_len: int,
    label_cols: list[str],
    block_arr: np.ndarray | None = None,
    context_cols: list[str] | None = None,
) -> pd.DataFrame:
    """One index row per length-L window of this split (fully vectorised).

    A window ending at position `end` is only kept if all of [end-L+1, end] is
    in this split and has valid features, so a window never straddles a split or
    the embargo. When `block_arr` is given (monthly mode), the whole window must
    also lie inside a single block (one monthly day), so no window crosses a
    monthly day boundary. Returns the window-index rows as a DataFrame so the
    builder scales to millions of windows without a Python dict per window.
    """
    context_cols = context_cols or []
    empty_cols = ["venue", "symbol", "split", "end_event_id", "end_timestamp_exchange_ns",
                  "window_start_index", "window_end_index", "seq_len", *label_cols, *context_cols]
    positions = np.flatnonzero(split_arr == split)
    if positions.size < seq_len:
        return pd.DataFrame(columns=empty_cols)
    lo, hi = int(positions.min()), int(positions.max()) + 1

    invalid = (~valid).astype("int64")
    pref = np.concatenate([[0], np.cumsum(invalid)])
    in_split = (split_arr == split).astype("int64")
    split_pref = np.concatenate([[0], np.cumsum(in_split)])

    ends = np.arange(lo + seq_len - 1, hi, dtype="int64")
    starts = ends - seq_len + 1
    ok = (pref[ends + 1] - pref[starts] == 0) & (split_pref[ends + 1] - split_pref[starts] == seq_len)
    if block_arr is not None:
        codes = pd.factorize(block_arr)[0]  # fast integer day-ids
        ok &= codes[starts] == codes[ends]
    ends = ends[ok]
    starts = starts[ok]
    if ends.size == 0:
        return pd.DataFrame(columns=empty_cols)

    out = {
        "venue": df["venue"].to_numpy()[ends],
        "symbol": df["symbol"].to_numpy()[ends],
        "split": split,
        "end_event_id": df["event_id"].to_numpy()[ends].astype("int64"),
        "end_timestamp_exchange_ns": df["timestamp_exchange_ns"].to_numpy()[ends].astype("int64"),
        "window_start_index": starts,
        "window_end_index": ends,
        "seq_len": seq_len,
    }
    for c in (*label_cols, *context_cols):
        out[c] = df[c].to_numpy()[ends]
    return pd.DataFrame(out)


def _split_arrays(
    df: pd.DataFrame, config: ExperimentConfig, fold: MonthlyFold | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-row split labels and (monthly mode) per-row day block ids."""
    if fold is not None:
        date_vals = df["monthly_date"].astype("string").to_numpy()
        return assign_monthly_splits(date_vals, fold), date_vals
    return assign_splits(len(df), config.splits), None


def build_datasets(
    config: ExperimentConfig,
    labelled: LabelledTableIndex,
    run_id: str,
    project_root: str | Path | None = None,
    fold: MonthlyFold | None = None,
) -> DatasetIndex:
    """Build the datasets for one (optionally fold-scoped) split.

    In fraction mode (`fold is None`) this writes to data/datasets/{run_id}/ and
    artefacts/runs/{run_id}/ exactly as before. With a `fold`, outputs go under a
    per-fold subfolder and the scaler is fit on that fold's training months only.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    data_root = root / config.data.processed_dir.parent
    subdir = f"{run_id}/{fold.name}" if fold is not None else run_id
    datasets_dir = data_root / "datasets" / subdir
    run_dir = root / config.data.artefact_dir / "runs" / run_id
    fold_dir = run_dir / "folds" / fold.name if fold is not None else run_dir
    transforms_dir = fold_dir / "transforms"

    feature_cols = feature_value_columns(config)
    label_cols = label_columns(config)
    seq_len = config.datasets.sequence_length
    exclude_crossed = config.datasets.exclude_crossed

    # first pass: load each symbol, work out the splits, and collect train rows
    per_symbol: list[tuple] = []
    train_feature_frames: list[pd.DataFrame] = []
    for lp in labelled.partitions:
        path = root / lp.file_path
        if not path.exists():
            raise FileNotFoundError(f"Features-labels partition is missing: {path}")
        df = pd.read_parquet(path).sort_values("timestamp_exchange_ns", kind="mergesort")
        df = df.reset_index(drop=True)
        split_arr, block_arr = _split_arrays(df, config, fold)
        valid = _valid_feature_mask(df, feature_cols, exclude_crossed)
        per_symbol.append((lp, df, split_arr, valid, block_arr))
        train_rows = (split_arr == SPLIT_TRAIN) & valid
        train_feature_frames.append(df.loc[train_rows, feature_cols])

    # fit the scaler on the training rows only and save it
    if train_feature_frames:
        train_features = pd.concat(train_feature_frames, ignore_index=True)
    else:
        train_features = pd.DataFrame(columns=feature_cols)
    scaler = FeatureScaler.fit(train_features, feature_cols)
    scaler_path = transforms_dir / "scaler.pkl"
    scaler.save(scaler_path)

    # fit the latent state-space model on this fold's training days and write the
    # causal filtered-state context for every row
    latent_paths = _fit_and_write_latent_state(
        config, per_symbol, fold_dir, transforms_dir, root)

    # second pass: build the tabular rows and sequence windows per split
    tabular: dict[str, list[pd.DataFrame]] = {s: [] for s in SPLITS}
    sequence: dict[str, list[pd.DataFrame]] = {s: [] for s in SPLITS}

    for lp, df, split_arr, valid, block_arr in per_symbol:
        scaled = scaler.transform_frame(df)  # scales features, leaves labels alone
        scaled["split"] = split_arr
        for split in SPLITS:
            usable = (split_arr == split) & valid
            sub = scaled.loc[usable, tabular_columns(config)]
            if len(sub):
                tabular[split].append(sub)
            sub_seq = _build_sequences(
                df, split_arr, valid, split, seq_len, label_cols, block_arr,
                context_feature_columns(config),
            )
            if len(sub_seq):
                sequence[split].append(sub_seq)

    # write the parquet files
    datasets_dir.mkdir(parents=True, exist_ok=True)
    tabular_paths: dict[str, str] = {}
    sequence_paths: dict[str, str] = {}
    stats: list[DatasetSplitStats] = []

    for split in SPLITS:
        if tabular[split]:
            tab = pd.concat(tabular[split], ignore_index=True)
        else:
            tab = pd.DataFrame(columns=tabular_columns(config))
        tab_path = datasets_dir / f"tabular_{split}.parquet"
        tab.to_parquet(tab_path, engine="pyarrow", index=False)
        tabular_paths[split] = tab_path.relative_to(root).as_posix()

        if sequence[split]:
            seq = pd.concat(sequence[split], ignore_index=True)[sequence_index_columns(config)]
        else:
            seq = pd.DataFrame(columns=sequence_index_columns(config))
        seq_path = datasets_dir / f"sequence_index_{split}.parquet"
        seq.to_parquet(seq_path, engine="pyarrow", index=False)
        sequence_paths[split] = seq_path.relative_to(root).as_posix()

        t_min = int(tab["timestamp_exchange_ns"].min()) if len(tab) else None
        t_max = int(tab["timestamp_exchange_ns"].max()) if len(tab) else None
        stats.append(
            DatasetSplitStats(
                split=split, n_tabular=len(tab), n_sequence=len(seq), time_min=t_min, time_max=t_max
            )
        )

    index = DatasetIndex(
        run_id=run_id,
        fold_id=fold.fold_id if fold is not None else None,
        fold_name=fold.name if fold is not None else None,
        symbols=list(config.data.symbols),
        feature_columns=feature_cols,
        label_columns=label_cols,
        sequence_length=seq_len,
        split_fractions={
            "train": config.splits.train_fraction,
            "validation": config.splits.validation_fraction,
            "test": config.splits.test_fraction,
        },
        embargo_events=config.splits.embargo_events,
        tabular_paths=tabular_paths,
        sequence_paths=sequence_paths,
        scaler_path=scaler_path.relative_to(root).as_posix(),
        latent_context_path=latent_paths[0],
        latent_transform_path=latent_paths[1],
        latent_state_dim=latent_paths[2],
        stats=stats,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    index.save(fold_dir / "dataset_metadata.yaml")
    return index


def _fit_and_write_latent_state(
    config: ExperimentConfig, per_symbol: list[tuple], fold_dir: Path,
    transforms_dir: Path, root: Path,
) -> tuple[str, str, int]:
    """Fit the per-fold SSM on training days and write the filtered context.

    Returns (context_path, transform_path, state_dim) as project-relative strings;
    all empty/zero when the latent module is disabled or its observation columns
    are absent. The SSM is fit on training-day observations only; the emitted
    context is the causal filtered state, reset at each monthly-day boundary.
    """
    ls = getattr(config, "latent_state", None)
    if ls is None or not ls.enabled:
        return "", "", 0
    obs_cols = list(ls.observation_columns)
    for _, df, _, _, _ in per_symbol:
        if not all(c in df.columns for c in obs_cols):
            return "", "", 0
    from ..features.latent_state import (
        fit_latent_state,
        filtered_context,
        save_ssm,
    )

    # training-day observation blocks (per day, train split, valid rows)
    blocks: list[np.ndarray] = []
    for _, df, split_arr, valid, block_arr in per_symbol:
        train_mask = (split_arr == SPLIT_TRAIN) & valid
        if not train_mask.any():
            continue
        sub = df.loc[train_mask]
        day_key = sub["monthly_date"].astype("string") if "monthly_date" in sub.columns else None
        if day_key is not None:
            for _, day in sub.groupby(day_key, sort=False):
                blocks.append(day[obs_cols].to_numpy(dtype="float64"))
        else:
            blocks.append(sub[obs_cols].to_numpy(dtype="float64"))
    if not blocks:
        return "", "", 0

    ssm = fit_latent_state(
        blocks, obs_cols, state_dim=int(ls.state_dim),
        max_em_iterations=int(ls.max_em_iterations), loglik_tol=float(ls.loglik_tol),
    )
    transform_path = transforms_dir / "latent_state_space.pkl"
    save_ssm(ssm, transform_path)

    # causal filtered context for every row of every partition
    ctx_frames: list[pd.DataFrame] = []
    for _, df, _, _, _ in per_symbol:
        ctx = filtered_context(df, ssm, day_column="monthly_date",
                               reset_each_day=bool(ls.reset_each_monthly_day))
        keys = df[["venue", "symbol", "timestamp_exchange_ns"]].reset_index(drop=True)
        ctx_frames.append(pd.concat([keys, ctx.reset_index(drop=True)], axis=1))
    context = pd.concat(ctx_frames, ignore_index=True)
    context_dir = fold_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    context_path = context_dir / "latent_state_context.parquet"
    context.to_parquet(context_path, engine="pyarrow", index=False)
    return (context_path.relative_to(root).as_posix(),
            transform_path.relative_to(root).as_posix(), int(ls.state_dim))
