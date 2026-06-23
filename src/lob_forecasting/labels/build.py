"""Build the labels and write the features-labels tables.

Direction thresholds, regime bucket edges and (later) the scaler are all fitted
on training rows only. In monthly mode the canonical training rows are the
training months of the first expanding fold (the earliest data), so every fitted
transform is strictly causal. Labels are computed per monthly day so no forward
return crosses a day boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..datasets.splits import SPLIT_TRAIN, assign_splits
from ..features.feature_table import FeatureTableIndex
from ..features.regimes import (
    REGIME_DESCRIPTORS,
    RegimeThresholds,
    assign_regime_labels,
    fit_regime_thresholds,
)
from .compute import compute_labels, fit_thresholds, label_distribution
from .label_schema import (
    LabelledTableIndex,
    LabelPartition,
    LabelThresholds,
    enforce_labelled_schema,
    horizons,
)
from .markout import compute_markout_labels


def _canonical_training_mask(df: pd.DataFrame, config: ExperimentConfig) -> np.ndarray:
    """Boolean mask of the canonical training rows for fitting transforms."""
    if config.splits.is_monthly and "monthly_date" in df.columns:
        from ..datasets.monthly_splits import generate_folds

        folds = generate_folds(config)
        train_dates = {d.isoformat() for d in folds[0].train_dates}
        return df["monthly_date"].astype("string").isin(train_dates).to_numpy()
    return assign_splits(len(df), config.splits) == SPLIT_TRAIN


def _training_relative_spread(frames: list[pd.DataFrame], config: ExperimentConfig) -> np.ndarray:
    """Gather relative_spread from the canonical training rows of every frame."""
    parts: list[np.ndarray] = []
    for df in frames:
        mask = _canonical_training_mask(df, config)
        parts.append(df["relative_spread"].to_numpy(dtype="float64")[mask])
    return np.concatenate(parts) if parts else np.array([], dtype="float64")


def fit_label_thresholds(
    frames: list[pd.DataFrame], config: ExperimentConfig
) -> tuple[dict[str, float], float, int]:
    """Fit the direction thresholds from the frames' canonical training rows."""
    train_rs = _training_relative_spread(frames, config)
    n_train = int(np.isfinite(train_rs).sum())
    thresholds, median = fit_thresholds(
        train_rs, config.labels.direction_threshold_alpha, horizons(config)
    )
    return thresholds, median, n_train


def _fit_regime_thresholds(
    frames: list[pd.DataFrame], config: ExperimentConfig
) -> RegimeThresholds | None:
    """Fit regime bucket edges on the canonical training rows, pooled across frames."""
    if not config.features.include_regime_features:
        return None
    parts: list[pd.DataFrame] = []
    for df in frames:
        mask = _canonical_training_mask(df, config)
        parts.append(df.loc[mask, list(REGIME_DESCRIPTORS)])
    pooled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=REGIME_DESCRIPTORS)
    return fit_regime_thresholds(pooled, config)


def _compute_frame_labels(
    df: pd.DataFrame, config: ExperimentConfig, thresholds: dict[str, float]
) -> pd.DataFrame:
    """Labels for one feature frame, computed per monthly day in monthly mode."""
    hs = horizons(config)
    lab = config.labels
    want_markout = lab.include_markout_targets or lab.include_adverse_selection_targets

    def per_block(block: pd.DataFrame) -> pd.DataFrame:
        out = compute_labels(block["mid"], hs, thresholds)
        if want_markout:
            mk = compute_markout_labels(
                block["mid"],
                block["spread"],
                hs,
                include_markout=lab.include_markout_targets,
                include_adverse=lab.include_adverse_selection_targets,
            )
            out = pd.concat([out, mk], axis=1)
        return out

    if config.data.monthly_snapshot.enabled and "monthly_date" in df.columns:
        parts = [per_block(b) for _, b in df.groupby("monthly_date", sort=False)]
        return pd.concat(parts).sort_index()
    return per_block(df)


def build_labels(
    config: ExperimentConfig,
    features: FeatureTableIndex,
    run_id: str,
    project_root: str | Path | None = None,
) -> LabelledTableIndex:
    """Add the labels and write features_labels.parquet plus fitted-transform files."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    hs = horizons(config)
    transforms_dir = (
        root / config.data.artefact_dir / "runs" / run_id / "transforms"
    )

    # read every feature partition
    frames: list[tuple] = []
    for fp in features.partitions:
        path = root / fp.file_path
        if not path.exists():
            raise FileNotFoundError(f"Feature partition is missing: {path}")
        frames.append((fp, pd.read_parquet(path)))

    raw_frames = [df for _, df in frames]

    # fit direction thresholds on canonical training rows and save them
    thresholds, median, n_train = fit_label_thresholds(raw_frames, config)
    LabelThresholds(
        thresholds=thresholds,
        source=config.labels.direction_threshold_mode,
        alpha=config.labels.direction_threshold_alpha,
        median_relative_spread_train=median,
        n_train_rows=n_train,
    ).save(transforms_dir / "label_thresholds.yaml")

    # fit regime bucket edges on canonical training rows and save them
    regime_thresholds = _fit_regime_thresholds(raw_frames, config)
    if regime_thresholds is not None:
        regime_thresholds.save(transforms_dir / "regime_thresholds.yaml")

    # apply to every row and write the tables
    partitions: list[LabelPartition] = []
    for fp, df in frames:
        labels_df = _compute_frame_labels(df, config, thresholds)
        combined = pd.concat([df, labels_df], axis=1)
        if regime_thresholds is not None:
            regimes = assign_regime_labels(df[list(REGIME_DESCRIPTORS)], regime_thresholds)
            combined = pd.concat([combined, regimes], axis=1)
        combined = enforce_labelled_schema(combined, config)

        out_path = root / Path(fp.file_path).with_name("features_labels.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path, engine="pyarrow", index=False)

        partitions.append(
            LabelPartition(
                venue=fp.venue,
                symbol=fp.symbol,
                file_path=out_path.relative_to(root).as_posix(),
                row_count=len(combined),
                label_distribution=label_distribution(labels_df, hs),
            )
        )

    index = LabelledTableIndex(
        partitions=partitions,
        thresholds=thresholds,
        threshold_source=config.labels.direction_threshold_mode,
        alpha=config.labels.direction_threshold_alpha,
    )
    index.save(root / config.data.processed_dir.parent / "features" / "labelled_index.yaml")
    return index
