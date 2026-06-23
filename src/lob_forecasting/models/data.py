"""ModelData: what a model gets to work with, plus a loader for it.

For one split it holds the tabular features+labels, the fitted scaler (so the
imbalance rule can undo the scaling and see the raw imbalance), and, for the
TCN, the actual sequence windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.scaler import FeatureScaler

_ID_COLS = ["venue", "symbol", "event_id", "timestamp_exchange_ns"]


@dataclass
class ModelData:
    """Model inputs for one split. Sequence windows are only there if loaded."""

    split: str
    frame: pd.DataFrame
    feature_names: list[str]
    horizons: list[int]
    scaler: FeatureScaler | None = None
    sequence_index: pd.DataFrame | None = None
    sequence_length: int = 0
    sequence_matrices: list[np.ndarray] | None = None
    seq_mat_code: np.ndarray | None = None
    seq_start: np.ndarray | None = None
    context_matrix: np.ndarray | None = None  # (n_windows, C) regime one-hot
    context_names: list[str] | None = None

    # tabular access

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    def X(self) -> np.ndarray:
        """Scaled feature matrix, shape (n_rows, n_features)."""
        return self.frame[self.feature_names].to_numpy(dtype="float64")

    def ids(self) -> pd.DataFrame:
        return self.frame[_ID_COLS].reset_index(drop=True)

    def true_return(self, h: int) -> np.ndarray:
        return self.frame[f"r_h{h}"].to_numpy(dtype="float64")

    def true_direction(self, h: int) -> np.ndarray:
        return self.frame[f"y_dir_h{h}"].astype("float64").to_numpy()

    def available(self, h: int) -> np.ndarray:
        return self.frame[f"label_available_h{h}"].to_numpy(dtype="bool")

    def raw_feature(self, name: str) -> np.ndarray:
        """Undo the scaling to get a feature back in its original units."""
        x = self.frame[name].to_numpy(dtype="float64")
        if self.scaler is None or name not in self.scaler.feature_names:
            return x
        i = self.scaler.feature_names.index(name)
        return x * self.scaler.scale_[i] + self.scaler.mean_[i]

    # sequence windows for the TCN

    @property
    def n_sequences(self) -> int:
        return 0 if self.sequence_index is None else len(self.sequence_index)

    def seq_window_batch(self, idx: np.ndarray) -> np.ndarray:
        if self.sequence_matrices is None or self.seq_start is None or self.seq_mat_code is None:
            raise ValueError("Sequence windows were not loaded for this ModelData.")
        idx = np.asarray(idx)
        L = self.sequence_length
        f = len(self.feature_names)
        out = np.empty((len(idx), L, f), dtype="float32")
        starts = self.seq_start[idx]
        codes = self.seq_mat_code[idx]
        for j in range(len(idx)):
            mat = self.sequence_matrices[int(codes[j])]
            s = int(starts[j])
            out[j] = mat[s : s + L]
        return out

    def seq_X(self) -> np.ndarray:
        return self.seq_window_batch(np.arange(self.n_sequences))

    def seq_ids(self) -> pd.DataFrame:
        seq = self.sequence_index
        return pd.DataFrame(
            {
                "venue": seq["venue"].to_numpy(),
                "symbol": seq["symbol"].to_numpy(),
                "event_id": seq["end_event_id"].to_numpy(),
                "timestamp_exchange_ns": seq["end_timestamp_exchange_ns"].to_numpy(),
            }
        )

    def seq_true_return(self, h: int) -> np.ndarray:
        return self.sequence_index[f"r_h{h}"].to_numpy(dtype="float64")

    def seq_true_direction(self, h: int) -> np.ndarray:
        return self.sequence_index[f"y_dir_h{h}"].astype("float64").to_numpy()

    def seq_available(self, h: int) -> np.ndarray:
        return self.sequence_index[f"label_available_h{h}"].to_numpy(dtype="bool")

    # execution-aware sequence targets for the multi-task TCN

    @property
    def context_dim(self) -> int:
        return 0 if self.context_matrix is None else int(self.context_matrix.shape[1])

    def seq_context_batch(self, idx: np.ndarray) -> np.ndarray:
        if self.context_matrix is None:
            return np.zeros((len(idx), 0), dtype="float32")
        return self.context_matrix[np.asarray(idx)]

    def seq_markout(self, side: str, h: int) -> np.ndarray:
        return self.sequence_index[f"markout_{side}_h{h}"].to_numpy(dtype="float64")

    def seq_adverse(self, side: str, h: int) -> np.ndarray:
        return self.sequence_index[f"adverse_{side}_h{h}"].to_numpy(dtype="float64")

    def seq_markout_available(self, h: int) -> np.ndarray:
        col = f"markout_available_h{h}"
        if col not in self.sequence_index.columns:
            return self.seq_available(h)
        return self.sequence_index[col].to_numpy(dtype="bool")


def _gather_window_matrices(
    config, seq: pd.DataFrame, scaler: FeatureScaler, feature_names: list[str], root: Path
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Load each symbol's scaled feature matrix once and index the windows into it.

    Returns the per-symbol matrices, a per-window matrix code, and a per-window
    start index. Windows are then sliced on demand (a batch at a time) instead of
    materialising every (L, F) window up front.
    """
    if len(seq) == 0:
        return [], np.zeros(0, dtype="int64"), np.zeros(0, dtype="int64")

    features_root = root / config.data.processed_dir.parent / "features"
    venues = seq["venue"].to_numpy()
    symbols = seq["symbol"].to_numpy()
    starts = seq["window_start_index"].to_numpy().astype("int64")

    pairs = list(zip(venues.tolist(), symbols.tolist()))
    unique = list(dict.fromkeys(pairs))
    code_of = {key: i for i, key in enumerate(unique)}

    matrices: list[np.ndarray] = []
    for venue, symbol in unique:
        fl = pd.read_parquet(
            features_root / f"venue={venue}" / f"symbol={symbol}" / "features_labels.parquet"
        )
        fl = fl.sort_values("timestamp_exchange_ns", kind="mergesort").reset_index(drop=True)
        matrices.append(scaler.transform(fl[feature_names]).astype("float32"))

    codes = np.fromiter((code_of[p] for p in pairs), dtype="int64", count=len(pairs))
    return matrices, codes, starts


def load_model_data(
    config,
    dataset_index,
    split: str,
    project_root: str | Path | None = None,
    with_sequences: bool = False,
    include_latent_context: bool = False,
) -> ModelData:
    """Load the ModelData for one split from the dataset files.

    When `include_latent_context` is set and the fold has a fitted latent
    state-space context, the causal filtered state and variance columns are
    appended to the sequence context matrix. This is set per model variant so SSM
    and no-SSM runs can be paired on identical datasets.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    feature_names = list(dataset_index.feature_columns)
    horizons = list(config.sampling.horizons_events)

    frame = pd.read_parquet(root / dataset_index.tabular_paths[split])
    scaler = FeatureScaler.load(root / dataset_index.scaler_path)

    md = ModelData(
        split=split,
        frame=frame,
        feature_names=feature_names,
        horizons=horizons,
        scaler=scaler,
    )
    if with_sequences:
        seq = pd.read_parquet(root / dataset_index.sequence_paths[split])
        md.sequence_index = seq
        md.sequence_length = config.datasets.sequence_length
        matrices, codes, starts = _gather_window_matrices(config, seq, scaler, feature_names, root)
        md.sequence_matrices = matrices
        md.seq_mat_code = codes
        md.seq_start = starts
        ctx_parts: list[np.ndarray] = []
        ctx_names: list[str] = []
        if config.features.include_regime_features:
            from ..features.regimes import encode_context_one_hot

            ctx, names = encode_context_one_hot(seq)
            ctx_parts.append(ctx)
            ctx_names.extend(names)
        if include_latent_context and getattr(dataset_index, "latent_context_path", ""):
            lat, lat_names = _latent_context_for_sequences(dataset_index, seq, root)
            if lat is not None:
                ctx_parts.append(lat)
                ctx_names.extend(lat_names)
        if ctx_parts:
            md.context_matrix = np.hstack(ctx_parts).astype("float32")
            md.context_names = ctx_names
    return md


def _latent_context_for_sequences(dataset_index, seq: pd.DataFrame, root: Path):
    """Align per-fold filtered SSM context to each window-end event (by timestamp)."""
    from ..features.latent_state import latent_state_columns

    path = root / dataset_index.latent_context_path
    if not path.exists() or len(seq) == 0:
        return None, []
    cols = latent_state_columns(int(dataset_index.latent_state_dim))
    ctx = pd.read_parquet(path)
    key = pd.DataFrame({
        "venue": seq["venue"].to_numpy(),
        "symbol": seq["symbol"].to_numpy(),
        "timestamp_exchange_ns": seq["end_timestamp_exchange_ns"].to_numpy(),
    })
    merged = key.merge(ctx, on=["venue", "symbol", "timestamp_exchange_ns"], how="left")
    mat = merged[cols].to_numpy(dtype="float32")
    mat = np.nan_to_num(mat)
    return mat, cols
