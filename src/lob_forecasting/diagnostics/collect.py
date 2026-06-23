"""Load everything the report needs.

Any of these can be missing (a stage might not have run), in which case the
field is just None and the report leaves that section out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ..config import ExperimentConfig
from ..datasets.dataset_schema import DatasetIndex
from ..features.feature_table import FeatureTableIndex
from ..ingestion.manifest import RawDataManifest
from ..labels.label_schema import LabelledTableIndex
from ..normalisation.event_table import EventTableIndex
from ..orderbook.book_table import BookTableIndex


def _read_parquet(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def _read_yaml(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class ReportData:
    """Everything the report draws from. Any field may be None."""

    config: ExperimentConfig
    run_id: str
    project_root: Path

    manifest: RawDataManifest | None = None
    events: EventTableIndex | None = None
    books: BookTableIndex | None = None
    features: FeatureTableIndex | None = None
    labelled: LabelledTableIndex | None = None
    datasets: DatasetIndex | None = None

    features_labels: pd.DataFrame | None = None  # all symbols together, for the histograms
    predictive_metrics: pd.DataFrame | None = None
    confusion: pd.DataFrame | None = None
    backtest_metrics: pd.DataFrame | None = None
    backtest_trades: pd.DataFrame | None = None
    threshold_selection: dict | None = None


def load_report_data(
    config: ExperimentConfig, run_id: str, project_root: str | Path | None = None
) -> ReportData:
    """Gather all the artefacts for a run into a ReportData."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    data_root = root / config.data.processed_dir.parent
    run_dir = root / config.data.artefact_dir / "runs" / run_id

    rd = ReportData(config=config, run_id=run_id, project_root=root)

    # the per-stage index files
    man_path = root / config.ingestion.manifest_path
    if man_path.exists():
        rd.manifest = RawDataManifest.load(man_path)

    events_idx = data_root / "processed" / "events" / "events_index.yaml"
    if events_idx.exists():
        rd.events = EventTableIndex.load(events_idx)
    books_idx = data_root / "processed" / "books" / "books_index.yaml"
    if books_idx.exists():
        rd.books = BookTableIndex.load(books_idx)
    features_idx = data_root / "features" / "features_index.yaml"
    if features_idx.exists():
        rd.features = FeatureTableIndex.load(features_idx)
    labelled_idx = data_root / "features" / "labelled_index.yaml"
    if labelled_idx.exists():
        rd.labelled = LabelledTableIndex.load(labelled_idx)
    meta_path = run_dir / "dataset_metadata.yaml"
    if meta_path.exists():
        rd.datasets = DatasetIndex.load(meta_path)

    # the actual feature rows, for the distribution plots
    if rd.labelled is not None and len(rd.labelled):
        frames = []
        for lp in rd.labelled.partitions:
            p = root / lp.file_path
            if p.exists():
                frames.append(pd.read_parquet(p))
        if frames:
            rd.features_labels = pd.concat(frames, ignore_index=True)

    # evaluation and backtest outputs
    rd.predictive_metrics = _read_parquet(run_dir / "metrics" / "predictive_metrics.parquet")
    rd.confusion = _read_parquet(run_dir / "metrics" / "confusion_matrices.parquet")
    rd.backtest_metrics = _read_parquet(run_dir / "backtests" / "taker_metrics.parquet")
    rd.backtest_trades = _read_parquet(run_dir / "backtests" / "taker_trades.parquet")
    rd.threshold_selection = _read_yaml(run_dir / "backtests" / "threshold_selection.yaml")

    return rd
