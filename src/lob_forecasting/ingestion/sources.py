"""Where raw data comes from. Each adapter lists the files it should produce
and hands back the bytes for one file. The rest of the pipeline doesn't care
which adapter was used once the files and manifest exist.

Adapters:
  synthetic      - make up a top-K snapshot CSV (no files needed)
  local_archive  - copy from local files
  fixture        - same as local_archive, just a clearer name for tests
  url_archive    - download over http(s)
"""

from __future__ import annotations

import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import timedelta
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig, IngestionConfig
from ..utils.hashing import stable_seed


class IngestionError(RuntimeError):
    """A raw file couldn't be fetched or failed its checksum."""


@dataclass
class SourceSpec:
    """One raw file we want, for a given (symbol, date)."""

    venue: str
    symbol: str
    date: date_type
    file_format: str
    source_url_or_path: str
    expected_checksum: str | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


# base prices for the fake data; anything else starts at 100
_DEFAULT_BASE_PRICE = {
    "BTCUSDT": 50_000.0,
    "ETHUSDT": 3_000.0,
}


def _dates_in_range(start: date_type, end: date_type) -> list[date_type]:
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def _config_dates(config: ExperimentConfig) -> list[date_type]:
    """Dates to ingest: the monthly snapshots when enabled, else the full range."""
    if config.data.monthly_snapshot.enabled and config.data.monthly_dates:
        return config.data.monthly_dates
    return _dates_in_range(config.data.start_date, config.data.end_date)


class DataSourceAdapter(ABC):
    """What every source adapter has to provide."""

    mode: str

    @abstractmethod
    def enumerate(self, config: ExperimentConfig, project_root: Path) -> list[SourceSpec]:
        """List the raw files that should exist for this config."""

    @abstractmethod
    def read_bytes(self, spec: SourceSpec) -> bytes:
        """Return the raw bytes for one file."""


class SyntheticSampleAdapter(DataSourceAdapter):
    """Makes up a top-K snapshot CSV so a fresh checkout can run with no data.

    The CSV has one row per book observation:
        timestamp_ms, bid_px_1, bid_qty_1, ask_px_1, ask_qty_1, ... (K levels)
    """

    mode = "synthetic"

    def __init__(self, ing: IngestionConfig) -> None:
        self.ing = ing
        self.syn = ing.synthetic

    def enumerate(self, config: ExperimentConfig, project_root: Path) -> list[SourceSpec]:
        if config.data.monthly_snapshot.enabled and config.data.monthly_dates:
            dates = config.data.monthly_dates  # exactly the first-of-month snapshots
        else:
            all_dates = _dates_in_range(config.data.start_date, config.data.end_date)
            dates = all_dates[: self.syn.num_days]
        specs: list[SourceSpec] = []
        for symbol in config.data.symbols:
            for d in dates:
                specs.append(
                    SourceSpec(
                        venue=config.data.venue,
                        symbol=symbol,
                        date=d,
                        file_format="csv",
                        source_url_or_path=f"synthetic://{config.data.venue}/{symbol}/{d.isoformat()}",
                        notes="synthetic top-K snapshot sample",
                        extra={"top_k": config.data.top_k},
                    )
                )
        return specs

    def read_bytes(self, spec: SourceSpec) -> bytes:
        top_k = int(spec.extra.get("top_k", 10))
        rows = self.syn.rows_per_day
        base = self.syn.base_price
        if base is None:
            base = _DEFAULT_BASE_PRICE.get(spec.symbol, 100.0)
        # seed from symbol+date so each file is reproducible on its own
        seed = stable_seed(spec.symbol, spec.date.isoformat(), self.syn.seed)
        return _generate_snapshot_csv(base=base, rows=rows, top_k=top_k, seed=seed, day=spec.date)


def _generate_snapshot_csv(
    *, base: float, rows: int, top_k: int, seed: int, day: date_type
) -> bytes:
    rng = np.random.default_rng(seed)
    tick = max(round(base * 1e-5, 8), 0.01)
    half_spread = tick  # so best_ask - best_bid is one tick on each side

    # mid price is a random walk
    steps = rng.normal(0.0, tick, size=rows)
    mid = base + np.cumsum(steps)

    # timestamps: midnight of the day, then every 100 ms
    day_start_ms = int(np.datetime64(day.isoformat(), "ms").astype("int64"))
    ts = day_start_ms + 100 * np.arange(rows, dtype=np.int64)

    header_cols = ["timestamp_ms"]
    for k in range(1, top_k + 1):
        header_cols += [f"bid_px_{k}", f"bid_qty_{k}", f"ask_px_{k}", f"ask_qty_{k}"]
    lines = [",".join(header_cols)]

    # random sizes for every level; [:, :, 0] is bid, [:, :, 1] is ask
    qty = rng.uniform(0.1, 5.0, size=(rows, top_k, 2))
    for i in range(rows):
        best_bid = mid[i] - half_spread
        best_ask = mid[i] + half_spread
        cells = [str(int(ts[i]))]
        for k in range(top_k):
            bid_px = round(best_bid - k * tick, 2)
            ask_px = round(best_ask + k * tick, 2)
            bid_qty = round(float(qty[i, k, 0]), 4)
            ask_qty = round(float(qty[i, k, 1]), 4)
            cells += [f"{bid_px}", f"{bid_qty}", f"{ask_px}", f"{ask_qty}"]
        lines.append(",".join(cells))

    return ("\n".join(lines) + "\n").encode("utf-8")


class LocalArchiveAdapter(DataSourceAdapter):
    """Reads raw files from a local folder: {source_root}/{venue}/{symbol}/{date}.{ext}"""

    mode = "local_archive"

    def __init__(self, ing: IngestionConfig) -> None:
        if ing.source_root is None:
            raise IngestionError("local_archive/fixture mode requires 'source_root'")
        self.ing = ing
        self.source_root = Path(ing.source_root)

    def _path_for(self, venue: str, symbol: str, d: date_type) -> Path:
        return self.source_root / venue / symbol / f"{d.isoformat()}.{self.ing.file_format}"

    def enumerate(self, config: ExperimentConfig, project_root: Path) -> list[SourceSpec]:
        specs: list[SourceSpec] = []
        for symbol in config.data.symbols:
            for d in _config_dates(config):
                src = self._path_for(config.data.venue, symbol, d)
                if not src.exists():
                    continue  # only list files that are actually there
                specs.append(
                    SourceSpec(
                        venue=config.data.venue,
                        symbol=symbol,
                        date=d,
                        file_format=self.ing.file_format,
                        source_url_or_path=str(src),
                        notes=f"{self.mode} copy",
                    )
                )
        return specs

    def read_bytes(self, spec: SourceSpec) -> bytes:
        src = Path(spec.source_url_or_path)
        if not src.exists():
            raise IngestionError(f"Source file not found: {src}")
        return src.read_bytes()


class FixtureAdapter(LocalArchiveAdapter):
    """Same as LocalArchiveAdapter, named differently for test fixtures."""

    mode = "fixture"


class TardisArchiveAdapter(DataSourceAdapter):
    """Reads Tardis-style daily archives laid out as

        {source_root}/{YYYY}/{MM}/{DD}/{SYMBOL}.csv.gz

    (e.g. the `binance-futures/book_snapshot_5` export). Only the configured
    monthly-snapshot dates are enumerated. The bytes are copied verbatim into the
    raw tree; the Tardis columns are parsed later by the normalisation adapter.
    """

    mode = "tardis_archive"

    def __init__(self, ing: IngestionConfig) -> None:
        if ing.source_root is None:
            raise IngestionError("tardis_archive mode requires 'source_root'")
        self.ing = ing
        self.source_root = Path(ing.source_root)
        self.file_format = ing.file_format or "csv.gz"

    def _path_for(self, symbol: str, d: date_type) -> Path:
        return (
            self.source_root
            / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}"
            / f"{symbol}.{self.file_format}"
        )

    def enumerate(self, config: ExperimentConfig, project_root: Path) -> list[SourceSpec]:
        specs: list[SourceSpec] = []
        for symbol in config.data.symbols:
            for d in _config_dates(config):
                src = self._path_for(symbol, d)
                if not src.exists():
                    continue
                specs.append(
                    SourceSpec(
                        venue=config.data.venue,
                        symbol=symbol,
                        date=d,
                        file_format=self.file_format,
                        source_url_or_path=str(src),
                        notes="tardis_archive copy",
                    )
                )
        return specs

    def read_bytes(self, spec: SourceSpec) -> bytes:
        src = Path(spec.source_url_or_path)
        if not src.exists():
            raise IngestionError(f"Tardis source file not found: {src}")
        return src.read_bytes()


class UrlArchiveAdapter(DataSourceAdapter):
    """Downloads raw files over http(s) using url_template."""

    mode = "url_archive"

    def __init__(self, ing: IngestionConfig) -> None:
        if not ing.base_url:
            raise IngestionError("url_archive mode requires 'base_url'")
        self.ing = ing

    def _url_for(self, symbol: str, d: date_type) -> str:
        return self.ing.url_template.format(
            base_url=self.ing.base_url.rstrip("/"),
            symbol=symbol,
            date=d.isoformat(),
            ext=self.ing.file_format,
        )

    def enumerate(self, config: ExperimentConfig, project_root: Path) -> list[SourceSpec]:
        specs: list[SourceSpec] = []
        for symbol in config.data.symbols:
            for d in _config_dates(config):
                specs.append(
                    SourceSpec(
                        venue=config.data.venue,
                        symbol=symbol,
                        date=d,
                        file_format=self.ing.file_format,
                        source_url_or_path=self._url_for(symbol, d),
                        notes="url_archive download",
                    )
                )
        return specs

    def read_bytes(self, spec: SourceSpec) -> bytes:
        try:
            with urllib.request.urlopen(spec.source_url_or_path, timeout=60) as resp:
                return resp.read()
        except Exception as exc:
            raise IngestionError(f"Failed to download {spec.source_url_or_path}: {exc}") from exc


def build_adapter(ing: IngestionConfig) -> DataSourceAdapter:
    """Pick the adapter for ing.mode."""
    if ing.mode == "synthetic":
        return SyntheticSampleAdapter(ing)
    if ing.mode == "local_archive":
        return LocalArchiveAdapter(ing)
    if ing.mode == "fixture":
        return FixtureAdapter(ing)
    if ing.mode == "tardis_archive":
        return TardisArchiveAdapter(ing)
    if ing.mode == "url_archive":
        return UrlArchiveAdapter(ing)
    raise IngestionError(f"Unknown ingestion mode: {ing.mode!r}")
