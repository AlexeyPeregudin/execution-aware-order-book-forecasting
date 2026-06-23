"""Ingestion: get the raw files and write the manifest."""

from .ingest import (
    ingest_raw_data,
    make_source_id,
    verify_raw_files,
)
from .manifest import ManifestEntry, RawDataManifest, write_checksums
from .sources import (
    DataSourceAdapter,
    FixtureAdapter,
    IngestionError,
    LocalArchiveAdapter,
    SourceSpec,
    SyntheticSampleAdapter,
    UrlArchiveAdapter,
    build_adapter,
)

__all__ = [
    # Driver
    "ingest_raw_data",
    "verify_raw_files",
    "make_source_id",
    # Manifest
    "RawDataManifest",
    "ManifestEntry",
    "write_checksums",
    # Sources
    "DataSourceAdapter",
    "SourceSpec",
    "SyntheticSampleAdapter",
    "LocalArchiveAdapter",
    "FixtureAdapter",
    "UrlArchiveAdapter",
    "build_adapter",
    "IngestionError",
]
