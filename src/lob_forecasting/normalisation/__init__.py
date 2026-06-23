"""Normalisation: raw files -> a clean event table."""

from .adapters import (
    NormalisationError,
    ParseResult,
    SnapshotCsvAdapter,
    VenueAdapter,
    build_venue_adapter,
)
from .event_table import (
    CANONICAL_EVENT_TYPES,
    CANONICAL_SIDES,
    EVENT_COLUMNS,
    EVENT_DTYPES,
    EventPartition,
    EventTableIndex,
    enforce_event_schema,
    flag_counts,
)
from .normalise import finalise_events, normalise_events

__all__ = [
    # Orchestration
    "normalise_events",
    "finalise_events",
    # Event-table schema
    "EVENT_COLUMNS",
    "EVENT_DTYPES",
    "CANONICAL_EVENT_TYPES",
    "CANONICAL_SIDES",
    "enforce_event_schema",
    "flag_counts",
    "EventTableIndex",
    "EventPartition",
    # Adapters
    "VenueAdapter",
    "SnapshotCsvAdapter",
    "ParseResult",
    "build_venue_adapter",
    "NormalisationError",
]
