# Data manifest & storage schema

This describes the manifest the ingestion step writes and where everything lives
on disk.

## Storage layers

```text
data/raw/                 # exact files from source, never modified
data/interim/             # parsed but not final
data/processed/events/    # normalised event tables
data/processed/books/     # top-K book states
data/features/            # features and labels
data/datasets/            # split-specific ML datasets
artefacts/                # models, predictions, metrics, reports
```

Raw files are **immutable** after ingestion. Every derived table must be
reproducible from raw files and configs.

## Manifest entry schema (`sources.yml`)

`data_manifest/sources.yml` is **generated** by the ingestion module. Each
entry under `entries:` has the following fields:

| Field | Type | Required | Description |
|---|---|:--:|---|
| `source_id` | string | yes | Stable id, `{venue}_{symbol}_{YYYYMMDD}` |
| `venue` | string | yes | Venue name (e.g. `binance`) |
| `symbol` | string | yes | Instrument symbol (e.g. `BTCUSDT`) |
| `date` | date | yes | Data date (`YYYY-MM-DD`) |
| `source_url_or_path` | string | yes | Original source URL or path |
| `file_path` | string | yes | Path to the stored raw file, relative to project root |
| `file_format` | string | yes | Raw file format (`csv`, `zip`, ...) |
| `schema_version` | string | yes | Manifest/raw schema version |
| `created_at_utc` | string | yes | ISO-8601 UTC time the entry was created |
| `checksum_sha256` | string | yes | SHA-256 of the raw file bytes (mandatory) |
| `row_count_or_null` | int \| null | no | Row count if known (CSV), else null |
| `notes` | string | no | Free-text notes |
| `collection_start_utc` | string \| null | no | Live-stream collection start, if applicable |
| `collection_end_utc` | string \| null | no | Live-stream collection end, if applicable |

`data_manifest/checksums.yml` is a flat `file_path -> sha256` map mirroring the
manifest, for fast skip/verify.

## Raw synthetic snapshot format (`mode: synthetic`)

The synthetic source generates deterministic top-K order-book snapshots as CSV:

```text
timestamp_ms,
bid_px_1,bid_qty_1,ask_px_1,ask_qty_1, ... ,bid_px_K,bid_qty_K,ask_px_K,ask_qty_K
```

* `timestamp_ms` - UTC epoch milliseconds, strictly increasing (100 ms apart);
* bid prices decrease by level; ask prices increase by level; quantities > 0.

Generation is seeded from `(symbol, date, synthetic.seed)`, so the same config
yields byte-identical files and therefore stable checksums.
