"""SHA-256 helpers. Used to checksum raw files so runs are reproducible."""

from __future__ import annotations

import hashlib
from pathlib import Path

# read files 1 MiB at a time so we don't load huge files into memory
_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    """sha256 of some bytes, as a hex string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = _CHUNK) -> str:
    """sha256 of a file, read in chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts) -> int:
    """Turn some parts (strings, dates, ints) into a fixed 64-bit seed.

    Python's built-in hash() is randomised per process, so we hash the parts
    ourselves. Same inputs always give the same seed, which keeps the synthetic
    data identical across runs.
    """
    joined = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], "big")
