"""Lightweight progress helpers.

Uses tqdm when it is installed and attached to a real terminal; otherwise it
falls back to occasional flushed prints, so a backgrounded/piped run still shows
roughly where it is. Nothing here changes results, it is purely cosmetic.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from typing import Iterable, Iterator

try:  # optional dependency
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def log(msg: str) -> None:
    """Print a timestamped, flushed progress line."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def progress(iterable: Iterable, desc: str = "", total: int | None = None,
             every: int = 0) -> Iterator:
    """Iterate with a tqdm bar (interactive) or periodic flushed prints (piped)."""
    if _tqdm is not None and sys.stderr.isatty():
        yield from _tqdm(iterable, desc=desc, total=total, leave=False)
        return
    # fallback: print every `every` items (or ~20 updates if total is known)
    if every <= 0 and total:
        every = max(1, total // 20)
    every = every or 1000
    start = time.time()
    for i, item in enumerate(iterable):
        if i % every == 0:
            frac = f"{i}/{total}" if total else f"{i}"
            log(f"  {desc}: {frac}  ({time.time()-start:.0f}s)")
        yield item
