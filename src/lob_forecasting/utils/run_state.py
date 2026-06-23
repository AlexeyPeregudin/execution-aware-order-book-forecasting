"""Keeps track of the 'current' run id in a small text file.

Everything for a run lives under artefacts/runs/{run_id}/. The dataset step
picks the run id, and the later steps (train, evaluate, backtest, report) need
to use the same one. Rather than make the user pass the id around, we just
write it to a file and read it back.
"""

from __future__ import annotations

from pathlib import Path

_POINTER = "artefacts/current_run.txt"


def write_current_run(run_id: str, project_root: str | Path | None = None) -> Path:
    """Save run_id as the current run."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    path = root / _POINTER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run_id.strip() + "\n", encoding="utf-8")
    return path


def read_current_run(project_root: str | Path | None = None) -> str | None:
    """Read the current run id, or None if there isn't one yet."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    path = root / _POINTER
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None
