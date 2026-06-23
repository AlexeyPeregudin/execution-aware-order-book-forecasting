"""Load a config from YAML, apply CLI overrides, and save the resolved copy."""

from __future__ import annotations

import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .schema import ExperimentConfig


def generate_run_id() -> str:
    """A run id like 20240131T091500_ab12cd. Sorts by time, unique enough."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{ts}_{suffix}"


def get_git_commit() -> str | None:
    """Short git commit hash, or None if we're not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def parse_cli_overrides(overrides: list[str]) -> dict[str, Any]:
    """Turn ['data.top_k=5', 'random_seed=7'] into a nested dict.

    The value after '=' is parsed as YAML, so ints, floats, bools and lists
    like [10,50,200] all work without quoting.
    """
    result: dict[str, Any] = {}
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"Invalid override {raw!r}: expected 'dotted.key=value'")
        key_path, _, raw_value = raw.partition("=")
        keys = [k.strip() for k in key_path.strip().split(".")]
        if not all(keys):
            raise ValueError(f"Invalid key path in override {raw!r}")
        value = yaml.safe_load(raw_value.strip())
        # walk/create the nested dict, then set the leaf
        node: dict[str, Any] = result
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return result


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Merge overrides into base in place, recursing into nested dicts."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def load_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> tuple[ExperimentConfig, str]:
    """Read and validate a config file. Returns (config, run_id).

    overrides is an optional nested dict merged on top of the file before
    validation (build it with parse_cli_overrides).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    if overrides:
        _deep_merge(raw, overrides)

    config = ExperimentConfig.model_validate(raw)
    run_id = generate_run_id()
    return config, run_id


def save_resolved_config(
    config: ExperimentConfig,
    run_id: str,
    project_root: Path | None = None,
) -> Path:
    """Write the fully-resolved config to the run folder.

    Goes to {artefact_dir}/runs/{run_id}/resolved_config.yaml with a small
    _meta block (run id, time, git commit) so a run can be reproduced later.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    out_dir = root / config.data.artefact_dir / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "resolved_config.yaml"

    # mode="json" so dates and Paths come out as plain strings
    resolved: dict[str, Any] = config.model_dump(mode="json")
    resolved["_meta"] = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "config_schema_version": "1",
    }

    with out_path.open("w", encoding="utf-8") as fh:
        yaml.dump(resolved, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return out_path
