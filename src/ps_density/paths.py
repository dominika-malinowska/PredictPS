"""Load PredictPS data paths from YAML configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the repository root (directory that contains ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[2]


def _config_file() -> Path:
    """Return the paths YAML file, honouring ``PREDICTPS_PATHS_FILE``."""
    if env_path := os.environ.get("PREDICTPS_PATHS_FILE"):
        return Path(env_path)

    custom = repo_root() / "config" / "paths.yaml"
    if custom.exists():
        return custom

    return repo_root() / "config" / "paths.example.yaml"


def _resolve_placeholders(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_placeholders(item, context) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_placeholders(item, context) for item in value]
    if isinstance(value, str):
        resolved = value
        for key, replacement in context.items():
            resolved = resolved.replace("{" + key + "}", replacement)
        return resolved
    return value


def _to_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_paths(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_paths(item) for item in value]
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return Path(value)
    return value


@lru_cache
def load_paths() -> SimpleNamespace:
    """Load and resolve paths from ``config/paths.yaml``.

    ``PREDICTPS_DATA_ROOT`` overrides ``data_root``. ``PREDICTPS_PATHS_FILE``
    selects a YAML file other than ``config/paths.yaml``.

    Returns:
        Namespace of resolved paths. Strings that contain a slash become
        ``Path`` objects. Names without a slash (experiment names, the
        osmconvert executable, the OSM PBF filename) stay strings.
    """
    config_path = _config_file()
    with config_path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)

    data_root = os.environ.get("PREDICTPS_DATA_ROOT", raw["data_root"])
    data_root = str(data_root).rstrip("/\\")
    resolved = _resolve_placeholders(raw, {"data_root": data_root})
    resolved["data_root"] = Path(data_root)
    return _to_paths(resolved)
