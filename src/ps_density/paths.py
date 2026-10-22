"""Load config/paths.yaml with optional environment overrides."""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATHS_FILE = _REPO_ROOT / "config" / "paths.yaml"


class Paths(SimpleNamespace):
    """Attribute access for nested path mappings."""


def _to_namespace(value):
    if isinstance(value, dict):
        return Paths(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _interpolate(value, mapping):
    if isinstance(value, dict):
        return {key: _interpolate(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate(item, mapping) for item in value]
    if isinstance(value, str):
        return value.format_map(mapping)
    return value


def load_paths(paths_file=None):
    """Load and interpolate the paths YAML.

    Resolution order for the file:
      1. ``paths_file`` argument
      2. ``PREDICTPS_PATHS_FILE``
      3. ``config/paths.yaml`` in the repository

    ``PREDICTPS_DATA_ROOT`` overrides ``data_root`` after the file is read.
    """
    file_path = Path(
        paths_file or os.environ.get("PREDICTPS_PATHS_FILE") or _DEFAULT_PATHS_FILE
    )
    with file_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data_root = os.environ.get("PREDICTPS_DATA_ROOT", raw.get("data_root", ""))
    raw["data_root"] = data_root
    interpolated = _interpolate(raw, {"data_root": data_root})
    return _to_namespace(interpolated)


def tiles_from_bbox(aoi_lon, aoi_lat, exclude_tiles=None):
    """Expand a lon/lat box to 1x1 degree south-west corners.

    Tiles that intersect ``[min_lon, max_lon] x [min_lat, max_lat]`` are kept.
    ``exclude_tiles`` is a list of ``[lon, lat]`` south-west corners to drop.
    """
    min_lon, max_lon = sorted(float(v) for v in aoi_lon)
    min_lat, max_lat = sorted(float(v) for v in aoi_lat)
    west0 = math.floor(min_lon)
    west1 = math.ceil(max_lon)
    south0 = math.floor(min_lat)
    south1 = math.ceil(max_lat)
    excluded = {tuple(int(c) for c in pair) for pair in (exclude_tiles or [])}
    tiles = []
    for west in range(west0, west1):
        for south in range(south0, south1):
            if (west, south) not in excluded:
                tiles.append((west, south))
    return tiles


def prepare_tiles(cfg):
    """Resolve ``prepare.aoi_*`` (and optional ``exclude_tiles``) to SW corners."""
    exclude = getattr(cfg.prepare, "exclude_tiles", None) or []
    tiles = tiles_from_bbox(cfg.prepare.aoi_lon, cfg.prepare.aoi_lat, exclude)
    print("Prepare tiles:")
    for west, south in tiles:
        lat_f, lon_f = format_degree_tile(west, south)
        print(f"  {lat_f}{lon_f}  (SW {west}, {south})")
    return tiles


def format_degree_tile(west, south):
    """Return (lat_f, lon_f) for a 1x1 degree tile named by its top-left corner."""
    west = int(west)
    south = int(south)
    if south >= 0:
        lat_f = "N{:02d}".format(south + 1)
    else:
        lat_f = "S{:02d}".format(abs(south + 1))
    if west >= 0:
        lon_f = "E{:03d}".format(west)
    else:
        lon_f = "W{:03d}".format(abs(west))
    return lat_f, lon_f
