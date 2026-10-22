"""Load ``config/paths.yaml`` with optional environment overrides.

The YAML file is interpolated with ``{data_root}``. ``PREDICTPS_DATA_ROOT``
overrides ``data_root`` after the file is read. ``PREDICTPS_PATHS_FILE``
selects a different YAML file.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PATHS_FILE = _REPO_ROOT / "config" / "paths.yaml"


class Paths(SimpleNamespace):
    """Attribute access for nested path mappings."""


def _to_namespace(value):
    """Recursively convert mappings to ``Paths`` namespaces.

    Args:
        value: Nested dict, list, or scalar from the YAML file.

    Returns:
        Attribute-accessible path tree, or ``value`` unchanged.
    """
    if isinstance(value, dict):
        return Paths(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _interpolate(value, mapping):
    """Replace ``{data_root}`` (and other mapping keys) in string paths.

    Args:
        value: Nested dict, list, or string.
        mapping: Format mapping, typically ``{"data_root": ...}``.

    Returns:
        Structure with interpolated strings.
    """
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

    Args:
        paths_file: Optional path to a YAML file. Defaults to the env var or
            ``config/paths.yaml`` in the repository.

    Returns:
        Nested ``Paths`` object with interpolated string paths.
    """
    file_path = Path(
        paths_file or os.environ.get("PREDICTPS_PATHS_FILE") or _DEFAULT_PATHS_FILE
    )
    with file_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data_root = str(
        os.environ.get("PREDICTPS_DATA_ROOT", raw.get("data_root", "")) or ""
    ).rstrip("/\\")
    raw["data_root"] = data_root
    interpolated = _interpolate(raw, {"data_root": data_root})
    return _to_namespace(interpolated)


def tiles_from_bbox(aoi_lon, aoi_lat, exclude_tiles=None):
    """Expand a lon/lat box to 1x1 degree south-west corners.

    Tiles that intersect ``[min_lon, max_lon] x [min_lat, max_lat]`` are kept.
    ``exclude_tiles`` is a list of ``[lon, lat]`` south-west corners to drop.

    Args:
        aoi_lon: Two longitudes that bound the area of interest.
        aoi_lat: Two latitudes that bound the area of interest.
        exclude_tiles: Optional south-west corners to skip.

    Returns:
        List of ``(west, south)`` integer corners.
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
    """Resolve ``prepare.aoi_*`` (and optional ``exclude_tiles``) to SW corners.

    Args:
        cfg: Loaded paths object with a ``prepare`` block.

    Returns:
        List of ``(west, south)`` integer corners.
    """
    exclude = getattr(cfg.prepare, "exclude_tiles", None) or []
    tiles = tiles_from_bbox(cfg.prepare.aoi_lon, cfg.prepare.aoi_lat, exclude)
    print("Prepare tiles:")
    for west, south in tiles:
        lat_f, lon_f = format_degree_tile(west, south)
        print(f"  {lat_f}{lon_f}  (SW {west}, {south})")
    return tiles


def format_degree_tile(west, south):
    """Return ``(lat_f, lon_f)`` for a 1x1 degree tile named by its top-left corner.

    Args:
        west: West longitude of the tile (integer degrees).
        south: South latitude of the tile (integer degrees).

    Returns:
        Tuple of latitude and longitude labels, for example ``("N51", "E018")``.
    """
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
