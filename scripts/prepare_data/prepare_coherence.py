"""Download 1x1 degree summer VV rho tiles and mosaic them for training."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import rasterio
from rasterio.merge import merge

from ps_density.paths import format_degree_tile, load_paths, prepare_tiles

S3_PREFIX = "s3://sentinel-1-global-coherence-earthbigdata/data/tiles"


def mosaic_tifs(tile_paths, output_path):
    sources = [rasterio.open(path) for path in tile_paths]
    try:
        mosaic, transform = merge(sources)
        profile = sources[0].profile.copy()
        profile.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "compress": "deflate",
                "tiled": True,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dest:
            dest.write(mosaic)
    finally:
        for source in sources:
            source.close()


def download_tile(lat_f, lon_f, dest_path, season, polarisation, metric):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"Skipping existing {dest_path.name}")
        return dest_path

    remote = (
        f"{S3_PREFIX}/{lat_f}{lon_f}/{lat_f}{lon_f}_{season}_{polarisation}_{metric}.tif"
    )
    print(f"Downloading {remote}")
    result = subprocess.run(
        ["aws", "s3", "cp", remote, str(dest_path), "--no-sign-request"],
        check=False,
    )
    if result.returncode != 0 or not dest_path.exists():
        raise RuntimeError(f"Download failed for {lat_f}{lon_f}")
    return dest_path


def main(season="summer", polarisation="vv", metric="rho"):
    cfg = load_paths()
    output_dir = Path(cfg.train.input.coherence_summer_dir)
    tiles_dir = output_dir / "tiles"
    mosaic_name = f"{season}_{polarisation}_{metric}.tif"

    tile_paths = []
    for west, south in prepare_tiles(cfg):
        lat_f, lon_f = format_degree_tile(west, south)
        dest = tiles_dir / f"{lat_f}{lon_f}_{season}_{polarisation}_{metric}.tif"
        tile_paths.append(
            download_tile(lat_f, lon_f, dest, season, polarisation, metric)
        )

    mosaic_path = output_dir / mosaic_name
    print(f"Mosaicking {len(tile_paths)} tiles into {mosaic_path}")
    mosaic_tifs(tile_paths, mosaic_path)
    print("Coherence preparation done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="summer")
    parser.add_argument("--polarisation", default="vv")
    parser.add_argument("--metric", default="rho")
    args = parser.parse_args()
    main(season=args.season, polarisation=args.polarisation, metric=args.metric)
