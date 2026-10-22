"""Rasterize OSM infrastructure for each 1x1 degree tile and mosaic for training."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_image
from rasterio.merge import merge
from shapely.geometry import box, mapping

from ps_density.paths import format_degree_tile, load_paths, prepare_tiles
from ps_density.preprocessing_OSM_data import (
    BuildRoadRailHandler,
    convert_osm_osmconvert,
    map_value,
)


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


def rasterize_tile(o5m_path, west, south, east, north, output_path):
    handler = BuildRoadRailHandler()
    handler.apply_file(str(o5m_path), locations=True, idx="flex_mem")

    df_buil = pd.DataFrame(handler.buildings)
    df_road = pd.DataFrame(handler.roads)
    df_rail = pd.DataFrame(handler.railways)
    print(len(df_buil), len(df_road), len(df_rail), flush=True)

    frames = []
    if not df_rail.empty:
        df_rail["type"] = "rail"
        frames.append(df_rail)
    if not df_road.empty:
        df_road.rename(columns={"highway": "type"}, inplace=True)
        frames.append(df_road)
    if not df_buil.empty:
        df_buil["type"] = "building"
        frames.append(df_buil)

    if not frames:
        print(f"No OSM features in {o5m_path}, skipping raster")
        return False

    df_all = pd.concat(frames)
    df_all["value"] = df_all.apply(map_value, axis=1)
    df_all.sort_values(by=["value"], inplace=True)
    gdf_osm = gpd.GeoDataFrame(df_all)

    cube = make_geocube(
        gdf_osm,
        measurements=["value"],
        resolution=(-1 / 1200, 1 / 1200),
        geom=mapping(box(west, south, east, north)),
        fill=0,
        rasterize_function=partial(rasterize_image, all_touched=True),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cube.value.rio.to_raster(output_path)
    return True


def osm_pbf_inputs(osm_pbf):
    """Return every .osm.pbf to clip. A directory contributes all of its extracts."""
    osm_path = Path(osm_pbf)
    if osm_path.is_dir():
        pbfs = sorted(osm_path.glob("*.osm.pbf"))
        if not pbfs:
            raise FileNotFoundError(f"No .osm.pbf files in {osm_path}")
        return pbfs
    if osm_path.is_file():
        return [osm_path]
    raise FileNotFoundError(f"OSM PBF not found: {osm_path}")


def main():
    cfg = load_paths()
    osm_pbfs = osm_pbf_inputs(cfg.prepare.osm_pbf)
    print("OSM extracts:")
    for path in osm_pbfs:
        print(f"  {path}")

    mosaic_path = Path(cfg.train.input.osm)
    work_dir = mosaic_path.parent / "tiles"
    work_dir.mkdir(parents=True, exist_ok=True)

    osmconvert_bin = cfg.predict.osmconvert
    tile_paths = []

    for west, south in prepare_tiles(cfg):
        west, south = int(west), int(south)
        east, north = west + 1, south + 1
        lat_f, lon_f = format_degree_tile(west, south)
        tif_path = work_dir / f"{lat_f}{lon_f}_infrs_map.tif"
        if tif_path.exists():
            print(f"Infrastructure map {lat_f}{lon_f} already generated.")
            tile_paths.append(tif_path)
            continue

        o5m_path = work_dir / f"clipped_{lat_f}_{lon_f}.o5m"
        convert_osm_osmconvert(
            osm_pbfs,
            o5m_path,
            osmconvert_bin=osmconvert_bin,
            bbox=(west, south, east, north),
        )
        if not o5m_path.exists():
            print(f"Conversion failed for {lat_f}{lon_f}")
            continue

        if rasterize_tile(o5m_path, west, south, east, north, tif_path):
            tile_paths.append(tif_path)

    if not tile_paths:
        raise RuntimeError("No infrastructure tiles were written")

    print(f"Mosaicking {len(tile_paths)} tiles into {mosaic_path}")
    mosaic_tifs(tile_paths, mosaic_path)
    print("OSM preparation done")


if __name__ == "__main__":
    main()
