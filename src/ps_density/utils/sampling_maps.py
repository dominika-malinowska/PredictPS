"""Map of train, validation, and test chips for the Netherlands dataset."""

from __future__ import annotations

import os
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import seed_everything
from matplotlib.figure import Figure
from shapely.geometry import Polygon
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.datasets.splits import random_bbox_splitting
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.img_tiles as cimgt
except ImportError:  # pragma: no cover - cartopy is a project dependency
    ccrs = None
    cfeature = None
    cimgt = None


def _box_polygon(box) -> Polygon:
    """Convert a TorchGeo bounding box to a shapely polygon.

    Args:
        box: Object with ``minx``/``maxx``/``miny``/``maxy``, or a 4-tuple.

    Returns:
        Polygon in the box CRS.
    """
    if hasattr(box, "minx"):
        minx, maxx, miny, maxy = box.minx, box.maxx, box.miny, box.maxy
    else:
        minx, maxx, miny, maxy = box[0], box[1], box[2], box[3]
    return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])


def _split_region_polygons(split_dataset) -> list:
    """Polygons of ``random_bbox_splitting`` regions in a GeoDataset index.

    Duplicate spatial boxes (common on intersection datasets) are dropped.

    Args:
        split_dataset: One of the datasets returned by ``random_bbox_splitting``.

    Returns:
        Unique split-region polygons in the dataset CRS.
    """
    polygons = []
    seen = set()
    for hit in split_dataset.index.intersection(
        split_dataset.index.bounds, objects=True
    ):
        minx, maxx, miny, maxy = hit.bounds[:4]
        key = (round(minx, 8), round(maxx, 8), round(miny, 8), round(maxy, 8))
        if key in seen:
            continue
        seen.add(key)
        polygons.append(_box_polygon((minx, maxx, miny, maxy)))
    return polygons


def _take_batch(dataloader: DataLoader, batch_index: int):
    """Return one batch, or raise if the loader is shorter than requested."""
    last_index = -1
    for index, batch in enumerate(dataloader):
        last_index = index
        if index == batch_index:
            return batch
    raise IndexError(
        "Requested batch {0}, but the dataloader only yielded {1} batch(es). "
        "Lower the batch index.".format(batch_index, last_index + 1)
    )


def plot_sampling_map(
    dataset,
    seed: int = 54321,
    patch_size: int = 64,
    batch_size: int = 16,
    train_batch: int = 50,
    val_batch: int = 5,
    test_batch: int = 20,
    plot_split_regions: bool = False,
    save_path: Optional[str] = None,
) -> Figure:
    """Draw Netherlands tiles and one batch of train, val, and test chips.

    The split is ``random_bbox_splitting`` with fractions 0.7 / 0.15 / 0.15.
    Training chips come from ``RandomGeoSampler``. Validation and test chips
    come from ``GridGeoSampler`` with stride ``patch_size - 8``.

    Chip colours: train blue, validation red, test green. With
    ``plot_split_regions=True``, the same colours at lower opacity show the
    full split bounding boxes stored in each GeoDataset index.

    Args:
        dataset: TorchGeo intersection dataset used for training.
        seed: Split and sampler seed.
        patch_size: Chip size in pixels.
        batch_size: Chips drawn from the selected batch.
        train_batch: Train-batch index to draw.
        val_batch: Validation-batch index to draw.
        test_batch: Test-batch index to draw.
        plot_split_regions: If True, overlay train/val/test index bounds.
        save_path: Optional path for a JPEG (for example ``fig2.jpg``).

    Returns:
        Matplotlib figure.
    """
    if ccrs is None:
        raise ImportError("cartopy is required to draw the sampling map")

    seed_everything(seed, workers=True)
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_bbox_splitting(
        dataset, [0.7, 0.15, 0.15], generator=generator
    )

    train_sampler = RandomGeoSampler(
        train_dataset, patch_size, (train_batch + 1) * batch_size
    )
    val_sampler = GridGeoSampler(val_dataset, patch_size, patch_size - 8)
    test_sampler = GridGeoSampler(test_dataset, patch_size, patch_size - 8)

    def _loader(split, sampler):
        return DataLoader(
            dataset=split,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=stack_samples,
            generator=generator,
        )

    batches = {
        "train": (_take_batch(_loader(train_dataset, train_sampler), train_batch), "blue"),
        "val": (_take_batch(_loader(val_dataset, val_sampler), val_batch), "red"),
        "test": (_take_batch(_loader(test_dataset, test_sampler), test_batch), "green"),
    }

    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.add_feature(cfeature.BORDERS, color="gray")
    ax.add_feature(cfeature.COASTLINE, color="gray")
    ax.set_extent([2.5, 8.5, 49.5, 54.5])
    try:
        ax.add_image(cimgt.OSM(), 7, interpolation="spline36")
    except Exception as error:
        print("OSM basemap was not added: {0}".format(error))

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}

    tile_polygons = []
    for lon in range(3, 8):
        for lat in range(50, 54):
            tile_polygons.append(
                Polygon(
                    [
                        (lon, lat),
                        (lon + 1, lat),
                        (lon + 1, lat + 1),
                        (lon, lat + 1),
                    ]
                )
            )
    gpd.GeoDataFrame({"geometry": tile_polygons}).plot(
        ax=ax, color="none", edgecolor="black"
    )

    if plot_split_regions:
        for split, color in (
            (train_dataset, "blue"),
            (val_dataset, "red"),
            (test_dataset, "green"),
        ):
            polygons = _split_region_polygons(split)
            if polygons:
                gpd.GeoDataFrame({"geometry": polygons}).plot(
                    ax=ax, color=color, edgecolor="none", alpha=0.2
                )

    for _name, (batch, color) in batches.items():
        polygons = [_box_polygon(box) for box in batch["bbox"]]
        gpd.GeoDataFrame({"geometry": polygons}).plot(
            ax=ax, color=color, edgecolor=color, alpha=0.5
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")

    return fig
