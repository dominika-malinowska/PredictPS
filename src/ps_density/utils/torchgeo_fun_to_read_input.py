"""TorchGeo raster dataset subclasses and coherence scaling for model input."""

import matplotlib.pyplot as plt

import torch

from torchgeo.datasets import RasterDataset


class RasterDataset_imgs(RasterDataset):
    """GeoTIFF image stack (coherence or OSM) indexed by TorchGeo."""

    filename_glob = "*.tif"
    is_image = True


class RasterDataset_msks(RasterDataset):
    """GeoTIFF mask stack (EGMS PS density) indexed by TorchGeo."""

    filename_glob = "*.tif"
    is_image = False


class RasterDatasetOptical_imgs(RasterDataset):
    """RGB GeoTIFF used as map context in prediction review plots."""

    filename_glob = "*.tif"
    is_image = True
    separate_files = False
    all_bands = ["R", "G", "B", "NIR"]
    rgb_bands = ["R", "G", "B"]

    def plot(self, sample):
        """Plot RGB bands from an optical sample.

        Args:
            sample: TorchGeo sample with an ``image`` tensor.

        Returns:
            Matplotlib figure.
        """

        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image
        image = sample["image"][rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image / 1, min=0, max=1).numpy()

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig


def scale_coh(item: dict):
    """Scale stored coherence integers to [0, 1] floats.

    Args:
        item: Sample dict whose ``image`` tensor is coherence times 1000.

    Returns:
        The same dict with a scaled ``image`` tensor.
    """
    item["image"] = item["image"] / 1000.0
    return item


def plot_input(dataset: dict, rows: int = 1, width: int = 10):
    """Plot input channels and the PS-density mask side by side.

    Args:
        dataset: Sample dict with ``image`` and ``mask`` tensors.
        rows: Number of subplot rows when more than one row is needed.
        width: Width in inches of each subplot.
    """
    img = dataset["image"]
    mask = dataset["mask"]

    nb_input_chnls = img.shape[0]

    if rows == 1:
        # create a grid
        _, axs = plt.subplots(
            1, nb_input_chnls + 1, figsize=(nb_input_chnls + 1 * width, 1 * width)
        )

        for i in range(nb_input_chnls):
            axs[i].imshow(img[i].squeeze().numpy(), cmap="turbo", vmin=0, vmax=1)
            axs[i].axis("off")

        axs[nb_input_chnls].imshow(
            mask.squeeze().numpy(), cmap="turbo", vmin=0, vmax=60
        )
        axs[nb_input_chnls].axis("off")
    else:
        columns = int((nb_input_chnls + 1) / rows + (nb_input_chnls + 1 % rows > 0))
        _, axs = plt.subplots(rows, columns, figsize=(columns * width, rows * width))

        r = 0
        c = 0

        for i in range(nb_input_chnls):
            axs[r][c].imshow(img[i].squeeze().numpy(), cmap="turbo", vmin=0, vmax=1)
            axs[r][c].axis("off")

            if c < (columns - 1):
                c = c + 1
            else:
                c = 0
                r = r + 1

        axs[r][c].imshow(mask.squeeze().numpy(), cmap="turbo", vmin=0)
        axs[r][c].axis("off")


