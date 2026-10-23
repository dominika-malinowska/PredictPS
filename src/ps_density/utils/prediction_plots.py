"""Georeferenced prediction chips, mosaic merge, and review plots."""

# Import modules
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors

import torch
from torch import nn

import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from rasterio.merge import merge

# Set global font parameters for the plots
font_params = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
    "font.size": 12,
}
# Apply the global font parameters
plt.rcParams.update(font_params)


def _model_predict(model, image):
    """Run ``model`` on its parameter device and return a CPU tensor.

    Args:
        model: Lightning or torch module.
        image: Input tensor on CPU or GPU.

    Returns:
        Prediction tensor on CPU.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        prediction = model(image.to(device))
    return prediction.detach().cpu()


# Define functions for generating geotfiff of predictions
def create_in_memory_geochip(predicted_chip, geotransform, crs):
    # Source:
    # https://www.kaggle.com/code/luizclaudioandrade/torchgeo-101#Making-predictions,-applying-the-correct-georrefence-to-them-and-merging-them-in-one-file-that-is-saved-to-disk.
    """
    Apply georeferencing to the predicted chip.

    Args:
        predicted_chip (numpy array): The predicted segmentation chip (e.g., binary mask).
        geotransform (tuple): A tuple containing the geotransformation information of the chip (x-coordinate of the top left corner, x and y pixel size, rotation, y-coordinate of the top left corner, and rotation).
        crs (str): Spatial Reference System (e.g., EPSG code) of the chip.

    Return:
        A rasterio dataset that is georreferenced.
    """

    memfile = MemoryFile()
    dataset = memfile.open(
        driver="GTiff",
        height=predicted_chip.shape[2],
        width=predicted_chip.shape[3],
        count=predicted_chip.shape[1],  # Number of bands
        dtype=np.uint8,
        crs=crs,
        transform=geotransform,
        nodata=255,
    )

    dataset.write(predicted_chip[0].detach().numpy())
    return dataset


def georreferenced_chip_generator(dataloader, model, crs, pixel_size):
    # Source:
    # https://www.kaggle.com/code/luizclaudioandrade/torchgeo-101#Making-predictions,-applying-the-correct-georrefence-to-them-and-merging-them-in-one-file-that-is-saved-to-disk.
    """
    Apply georeferencing to the predicted chip.

    Args:
        dataloader (torch.utils.data.Dataloader): Dataloader with the data to be predicted.
        model (an https://github.com/qubvel/segmentation_models.pytorch model): model used
        for inference.
        crs (str): Spatial Reference System (e.g., EPSG code) of the chip.
        pixel_size (float): Pixel dimensoion in map units.

    Returns:
        A list of georeferenced numpy arrays of the predicted outputs.
    """
    georref_chips_list = []

    for sample in dataloader:
        for j in range(len(sample["bbox"])):
            image, bbox = sample["image"][j], sample["bbox"][j]

            # Image dimension should be [batch, channels, height, width]
            # Because we are predicting one image at a time, we need to add a batch dimension
            if len(image.size()) == 3:
                image = image.unsqueeze(0)
            prediction = _model_predict(model, image)

            geotransform = from_origin(bbox.minx, bbox.maxy, pixel_size, pixel_size)

            georref_chips_list.append(
                create_in_memory_geochip(prediction, geotransform, crs)
            )

    return georref_chips_list


def custom_avg(merged_data, new_data, merged_mask, new_mask, overlap, **kwargs):
    # https://gis.stackexchange.com/questions/469972/treat-overlapping-pixel-when-doing-merge-geotiff-images-using-rasterio
    # https://github.com/rasterio/rasterio/blob/main/rasterio/merge.py#L84

    """
    The function merges two sets of data arrays based on specified masks and overlap,
    updating the merged data with valid values from the new data while considering the overlap
    between them. It is used to blend the boundaries of overlapping chips during the merging process.

    Args:
        merged_data (array): variable to be updated with new data. It should have the same shape as
            the new_data parameter.
        new_data (array): data to merge same shape as merged_data
        merged_mask (array): a boolean mask that indicates which pixels in the merged_data array are invalid.
            It has the same shape as the merged_data array.
        new_mask (array): a boolean mask that indicates which pixels in the new_data array are
            considered invalid. It has the same shape as the merged_data array and is used to
            identify areas where the data in new_data should not be merged with the existing data
        overlap (int): the number of overlapping pixels between the merged_data and the new_data.
            It is used to calculate the weights for blending the two datasets during
            the merging process.

    Returns:
        None
    """

    # Creates an array with the same shape as merged_mask and True/False values randomly assigned
    mask = np.empty_like(merged_mask, dtype="bool")
    # Edit mask so that it is true if either merged_mask or new_mask is true
    # Mask will be True where the data is invalid
    np.logical_or(merged_mask, new_mask, out=mask)
    # Change mask so that True becomes False and False becomes True
    # Thus, mask is true where data is valid
    np.logical_not(mask, out=mask)

    # Create a weights array with ones
    weights = np.ones(mask.shape)

    for i in range(overlap - 1, 0 - 1, -1):
        weights[:, -i - 1, :] = (i + 1) / overlap
        weights[:, :, i] = (i + 1) / overlap

    merged_data[mask] = np.rint(
        (1 - weights[mask]) * merged_data[mask] + weights[mask] * new_data.data[mask]
    ).astype(int)

    # Now mask will be a opposite to the new_mask i.e. True where new_mask is False and vice versa
    # Therefore, mask now shows where the new_data is valid and should be copied to merged_data
    np.logical_not(new_mask, out=mask)
    # Update the mask so that it is True if both merged_mask and mask are True
    # i.e. mereged_mask is True (invalid data) and new_mask is False (valid data)
    # This is done to ensure that the new_data is only copied to merged_data where it is valid and merged_data is invalid
    np.logical_and(merged_mask, mask, out=mask)
    # Copy the new_data to merged_data where mask is True, i.e. where new_data is valid and merged_data is invalid
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


def merge_georeferenced_chips(chips_list, output_path, overlap, bounds):
    # Source:
    # https://www.kaggle.com/code/luizclaudioandrade/torchgeo-101#Making-predictions,-applying-the-correct-georrefence-to-them-and-merging-them-in-one-file-that-is-saved-to-disk.
    """
    Merge a list of georeferenced chips into a single GeoTIFF file.

    Args:
        chips_list (list): A list of Rasterio datasets representing the georeferenced chips.
        output_path (str): The path where the merged GeoTIFF file will be saved.
        overlap (int): The number of overlapping pixels between the merged chips.
        bounds (tuple): A tuple containing the spatial extent of the area of interest (minx, miny, maxx, maxy).

    Returns:
        None
    """

    # Create a wrapper function that includes the overlap parameter
    def custom_avg_with_overlap(merged_data, new_data, merged_mask, new_mask, **kwargs):
        return custom_avg(merged_data, new_data, merged_mask, new_mask, overlap)

    # Merge the chips using Rasterio's merge function
    merged, merged_transform = merge(
        chips_list, method=custom_avg_with_overlap, nodata=255, bounds=bounds
    )

    # Calculate the number of rows and columns for the merged output
    rows, cols = merged.shape[1], merged.shape[2]

    # Update the metadata of the merged dataset
    # The metadata is copied from the first chip in the list, and then the height, width, and transform are updated
    merged_metadata = chips_list[0].meta
    merged_metadata.update(
        {"height": rows, "width": cols, "transform": merged_transform}
    )

    # Write the merged array to a new GeoTIFF file
    # The output file is opened in write mode ('w'), and the updated metadata is passed in
    with rasterio.open(output_path, "w", **merged_metadata) as dst:
        dst.write(merged)

    # Close all the chips in the list
    # This is important to free up system resources
    for chip in chips_list:
        chip.close()


# Define cmap IBM (color blind friendly) for plotting infrastructure
cmap_ibm = matplotlib.colors.ListedColormap(
    ["#ffffff", "#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000"]
)
cmap_ranges = [0, 0.54, 0.58, 0.72, 0.85, 0.87, 1.0]
norm = matplotlib.colors.BoundaryNorm(cmap_ranges, cmap_ibm.N)


colors = [
    "#d73027",
    "#f46d43",
    "#fdae61",
    "#fee090",
    "#ffffbf",
    "#e0f3f8",
    "#abd9e9",
    "#74add1",
    "#4575b4",
][::-1]
cmap_cb = matplotlib.colors.LinearSegmentedColormap.from_list("custom_cmap", colors)


# Define helper function to set axis of the plot of predictions
def set_axis_ticks(ax, bbox, max_ticks, plot_axis):
    """
    The function sets the ticks and labels of the x and y axes of a plot based on the bounding box of the patch.

    Args:
        ax (matplotlib axis): The axis of the plot.
        bbox (rasterio bounding box): The bounding box of the patch.
        max_ticks (int): The maximum number of ticks on the x and y axes.
        plot_axis (bool): A boolean indicating whether the axis should be plotted.

    Returns:
        None
    """
    if plot_axis:
        ax.set_xticks([0, max_ticks])
        ax.set_xticklabels(
            [
                f"{bbox.minx:.2f}",
                f"{bbox.maxx:.2f}",
            ],
            fontsize=14,
        )
        ax.set_yticks([0, max_ticks])
        ax.set_yticklabels(
            [
                f"{bbox.maxy:.2f}",
                f"{bbox.miny:.2f}",
            ],
            fontsize=14,
        )
    else:
        ax.axis("off")


def create_colorbar(
    image,
    position,
    label,
    labelpad,
    ticks=None,
    tick_labels=None,
    second_label=None,
    second_ticks=None,
    second_tick_labels=None,
):
    """
    The function creates a colorbar for a plot.

    Args:
        image (matplotlib image): The image to which the colorbar corresponds.
        position (list): The position of the colorbar on the plot.
        label (str): The label of the colorbar.
        labelpad (int): The padding of the label.
        ticks (list): The ticks of the colorbar.
        tick_labels (list): The labels of the ticks.
        second_label (str): The label of the second axis of the colorbar.
        second_ticks (list): The ticks of the second axis of the colorbar.
        second_tick_labels (list): The labels of the ticks of the second axis of the colorbar.

    Returns:
        cbar (matplotlib colorbar): The colorbar.
    """
    fig = plt.gcf()
    cbar_ax = fig.add_axes(position)
    cbar = plt.colorbar(image, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(label, fontsize=18, labelpad=labelpad)
    if ticks is not None:
        cbar.set_ticks(ticks)
    if tick_labels is not None:
        cbar.set_ticklabels(tick_labels)
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=90, ha="center")
    cbar.ax.tick_params(labelsize=16)  # Set tick font size

    # Add a second axis if a second label is provided
    if second_label is not None:
        cbar_ax2 = cbar_ax.twiny()
        cbar_ax2.set_xlabel(second_label, fontsize=18, labelpad=-labelpad + 10)
        if second_ticks is not None:
            cbar_ax2.set_xticks(second_ticks)
        if second_tick_labels is not None:
            cbar_ax2.set_xticklabels(second_tick_labels)
        # Make sure the second axis does not cover the first axis
        cbar_ax2.set_frame_on(False)
        cbar_ax2.tick_params(labelsize=16)  # Set tick font size

    return cbar


# Define functions for plotting predictions
def plot_input_pred_patch(
    dataloader,
    model,
    opt,
    batch_size,
    plot_axis,
    plot_bar,
    plot_titles,
    outputdir_plots,
    specific_batch=None,
    specific_patch=None,
    box_coords=None,
    paper_fig=None,
):
    """
    The function generates plots of the input, ground truth, and predicted patches for each patch in a batch
    and in addition to saving them it returns a DataFrame containing the Mean Absolute Error (MAE) values.

    Args:
        dataloader (torch.utils.data.Dataloader): The dataloader with the data to be predicted.
        model (torch.nn.Module): The model used for inference.
        opt (dict): The dictionary containing the optical image data.
        batch_size (int): The batch size.
        plot_axis (bool): A boolean indicating whether the axis should be plotted.
        plot_bar (bool): A boolean indicating whether the colorbar should be plotted.
        plot_titles (bool): A boolean indicating whether the titles of the plots should be plotted.
        outputdir_plots (str): The directory where the plots will be saved.
        specific_batch (int): The specific batch to plot.
        specific_patch (int): The specific patch to plot.
        box_coords (list): The coordinates of the box to plot.
        paper_fig (str): Optional paper filename stem (for example ``fig6a``).
            The JPEG is always saved as ``model_pred_b{batch}_{i}.jpg``; when
            this is set, a second copy is written as ``{paper_fig}.jpg``.

    Returns:
        df_mae_test (pandas DataFrame): The DataFrame containing the Mean Absolute Error (MAE) values.


    """

    # Plot model input and predictions so that each patch in a batch is on a separate plot
    # For each patch calculate MAE and save it to a csv

    MAE_calculation = nn.L1Loss()

    # Specify number of rows and columns
    rows = 1
    columns = 5
    width = 5

    # disable randomness, dropout, etc...
    model.eval()

    # Initialise variable for storing the mae values
    data = []

    # Row and column identifier
    r = 0
    c = 0

    # Loop through the dataloader and create plot for each patch in the batch
    for b, batch in enumerate(dataloader):
        # If a specific batch is provided and this is not it, skip this iteration
        if specific_batch is not None and b != specific_batch:
            continue

        img = batch["image"]
        msk = batch["mask"]
        pred = _model_predict(model, img)

        # Iterate over the batch
        for i in range(batch_size):
            # If a specific patch is provided and this is not it, skip this iteration
            if specific_patch is not None and i != specific_patch:
                continue

            _, axs = plt.subplots(
                rows, columns, figsize=(columns * width, rows * width)
            )

            # OPTICAL IMAGE
            imgs_opt_temp = opt[batch["bbox"][i]]["image"]
            opt_R = imgs_opt_temp[0]
            opt_G = imgs_opt_temp[1]
            opt_B = imgs_opt_temp[2]

            opt_R_normalized = opt_R / 255.0
            opt_G_normalized = opt_G / 255.0
            opt_B_normalized = opt_B / 255.0

            # Stack the three tensors along the third axis to create an RGB image tensor
            rgb_image = np.stack(
                (opt_R_normalized, opt_G_normalized, opt_B_normalized), axis=-1
            )
            axs[c].imshow(rgb_image)

            set_axis_ticks(axs[c], batch["bbox"][i], 512, plot_axis)

            # INPUT CHANNEL 1
            im1 = axs[c + 1].imshow(
                img[i][0].squeeze().numpy(), cmap=cmap_cb, vmin=0, vmax=1
            )

            set_axis_ticks(axs[c + 1], batch["bbox"][i], 64, plot_axis)

            if plot_bar:
                create_colorbar(
                    im1, [0.3, -0.01, 0.1, 0.03], "Long-term coherence", -55
                )

            # INPUT CHANNEL 2
            try:
                im2 = axs[c + 2].imshow(
                    img[i][1].squeeze().numpy(), cmap=cmap_ibm, norm=norm
                )
            except IndexError:
                # if plotting output of the model with only one input channel, leave this blank
                im2 = axs[c + 2].imshow(np.ones((64, 64)), cmap=cmap_ibm, norm=norm)

            set_axis_ticks(axs[c + 2], batch["bbox"][i], 64, plot_axis)

            if plot_bar:
                create_colorbar(
                    im2,
                    [0.46, -0.01, 0.1, 0.03],
                    "OSM category",
                    -140,
                    [0.27, 0.56, 0.65, 0.795, 0.86, 0.935],
                    [
                        "None",
                        "Trunk roads",
                        "Primary and \n secondary roads",
                        "Motorways",
                        "Railways",
                        "Buildings",
                    ],
                )

            # GROUND TRUTH
            im3 = axs[c + 3].imshow(
                msk[i].squeeze().detach().numpy(), cmap=cmap_cb, vmin=0, vmax=40
            )

            set_axis_ticks(axs[c + 3], batch["bbox"][i], 64, plot_axis)

            if plot_bar:
                # create_colorbar(im3, [0.62, -0.01, 0.1, 0.03],  '# of PSs in km$^2$', 0,
                #                 ticks=[0,20,40,60,80], tick_labels=['0', '3,900', '7,800','11,700', '\u226515,600'],
                #                 second_label="# of PSs in a pixel",
                #                 second_ticks=[0, 20, 40, 60, 80],
                #                 second_tick_labels=['0','20','40','60','\u226580'] )

                create_colorbar(
                    im3,
                    [0.62, -0.01, 0.1, 0.03],
                    "# of PSs in km$^2$",
                    0,
                    ticks=[0, 20, 40],
                    tick_labels=["0", "3,900", "\u22657,800"],
                    second_label="# of PSs in a pixel",
                    second_ticks=[0, 20, 40],
                    second_tick_labels=["0", "20", "\u226540"],
                )
                # create_colorbar(im3, [0.62, -0.01, 0.1, 0.03], "Nb of points per pixel", -50)

            # PREDICTION
            im4 = axs[c + 4].imshow(
                pred[i].squeeze().detach().numpy(), cmap=cmap_cb, vmin=0, vmax=40
            )

            set_axis_ticks(axs[c + 4], batch["bbox"][i], 64, plot_axis)

            if plot_bar:
                # create_colorbar(im4, [0.78, -0.01, 0.1, 0.03], "# of PSs in a pixel", -50)

                # create_colorbar(im4, [0.78, -0.01, 0.1, 0.03],  '# of PSs in km$^2$', 0,
                #     ticks=[0,20,40,60,80], tick_labels=['0', '3,900', '7,800','11,700', '\u226515,600'],
                #     second_label="# of PSs in a pixel",
                #     second_ticks=[0, 20, 40, 60, 80],
                #     second_tick_labels=['0','20','40','60','\u226580'] )

                create_colorbar(
                    im4,
                    [0.78, -0.01, 0.1, 0.03],
                    "# of PSs in km$^2$",
                    0,
                    ticks=[0, 20, 40],
                    tick_labels=["0", "3,900", "\u22657,800"],
                    second_label="# of PSs in a pixel",
                    second_ticks=[0, 20, 40],
                    second_tick_labels=["0", "20", "\u226540"],
                )

            # Draw the box if box_coords is not None
            if box_coords is not None:
                for plot_id in list(range(5)):
                    if plot_id == 0:
                        rect = plt.Rectangle(
                            (box_coords[0] * 8, box_coords[1] * 8),
                            box_coords[2] * 8,
                            box_coords[3] * 8,
                            linewidth=1,
                            edgecolor="r",
                            facecolor="none",
                        )
                    else:
                        rect = plt.Rectangle(
                            (box_coords[0], box_coords[1]),
                            box_coords[2],
                            box_coords[3],
                            linewidth=1,
                            edgecolor="r",
                            facecolor="none",
                        )
                    axs[c + plot_id].add_patch(rect)

            # PLOT TITLES
            if plot_titles:
                axs[0].set_title("Google Satellite", fontsize=22)
                axs[1].set_title("Long-term coherence", fontsize=22)
                axs[2].set_title("Infrastructure map", fontsize=22)
                axs[3].set_title("PS density", fontsize=22)
                axs[4].set_title("Prediction", fontsize=22)

            # SAVE FIG
            plt.savefig(
                os.path.join(outputdir_plots, f"model_pred_b{b}_{i}.jpg"),
                dpi=600,
                bbox_inches="tight",
            )
            if paper_fig:
                plt.savefig(
                    os.path.join(outputdir_plots, f"{paper_fig}.jpg"),
                    dpi=600,
                    bbox_inches="tight",
                )

            plt.close()

            # CALCULATE MAE
            mae_value = MAE_calculation(msk[i], pred[i]).item()

            entry = {
                "minx": batch["bbox"][i].minx,
                "miny": batch["bbox"][i].miny,
                "batch": b,
                "i": i,
                "mae": mae_value,
            }

            # Append the dictionary to the list
            data.append(entry)

    df_mae_test = pd.DataFrame(data)
    return df_mae_test
