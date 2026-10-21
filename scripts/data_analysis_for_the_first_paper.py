"""
The code reads data, fits linear regression models, evaluates the models,
and generates scatter plots and histograms for analysis.

It is used to generate plots for the first paper.
"""

# Import modules #
import os
import pathlib
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.axes import Axes

from PIL import Image

import seaborn as sns

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

from shapely.geometry import Polygon

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

import statsmodels.formula.api as smf

from ps_density.paths import load_paths
from ps_density.plotting_scatter_with_3rd_dim import plot_scatter_denisty


# needed to silence errors with big reading big rasters
Image.MAX_IMAGE_PIXELS = None
pd.options.mode.chained_assignment = None

# Set global font parameters for the plots
font_params = {
    "font.family": "serif",  # Choose the font family (e.g., 'serif', 'sans-serif', etc.)
    "font.serif": "Times New Roman",  # Specify the serif font
    "font.size": 12,  # Set the font size
}
# Apply the global font parameters
plt.rcParams.update(font_params)


# Function definitions #


def fetch_data(raster_file, plot_title="", null_value=-9999) -> np.array:
    # Code based on https://towardsdatascience.com/raster-data-scatter-plot-using-python-4bcdc8286769
    # fetch data cleans and ravels rasters. returns raveled raster
    """
    The function reads a raster file, cleans it from null values, and returns a raveled raster.

    Arguments:
    raster_file (str): the path to the raster file
    plot_title (str): the title of the plot
    null_value (int): the null value in the raster

    Returns:
    np.array: the raveled raster

    """

    fname = raster_file
    print(fname, 0)

    # Using PIL to import raster, and convert it to numpy array
    im = Image.open(fname)
    j = np.array(im)

    # Data cleansing, remove the null value
    #     clean = lambda x: np.nan if x == null_value else x
    def clean(x):
        return np.nan if x <= null_value else x

    cleann = np.vectorize(clean)
    z = np.ravel(j)  # ravelling to make a x or y axis
    y = cleann(z)
    retval = y[~np.isnan(y)]

    # some visualization to describe the data
    plt.figure(figsize=(7, 3))
    plt.hist(retval, color="black", bins=32)
    plt.title(plot_title)

    # return raveled data
    return retval


def load_data(root, filename) -> np.array:
    """
    This function loads the data from a raster file.
    If the data has been loaded before, it is read from a .npy file.

    Arguments:
    root (str): the root directory where the data is stored
    filename (str): the name of the raster file

    Returns:
    np.array: the data from the raster file

    """

    x_data_path = os.path.join(root, filename)
    np_arr_path = x_data_path.replace(".tif", ".npy")

    if pathlib.Path(np_arr_path).exists():
        x_data = np.load(np_arr_path)
    else:
        x_data = fetch_data(x_data_path)
        np.save(np_arr_path, x_data)

    return x_data


# Read data #
_cfg = load_paths()
root = str(_cfg.data_root)

# Read long-term coherence raster
path_ltcoh = os.path.join(root, "input", "global_coh_NL")

x_data = load_data(path_ltcoh, "summer_vv_rho.tif")
x1_data = load_data(path_ltcoh, "spring_vv_rho.tif")
x2_data = load_data(path_ltcoh, "fall_vv_rho.tif")
x3_data = load_data(path_ltcoh, "winter_vv_rho.tif")

# Read nb of points raster
y_data_path = os.path.join(root, "input", "mask", "mask_EGMS")
y_data = load_data(y_data_path, "r_nb_pts_max_all.tif")

# Read mean ndvi
z_data_path = os.path.join(root, "input", "ndvi_gee", "new")
z1_data = load_data(z_data_path, "mean_ndvi_S2_wi_nan_filled_resized.tif")
z2_data = load_data(z_data_path, "mean_ndvi_S2_su_nan_filled_resized.tif")

# Read std ndvi
z3_data = load_data(z_data_path, "std_ndvi_S2_wi_nan_filled_resized.tif")
z4_data = load_data(z_data_path, "std_ndvi_S2_su_nan_filled_resized.tif")

# Read OSM masks
OSM_mask_path = os.path.join(root, "OSM_buildings")
mask_buil_data = load_data(OSM_mask_path, "buildings.tif")
mask_rail_data = load_data(OSM_mask_path, "railway.tif")
mask_highway_data = load_data(OSM_mask_path, "highway.tif")
mask_moto_data = load_data(OSM_mask_path, "highway_motorway.tif")
mask_trunk_data = load_data(OSM_mask_path, "highway_trunk.tif")
mask_primary_data = load_data(OSM_mask_path, "highway_primary.tif")
mask_secondary_data = load_data(OSM_mask_path, "highway_secondary.tif")

# Read infrastructure map
infr_map = load_data(
    os.path.join(root, "input", "OSM_input_channel_v2"), "OSM_combined_v2.tif"
)

# Read prediction data
# paper - 1st submission
# pred_test_split = load_data(root, "merged_prediction_15122023_fillednd_clipped.tif")
# review
pred_test_split = load_data(
    root, "merged_prediction_fixed_stiching_20240607_204020.tif"
)


# Read water mask
water_mask = load_data(root, "water_land_mask.tif")

# Ground truth Rome dataset
rome_gt_path = os.path.join(root, "input", "Italy", "SNT", "msk")
rome_gt = load_data(rome_gt_path, "Rome_nb_pts_des_cropped.tif")

# Prediction Rome dataset
# paper - 1st submission
# rome_pred = load_data(root, "merged_prediction_Rome_split_54321_15122023.tif")
# review
rome_pred = load_data(root, "merged_prediction_fixed_stiching_Rome_20240610_123734.tif")


# Create dfs with input to analyse#
# Create df for Dutch data
d = {
    "coh_mean": x_data,
    "coh_mean_summer": x_data,
    "coh_mean_spring": x1_data,
    "coh_mean_fall": x2_data,
    "coh_mean_winter": x3_data,
    "nbpoints": y_data,
    "ndvi_winter": z1_data,
    "ndvi_summer": z2_data,
    "std_ndvi_winter": z3_data,
    "std_ndvi_summer": z4_data,
    "mask_building": mask_buil_data,
    "mask_railway": mask_rail_data,
    "mask_highway": mask_highway_data,
    "mask_motorway": mask_moto_data,
    "mask_trunk": mask_trunk_data,
    "mask_primary": mask_primary_data,
    "mask_secondary": mask_secondary_data,
    "infr_map": infr_map,
    "water_mask": water_mask,
    "pred_test": pred_test_split,
}
df = pd.DataFrame(data=d)

# ## test - update the osm masks to make sure that the pixels that have many osm types in
# are not counted multiple times ##
# df['mask_railway'] = np.where(df['infr_map'] == 0.86, 1, 0)
# df['mask_motorway'] = np.where(df['infr_map'] == 0.73, 1, 0)
# df['mask_trunk'] = np.where(df['infr_map'] == 0.55, 1, 0)
# df['mask_primary'] = np.where((df['infr_map'] == 0.59) & (df['mask_primary'] == 1), 1, 0)
# df['mask_secondary'] = np.where((df['infr_map'] == 0.59) & (df['mask_secondary'] == 1), 1, 0)

df["coh_mean"] = df["coh_mean"] / 1000
df["coh_mean_summer"] = df["coh_mean_summer"] / 1000
df["coh_mean_spring"] = df["coh_mean_spring"] / 1000
df["coh_mean_fall"] = df["coh_mean_fall"] / 1000
df["coh_mean_winter"] = df["coh_mean_winter"] / 1000

df["nbpoints"] = df["nbpoints"].astype("int")

# clear variables no longer needed
del (
    d,
    x_data,
    x1_data,
    x2_data,
    x3_data,
    y_data,
    z1_data,
    z2_data,
    z3_data,
    z4_data,
    mask_buil_data,
    mask_rail_data,
    mask_highway_data,
    mask_moto_data,
    mask_trunk_data,
    mask_primary_data,
    mask_secondary_data,
    infr_map,
    water_mask,
)

# Create Rome df
d_Rome = {
    "Rome_gt": rome_gt,
    "Rome_pred": rome_pred,
}
df_Rome = pd.DataFrame(data=d_Rome)

# Specify output directory for plots #
outputdir_plots = str(_cfg.outputs.paper_plots_dir)
if not outputdir_plots.endswith(os.sep):
    outputdir_plots += os.sep


# Fig.3 - PS distribution #
bins = np.arange(0, 80, 1)
bins = np.append(bins, [300])

hist, bin_edges = np.histogram(df.nbpoints, bins)  # make the histogram
hist = hist / len(df) * 100

fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(
    range(len(hist)), hist, width=0.7, align="center", edgecolor="black", linewidth=0.4
)

ax.set_ylabel("Share of all pixels", fontsize=20)
ax.set_title(
    "Distribution of PS density in the AoI",
    fontsize=22,
)
ax.set_xlabel("# of PSs in a pixel (# of PSs in km$^2$)", fontsize=20)

# plt.grid(axis="y")
ax.yaxis.set_major_formatter(PercentFormatter(1))
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax.set_yticklabels(
    ("0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"),
    fontsize=18,
)

plt.xlim([-1, 82])
# plt.ylim([0, 2.5])

plt.xticks(np.arange(0, 81, 10), rotation=0)
ax.set_xticklabels(
    (
        "0 \n (0)",
        "10 \n (1,950)",
        "20 \n (3,900)",
        "30 \n (5,850)",
        "40 \n (7,800)",
        "50 \n (9,750)",
        "60 \n (11,700)",
        "70 \n (13,650)",
        "\u226580 \n (\u226515,600)",
    ),
    fontsize=18,
)
# ax.set_xticklabels(np.append(np.arange(0, 80, 10), ">80"), fontsize=18)


# Create an inset plot
axins: Axes = inset_axes(ax, width="60%", height="60%", loc="upper right", borderpad=3)

axins.bar(
    range(len(hist)), hist, width=0.7, align="center", edgecolor="black", linewidth=0.4
)

# Set the y-axis for the inset plot
axins.yaxis.set_major_formatter(PercentFormatter(1))
axins.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
axins.set_yticklabels(("0%", "0.5%", "1%", "1.5%", "2%", "2.5%"), fontsize=18)

axins.set_xticks(np.arange(0, 81, 20))
axins.set_xticklabels(
    (
        "0 \n (0)",
        "20 \n (3,900)",
        "40 \n (7,800)",
        "60 \n (11,700)",
        "\u226580 \n (\u226515,600)",
    ),
    fontsize=18,
)

axins.set_xlim([-1, 70])
axins.set_ylim([0, 2.5])

# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="r")

plt.savefig(
    outputdir_plots + r"fig3a.eps",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# Fig.4  Coherence analysis #
# Fig.4 a/c/e #
# Define the seasons and their corresponding colors
seasons = ["spring", "summer", "fall", "winter"]
colors = ["#FFB000", "#FE6100", "#DC267F", "#785EF0"]

plot_setup_list = [
    ["mask_building", "building", "4a"],
    ["mask_railway", "railway", "4c"],
    ["mask_highway", "road", "4e"],
]

for i in range(0, 3):
    plot_setup = plot_setup_list[i]
    df_masked = df[(df[plot_setup[0]] == 1)]

    bins = np.arange(0, 1.1, 0.1)

    fig, ax = plt.subplots(figsize=[15, 5])
    ax.tick_params(axis="both", which="major", labelsize=24)

    # Initialize an empty list to store the histograms
    hists = []

    # Calculate the histogram for each season
    for season in seasons:
        hist, bin_edges = np.histogram(
            df_masked["coh_mean_" + season],
            bins,
            weights=(np.ones(len(df_masked)) / len(df_masked)),
        )
        hists.append(hist)

    # Calculate the left edge coordinates for the bars
    left_edge_coords = [
        list(map(lambda x: x - 0.4 + 0.2 * i, range(len(hist))))
        for i, hist in enumerate(hists)
    ]

    # Plot the bars for each season
    for i, season in enumerate(seasons):
        ax.bar(
            left_edge_coords[i],
            hists[i],
            width=0.2,
            align="edge",
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            label=season.capitalize(),
        )

    # Set the x-axis labels
    plt.xticks(rotation=0, ha="center")
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_xticklabels(
        [
            f"({bins[i]:.2f} - \n {bins[i + 1]:.2f}]"
            # "({:.2f} - \n {:.2f}]".format(bins[i], bins[i + 1])
            for i, _ in enumerate(hist)
        ]
    )

    plt.legend(loc="upper right", fontsize=20)

    plt.ylim([0, 0.4])
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    ax.set_title(
        f"Seasonal variation in long-term coherence for {plot_setup[1]} pixels",
        fontsize=24,
    )
    ax.set_xlabel("Long-term coherence bins", fontsize=22)
    ax.set_ylabel("Share of considered pixels", fontsize=22)

    plt.savefig(
        f"{outputdir_plots}fig{plot_setup[2]}.eps", dpi=300, bbox_inches="tight"
    )
    plt.close()


# Fig.4b/d/f #
# Define the seasons and their corresponding colors
seasons = ["summer", "winter"]
colors = ["#FE6100", "#785EF0"]

plot_setup_list = [
    ["mask_building", "building", "4b"],
    ["mask_railway", "railway", "4d"],
    ["mask_highway", "road", "4f"],
]

for i in range(0, 3):
    plot_setup = plot_setup_list[i]

    df_masked = df[(df[plot_setup[0]] == 1)]

    bins = [
        -1.0,
        0,
        1.00000000e-01,
        2.00000000e-01,
        3.00000000e-01,
        4.00000000e-01,
        5.00000000e-01,
        6.00000000e-01,
        7.00000000e-01,
        8.00000000e-01,
        9.00000000e-01,
        1.00000000e00,
    ]

    fig, ax = plt.subplots(figsize=[15, 5])
    ax.tick_params(axis="both", which="major", labelsize=24)

    # Initialize an empty list to store the histograms
    hists = []

    # Calculate the histogram for each season
    for season in seasons:
        hist, bin_edges = np.histogram(
            df_masked["ndvi_" + season],
            bins,
            weights=(np.ones(len(df_masked)) / len(df_masked)),
        )
        hists.append(hist)

    # Calculate the left edge coordinates for the bars
    left_edge_coords = [
        list(map(lambda x: x - 0.4 + 0.4 * i, range(len(hist))))
        for i, hist in enumerate(hists)
    ]

    # Plot the bars for each season
    for i, season in enumerate(seasons):
        ax.bar(
            left_edge_coords[i],
            hists[i],
            width=0.4,
            align="edge",
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            label=season.capitalize(),
        )

    # Set the x-axis labels
    plt.xticks(rotation=0, ha="center")
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_xticklabels(
        [
            "({:.1f} - \n {:.1f}]".format(bins[i], bins[i + 1])
            for i, j in enumerate(hist)
        ]
    )

    plt.legend(loc="upper right", fontsize=20)

    plt.ylim([0, 0.4])

    ax.yaxis.set_major_formatter(PercentFormatter(1))

    ax.set_title(
        "Seasonal variation in NDVI for {} pixels".format(plot_setup[1]), fontsize=24
    )
    ax.set_xlabel("NDVI bins", fontsize=22)
    ax.set_ylabel("Share of considered pixels", fontsize=22)

    plt.savefig(
        outputdir_plots + r"fig{}.eps".format(plot_setup[2]),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# Print share of pixels with no data to check if this is something that should be considered
print(
    "percent of winter pixels with nan in ndvi:",
    (len(df_masked[df_masked["ndvi_winter"] == -9]) / len(df_masked)) * 100,
)
print(
    "percent of summer pixels with nan in ndvi:",
    (len(df_masked[df_masked["ndvi_summer"] == -9]) / len(df_masked)) * 100,
)
print(
    "percent of pixels were both winter and summer with nan in ndvi:",
    (
        len(
            df_masked[
                (df_masked["ndvi_winter"] == -9) & (df_masked["ndvi_summer"] == -9)
            ]
        )
        / len(df_masked)
    )
    * 100,
)
print(
    "percent of pixels were either winter or summer with nan in ndvi:",
    (
        len(
            df_masked[
                (df_masked["ndvi_winter"] == -9) | (df_masked["ndvi_summer"] == -9)
            ]
        )
        / len(df_masked)
    )
    * 100,
)


# Fig.6 - Analysis of distribution of pixel with and without PSs #
# Nb pts analysis
# At least 1 PS per OSM category

# Create column to indicate if there are any PSs in a pixel
df["nbpoints_01"] = 0
df["nbpoints_01"][df.nbpoints > 0] = 1

# Set bins
bins = [0, 1, 2]

# Create subsets for each OSM category
df_subset_b = df[df["mask_building"] > 0]
df_subset_r = df[df["mask_railway"] > 0]
df_subset_h = df[df["mask_highway"] > 0]
df_subset_m = df[df["mask_motorway"] > 0]
df_subset_t = df[df["mask_trunk"] > 0]
df_subset_p = df[df["mask_primary"] > 0]
df_subset_s = df[df["mask_secondary"] > 0]

# Plot the histograms
fig, ax = plt.subplots(figsize=(10, 7))
values, bins, bars = plt.hist(
    [
        df_subset_b.nbpoints_01,
        df_subset_r.nbpoints_01,
        df_subset_m.nbpoints_01,
        df_subset_t.nbpoints_01,
        df_subset_p.nbpoints_01,
        df_subset_s.nbpoints_01,
    ],
    bins,
    label=[
        "Buildings",
        "Railways",
        "Motorways",
        "Trunk roads",
        "Primary roads",
        "Secondary roads",
    ],
    edgecolor="black",
    color=["#ffb000", "#fe6100", "#dc267f", "#648fff", "#785ef0", "#785ef0"],
    linewidth=0.5,
    weights=[
        np.ones(len(df_subset_b)) / len(df_subset_b),
        np.ones(len(df_subset_r)) / len(df_subset_r),
        np.ones(len(df_subset_m)) / len(df_subset_m),
        np.ones(len(df_subset_t)) / len(df_subset_t),
        np.ones(len(df_subset_p)) / len(df_subset_p),
        np.ones(len(df_subset_s)) / len(df_subset_s),
    ],
    rwidth=0.5,
)

# Add hatch to differentiate between primary and secondary roads
building_patch = bars[5][0]  # Access the first patch in the first bar
building_patch.set_hatch(".")
building_patch = bars[5][1]  # Access the first patch in the first bar
building_patch.set_hatch(".")

# Set the x-axis labels
plt.xticks([0.5, 1.5])
ax.set_xticklabels(("0", "at least 1"), fontsize=18)

# Set the y-axis labels
plt.ylim([0, 1])
plt.grid(axis="y")
ax.yaxis.set_major_formatter(PercentFormatter(1))
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
ax.set_yticklabels(
    ("0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"),
    fontsize=18,
)

# Add legend
plt.legend(loc="best", fontsize=18)

# Set titles
ax.set_title(
    "Distribuiton of pixels with and without PSs \n depending on infrastructure type",
    fontsize=22,
)
ax.set_xlabel("# of PSs in a pixel", fontsize=20)
ax.set_ylabel("Share of considered pixels", fontsize=20)


plt.savefig(
    outputdir_plots + r"fig6a.eps",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Print values for each category
print("Share of pixels with and without PSs by OSM category", values)


# Fig.8 Errors in prediction of test set #

# Calculate difference between ground truth and prediction
df["diff_pred"] = df.nbpoints - df.pred_test
# Select only pixels with a prediction as those are belonging to the test sets
df_test = df[df.pred_test < 255]

# Set-up contants for plotting
OSM_cat = [None, "building", "railway", "motorway", "trunk", "primary", "secondary"]
OSM_cat_label_list = [
    "all",
    "building",
    "railway",
    "motorway",
    "trunk road",
    "primary road",
    "secondary road",
]
color_list = ["blue", "#ffb000", "#fe6100", "#dc267f", "#648fff", "#785ef0", "#785ef0"]
plot_label_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]

# Generate plots for each OSM category
for i, category in enumerate(OSM_cat):

    # Set-up the subset of the data and some plot settings
    if category is None:
        hatch = None
        df_subset = df_test
        # Rename the category
        category = "all"
    elif category == "secondary":
        hatch = ["."]
        # df_subset = df_test[df_test["mask_{}".format(category)] == 1]
        df_subset = df_test[df_test[f"mask_{category}"] == 1]
    else:
        hatch = None
        # df_subset = df_test[df_test["mask_{}".format(category)] == 1]
        df_subset = df_test[df_test[f"mask_{category}"] == 1]

    # Print MAE for each category
    print(
        "MAE per category - {}: {:.3f}".format(
            category, np.mean(np.abs(df_subset.diff_pred))
        )
    )

    # Plot the histogram for the prediction errors
    # Set-up labels for file name and plot title
    plot_label = plot_label_list[i * 2]
    OSM_cat_label = OSM_cat_label_list[i]

    # Set-up bins for plotting the histogram
    bins = [-1000, -25, -15, -10, -5, -1, 2, 6, 11, 16, 26, 1000]

    # Set-up labels for the bins
    bin_cat = (
        " over -25 \n (over -4,875)",
        "-25 to -16 \n (-4,875 to -3,120)",
        "-15 to -11 \n (-2,925 to -2,145)",
        "-10 to -6 \n (-1,950 to -1,170)",
        "-5 to -2 \n (-975 to -390)",
        "-1 to 1 \n (-195 to 195)",
        "2 to 5 \n (390 to 975)",
        "6 to 10 \n (1,170 to 1,950)",
        "11 to 15 \n (2,145 to 2,925)",
        "16 to 25 \n (3,120 to 4,875)",
        "over 25 \n (over 4,875)",
    )

    # Calculate the histogram for the prediction errors
    bar_heights = {
        "diff": np.histogram(df_subset.diff_pred, bins)[0] / len(df_subset),
    }

    # Set location of labels, width of the bars and initiate a multiplier
    x = np.arange(len(bin_cat))  # the label locations
    width = 0.5  # the width of the bars
    multiplier = 0

    # Generate plots
    fig, ax = plt.subplots(figsize=(12, 6))

    for attribute, measurement in bar_heights.items():
        offset = 0
        rects = ax.bar(
            x + offset,
            measurement,
            width,
            label=attribute,
            color=color_list[i],
            hatch=hatch,
        )
        multiplier += 1

    # Format axes and titles
    ax.set_xticks(x)
    ax.set_xticklabels(bin_cat, rotation=90, fontsize=25)
    plt.xlabel("Prediction error", fontsize=27)
    plt.xlabel("Prediction error in number of PSs (in PSs in a km$^{2}$)", fontsize=27)

    plt.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(("0%", "20%", "40%", "60%", "80%", "100%"), fontsize=25)
    ax.set_ylabel("Share of considered pixels", fontsize=27)

    ax.set_title(
        "Prediction error distribution \n for {} pixels".format(OSM_cat_label),
        fontsize=30,
    )

    plt.savefig(
        outputdir_plots + r"fig8{}.eps".format(plot_label),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot the box plot for the prediction errors
    # Set-up labels for file name and plot title
    plot_label = plot_label_list[i * 2 + 1]
    OSM_cat_label = OSM_cat_label_list[i]

    # Define the bin edges for the box plot
    bin_edges = [0, 5, 10, 15, 20, 30, 40, 300]

    # Create a new column for the bin labels
    df_subset["nbpoints_bins"] = pd.cut(
        df_subset["nbpoints"], bins=bin_edges, right=False
    )

    # Generate plots
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="nbpoints_bins", y="diff_pred", data=df_subset, color=color_list[i])

    ax = plt.gca()
    for i, artist in enumerate(ax.artists):
        # Set the hatching pattern
        artist.set_hatch(hatch)
        # Set the edge color to match the face color
        artist.set_edgecolor("black")

    # Set the x-ticks
    ax.set_xticks(range(7))

    # Set the x-tick labels
    ax.set_xticklabels(
        (
            "[0,5)\n([0 - 975))",
            "[5,10)\n([975 - 1,950))",
            "[10,15)\n([1,950 - 2,925))",
            "[15,20)\n([2,925 - 3,900))",
            "[20,30)\n([3,900 - 5,850))",
            "[30,40)\n([5,850 - 7,800))",
            "40 and over\n(7,800 and over)",
        ),
        rotation=90,
        fontsize=25,
    )
    plt.xlabel("# of PSs in a pixel (# of PSs in a km$^{2}$)", fontsize=27)

    plt.ylim([-90, 90])
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.yticks([-75, -50, -25, 0, 25, 50, 75], fontsize=25)
    plt.ylabel("Prediction error", fontsize=27)

    ax.set_title(
        "Prediction error by PS density \n for {} pixels".format(OSM_cat_label),
        fontsize=30,
    )

    plt.savefig(
        outputdir_plots + r"fig8{}.eps".format(plot_label),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# Fig. 9 - plot with location of AoI #
# Create a GeoDataFrame for the polygons
gdf = gpd.GeoDataFrame(columns=["geometry", "color", "edge", "alpha"])
gdf_tiles = gpd.GeoDataFrame(columns=["geometry", "color", "edge", "alpha"])
gdf_splits = gpd.GeoDataFrame(columns=["geometry", "color", "edge", "alpha"])
gdf_batch = gpd.GeoDataFrame(columns=["geometry", "color", "edge", "alpha"])


# Create a Cartopy map
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()})

# Add basemap
ax.add_feature(cfeature.BORDERS, color="gray")
ax.add_feature(cfeature.COASTLINE, color="gray")

# ax.stock_img()
ax.set_extent([12, 13, 41.5, 42.1])  # Adjust the extent based on your data

request = cimgt.OSM()
ax.add_image(request, 10, interpolation="spline36")

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)
gl.xlabel_style = {"size": 14}
gl.ylabel_style = {"size": 14}


# Plot AOI
minx = 12.1
maxx = 12.953332000009372
miny = 41.62333399999531
maxy = 42.05
lon_lat_list = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]
polygon_geom = Polygon(lon_lat_list)
gdf_tiles = pd.concat(
    [
        gdf_tiles,
        gpd.GeoDataFrame(
            {
                "geometry": [polygon_geom],
                "color": ["black"],
                "edge": ["red"],
                "alpha": 1,
                "linewidth": 2,
            }
        ),
    ],
    ignore_index=True,
)

# Set the GeoDataFrame
gdf_tiles = gpd.GeoDataFrame(gdf_tiles, geometry="geometry")

# Plot the GeoDataFrame
gdf_tiles.plot(
    ax=ax,
    color="none",
    edgecolor=gdf_tiles["edge"],
    alpha=gdf_tiles["alpha"],
    linewidth=gdf_tiles["linewidth"],
)


# Set axis labels
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")


plt.savefig(
    outputdir_plots + r"fig9.eps",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# Fig.10 Rome vs test - nb of pxls with no PS #
# Define bins and categories
bins = [0, 5, 10, 15, 20, 30, 40, 1000]
bin_cat = (
    "[0,5) \n ([0 - 975))",
    "[5,10) \n ([975 - 1,950))",
    "[10,15) \n ([1,950 - 2,925))",
    "[15, 20) \n ([2,925 - 3,900))",
    "[20, 30) \n ([3,900 - 5,850))",
    "[30, 40) \n ([5,850 - 7,800))",
    "over 40 \n (over 7,800)",
)

# Calculate bar heights for the test split and the generalisation trial
bar_heights = {
    "Test split ": np.histogram(df_test.nbpoints, bins)[0] / len(df_test),
    "Generalisation trial": np.histogram(df_Rome.Rome_gt, bins)[0] / len(df_Rome),
}

# Define bar properties
x = np.arange(len(bin_cat))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

# Create the plot
fig, ax = plt.subplots(figsize=(10, 7))

# Add bars to the plot
colors = ["#018571", "#a6611a"]
for attribute, measurement, color in zip(
    bar_heights.keys(), bar_heights.values(), colors
):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=color)
    multiplier += 1

# Set labels and title
plt.xticks(x + width, rotation=90, fontsize=18)
ax.set_xticklabels(bin_cat, fontsize=18)
ax.set_xlabel("# of PSs in a pixel (# of PSs in km$^{2}$)", fontsize=20)

plt.grid(axis="y")
ax.yaxis.set_major_formatter(PercentFormatter(1))
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticklabels(("0%", "20%", "40%", "60%", "80%", "100%"), fontsize=18)
ax.set_ylabel("Share of considered pixels", fontsize=20)

ax.legend(loc="best", fontsize=18)

plt.savefig(
    outputdir_plots + r"fig10.eps",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# Fig.13 Scatter plot of displacement #
# Define plot settings
coordinates = [
    (
        (5.266666666, 5.319999999),
        (52.37333333584, 52.42666666884),
        [5.287, 5.319999999, 52.3734, 52.3734 + 0.017],
        [5.29, 5.3, 5.31],
        [52.38, 52.39],
        0.4,
        "fig13e.eps",
        "overestimation_2",
        "clipped2.shp",
        "underestimation.pkl",
    ),
    (
        (4.979999996, 5.033333329),
        (51.91333333, 51.966666663),
        [4.991, 5.015, 51.924, 51.95],
        [5.29, 5.3, 5.31],
        [52.38, 52.39],
        0.85,
        "fig13f.eps",
        "overestimation_3",
        "clipped.shp",
        "overestimation.pkl",
    ),
]

for (
    (min_lon, max_lon),
    (min_lat, max_lat),
    extent,
    x_ticks,
    y_ticks,
    shrink,
    output_file,
    shp_folder,
    shp_file,
    pkl_file,
) in coordinates:
    # Define the paths
    shapefile_path = os.path.join(
        root, "EGMS_data_for_paper_plots", shp_folder, shp_file
    )
    pickle_path = os.path.join(root, "EGMS_data_for_paper_plots", pkl_file)

    # Check if the pickle file exists
    if os.path.exists(pickle_path):
        # If it exists, read it
        gdf = pd.read_pickle(pickle_path)
    else:
        # If it doesn't exist, create it
        gdf = gpd.read_file(shapefile_path)
        gdf.to_pickle(pickle_path)

    # Explode MultiPoint geometries into individual points
    gdf_exploded = gdf.explode(index_parts=True)

    # Clip the GeoDataFrame
    clipped_gdf = gdf_exploded.cx[min_lon:max_lon, min_lat:max_lat]

    clipped_gdf.sort_values(by="vel", ascending=False, inplace=True)

    # Create a Cartopy map
    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax.set_extent(extent)

    request = cimgt.GoogleTiles(style="satellite")
    ax.add_image(request, 15, interpolation="spline36")

    scatter = ax.scatter(
        clipped_gdf.geometry.x,
        clipped_gdf.geometry.y,
        c=clipped_gdf[
            "vel"
        ],  # Replace 'displacement_column' with the actual column name
        cmap="RdYlBu",  # You can choose another colormap
        marker="o",
        edgecolors="k",  # Black edges for better visibility
        linewidths=0.1,  # Width of the edges
        s=5,  # Marker size
        vmin=-15,
        vmax=15,
        zorder=2,
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.top_labels = False
    gl.right_labels = False

    gl.xlocator = mticker.FixedLocator(x_ticks)
    gl.ylocator = mticker.FixedLocator(y_ticks)

    cbar = plt.colorbar(scatter, ax=ax, shrink=shrink)
    cbar.set_label("Velocity [mm/year]", fontsize=16)
    cbar.ax.tick_params(labelsize=15)

    plt.savefig(
        os.path.join(outputdir_plots, output_file),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# Linear regresssion test #

# Fitting straight line
# Linear y = ax+b
equation = "nbpoints ~ coh_mean"
y_hat_ver = "lin"
title = "y = {:.2f}x + {:.2f}"


m = smf.ols(equation, data=df).fit()
print(m.summary())

x = df["coh_mean"]
y = df["nbpoints"]

x_t = x
y_t = y

alpha = 0.05


yhat = m.fittedvalues
predictions = m.get_prediction(df).summary_frame(alpha)
plot_title = title.format(m.params[1], m.params[0])

plot_title = (
    plot_title
    + "\n R-squared: {:.3f}".format(m.rsquared)
    + "\n RMSE: {:.3f}".format(np.sqrt(m.mse_resid))
)

# print('\n\n', plot_title, '\n R-squared: {:.3f}'.format(m.rsquared),
#       '\n Standard error of regression: {:.2f} \n\n'.format(np.sqrt(m.mse_resid)))


# Fitting plot
df_sample = df.sample(frac=0.1)
x_s = df_sample["coh_mean"]
y_s = df_sample["nbpoints"]


# fig, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(x_s, y_s, label="Data", s=0.2)
# ax.scatter(x, yhat, color="r", label="Predicted")
# legend = ax.legend(loc="best")
# ax.set_title(plot_title)
# ax.set_xlabel("coh")
# ax.set_ylabel("nbpoints")
# ax.set_ylim([0, 200])


# plot correlation between coherence and nb of points
x = df["coh_mean"].values
y = df["nbpoints"].values
title = "Long-term coherence vs number of points in a pixel \n (all pixels)"
xlabel = "coh_mean"
ylabel = "nbpoints"
x_limit = 1
y_limit = 100
vmax = 200
y_limit_min = 0
x_limit_min = 0

dpi = 23
# dpi = 10


fig, ax = plot_scatter_denisty(
    x, y, title, xlabel, ylabel, x_limit, y_limit, vmax, x_limit_min, y_limit_min, dpi
)

plt.scatter(x, predictions["mean"], label="Regression line")

plt.savefig(
    os.path.join(outputdir_plots, "linear_regression_1in.eps"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# # Residuals plot
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(yhat, m.resid, label="Residuals", s=0.5)
# plt.axhline(0, color="darkorange", linestyle="--")
# legend = ax.legend(loc="best")
# ax.set_title("Fitted vs residuals")
# ax.set_xlabel("Fitted/predicted y")
# ax.set_ylabel("Residuals")

mse = mean_squared_error(y, yhat)
rmse = np.sqrt(mse)
print("rmse: ", rmse)

mae = mean_absolute_error(y, yhat)
print("mae: ", mae)


# Fitting polynomial
# polynomial y = a(np.sqrt(x))^2 + b*np.sqrt(x) + c
equation = "nbpoints ~ np.sqrt(coh_mean) + I(np.sqrt(coh_mean)**2)"
x_t = np.sqrt(x)
y_t = y
y_hat_ver = "poly"
title = "y = {:.2f}sqrt(x) ^2 + {:.2f}sqrt(x)  + {:.2f}"


m = smf.ols(equation, data=df).fit()
print(m.summary())

x = df["coh_mean"]
y = df["nbpoints"]

alpha = 0.05

y_hat_ver == "poly"
yhat = m.fittedvalues
predictions = m.get_prediction(df).summary_frame(alpha)
plot_title = title.format(m.params[2], m.params[1], m.params[0])

plot_title = (
    f"{plot_title}\n R-squared: {m.rsquared:.3f}" f"\n RMSE: {np.sqrt(m.mse_resid):.3f}"
)

# print('\n\n', plot_title, '\n R-squared: {:.3f}'.format(m.rsquared),
#       '\n Standard error of regression: {:.2f} \n\n'.format(np.sqrt(m.mse_resid)))


# Fitting plot
df_sample = df.sample(frac=0.1)
x_s = df_sample["coh_mean"]
y_s = df_sample["nbpoints"]


# fig, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(x_s, y_s, label="Data", s=0.2)
# ax.scatter(x, yhat, color="r", label="Predicted")
# legend = ax.legend(loc="best")
# ax.set_title(plot_title)
# ax.set_xlabel("coh")
# ax.set_ylabel("nbpoints")
# ax.set_ylim([0, 200])


# plot correlation between coherence and nb of points
x = df["coh_mean"].values
y = df["nbpoints"].values
title = "Long-term coherence vs number of points in a pixel \n (all pixels)"
xlabel = "coh_mean"
ylabel = "nbpoints"
x_limit = 1
y_limit = 100
vmax = 200
y_limit_min = 0
x_limit_min = 0

dpi = 23
# dpi = 10


fig, ax = plot_scatter_denisty(
    x, y, title, xlabel, ylabel, x_limit, y_limit, vmax, x_limit_min, y_limit_min, dpi
)

plt.scatter(x, predictions["mean"], label="Regression line")
plt.savefig(
    os.path.join(outputdir_plots, "poly_regression_1in.eps"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# # Residuals plot
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(yhat, m.resid, label="Residuals", s=0.5)
# plt.axhline(0, color="darkorange", linestyle="--")
# legend = ax.legend(loc="best")
# ax.set_title("Fitted vs residuals")
# ax.set_xlabel("Fitted/predicted y")
# ax.set_ylabel("Residuals")

mse = mean_squared_error(y, yhat)
rmse = np.sqrt(mse)
print("rmse: ", rmse)

mae = mean_absolute_error(y, yhat)
print("mae: ", mae)


np.random.seed(0)
# ONE INPUT
# Read data
X = df["coh_mean"].values
y = df["nbpoints"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Reshape the data so that the first dimension is the number of samples
# and the second dimension is the number of features
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"Mean absolute error: {mean_absolute_error(y_test, y_pred)}")
print(f"Coefficient of determination (R^2): {r2_score(y_test, y_pred)}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# TWO INPUTS
# Read data
X = df[["coh_mean", "infr_map"]].values
y = df["nbpoints"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"Mean absolute error: {mean_absolute_error(y_test, y_pred)}")
print(f"Coefficient of determination (R^2): {r2_score(y_test, y_pred)}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")


# Fitting polynomial


# ONE INPUT
# Read data
X = df[["coh_mean"]].values
y = df["nbpoints"].values

# Create PolynomialFeatures object with degree=2
poly = PolynomialFeatures(degree=2)

# Transform the features to 2nd degree polynomial features
X_poly = poly.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=0
)

# # Reshape the data so that the first dimension is the number of samples
# and the second dimension is the number of features
# X_train = X_train[:, np.newaxis]
# X_test = X_test[:, np.newaxis]

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"Mean absolute error: {mean_absolute_error(y_test, y_pred)}")
print(f"Coefficient of determination (R^2): {r2_score(y_test, y_pred)}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")


# TWO INPUTS
# Read data
X = df[["coh_mean", "infr_map"]].values
y = df["nbpoints"].values

# Create PolynomialFeatures object with degree=2
poly = PolynomialFeatures(degree=2)

# Transform the features to 2nd degree polynomial features
X_poly = poly.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=0
)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"Mean absolute error: {mean_absolute_error(y_test, y_pred)}")
print(f"Coefficient of determination (R^2): {r2_score(y_test, y_pred)}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"Powers that should be used in the equation: {poly.powers_}")
