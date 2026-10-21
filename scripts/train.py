"""
This script can be used to train a pixel-wise regression model 
for the prediction of the long-term coherence.
"""

# check Python version (3.9 needed)
# !python --version # jupyter
import platform
print(platform.python_version())

# Import modules
import os
import shutil
import warnings
import time
import datetime

import torch

from torchgeo.datasets import (
    BoundingBox,
)

from helper_functions.model_setup import training_setup
from helper_functions.torchgeo_fun_to_read_input import (
    scale_coh,
    plot_input,
    RasterDataset_imgs,
    RasterDataset_msks,
)

# Check if torch with cuda enabled and working
# torch.zeros(1).cuda()
torch.cuda.is_available()

# Ignore warning about number of warnings
# See https://github.com/Lightning-AI/lightning/issues/10182
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")


### Set-up parameters and import data ###
# Set up parameters

# Path for saving results (such as model checkpoints)
path_prefix = "/mnt/d/ML_model_data_paper/trainingnn_results"

experiment_name = "pixel_wise_regression_model_training_updated_OSM"
experiment_dir = os.path.join(path_prefix, experiment_name)
print(experiment_dir)

os.makedirs(experiment_dir, exist_ok=True)

# Import images and mask and generate dataset for training

# Import images and masks
root = "/mnt/d/ML_model_data_paper/input"

imgs = RasterDataset_imgs(
    paths=os.path.join(root, "global_coh_NL/summer_vv_rho"), transforms=scale_coh
)

# imgs_OSM = RasterDataset_imgs(paths=os.path.join(root, "OSM_input_channel_v2"))
imgs_OSM = RasterDataset_imgs(paths=os.path.join(root, "OSM_input_updated"))


msks = RasterDataset_msks(paths=os.path.join(root, "mask", "mask_EGMS"))

print("imgs info: ", "crs: ", imgs.crs, "res: ", imgs.res, "length: ", len(imgs))
print(
    "imgs OSM info: ", "crs: ", imgs.crs, "res: ", imgs.res, "length: ", len(imgs_OSM)
)
print("msks info: ", "crs: ", msks.crs, "res: ", msks.res, "length: ", len(msks))


### Pre-process images so that they are combined into one dataset ###
# Combine long-term coherence with infrastructure map
imgs = imgs & imgs_OSM
print(imgs)

print(msks)

# Create an intersection dataset with images and mask
dataset = imgs & msks

print(
    "dataset info: ",
    "crs: ",
    dataset.crs,
    "res: ",
    dataset.res,
    "length: ",
    len(dataset),
)

# Check if correct number of input channels and mask channels

# Define bounding box
bounding_box = BoundingBox(
    minx=3.0,
    maxx=7.999999996,
    miny=50.000000004,
    maxy=54.0,
    mint=0.0,
    maxt=9.223372036854776e18,
)

print(
    " nb of input channels: ",
    dataset[bounding_box]["image"].shape[0],
    "\n",
    "nb of mask channels: ",
    dataset[bounding_box]["mask"].shape[0],
)

# Check if correct mask values
print("Mask values: ", (torch.unique(dataset[bounding_box]["mask"])))
print("Nb of unique values: ", len((torch.unique(dataset[bounding_box]["mask"]))))

# Plot input for checks
plot_input(dataset[bounding_box], rows=1)


# ### Plot samples on a map (useful for testing) ###

# # Plot only for the selected seed
# seed = 54321

# # TEST SAMPLING
# seed_everything(seed, workers=True)

# patch_size = 64
# batch_size = 16
# nb_btch_epch = 10
# length = None

# length_tr = 1
# length_val = length_test = 1
# num_workers = 0

# # DIVIDE DATASET
# # random_bbox_splitting: A 0.8/0.2 split will put 80% of each file in one dataset
# # and the other 20% in the other dataset. Every file will occur in both datasets.
# seed_everything(seed, workers=True)
# generator = torch.Generator().manual_seed(seed)
# (
#     train_dataset,
#     val_dataset,
#     test_dataset,
# ) = random_bbox_splitting(dataset, [0.7, 0.15, 0.15], generator)

# print(
#     "Nb of tiles in train, val, and test sets:",
#     len(train_dataset),
#     len(val_dataset),
#     len(test_dataset),
# )

# # DEFINE SAMPLERS
# sampler = RandomGeoSampler(train_dataset, patch_size, length)

# val_sampler = GridGeoSampler(
#     # Usually the stride should be slightly smaller than the chip size such that each chip
#     # has some small overlap with surrounding chips.
#     val_dataset,
#     patch_size,
#     patch_size - 8,
# )

# test_sampler = GridGeoSampler(
#     # Usually the stride should be slightly smaller than the chip size such that each chip
#     # has some small overlap with surrounding chips.
#     test_dataset,
#     patch_size,
#     patch_size - 8,
# )

# # DEFINE DATALOADER
# train_dataloader = DataLoader(
#     dataset=train_dataset,
#     batch_size=batch_size,
#     sampler=sampler,
#     num_workers=num_workers,
#     collate_fn=stack_samples,
#     generator=generator,
# )

# val_dataloader = DataLoader(
#     dataset=val_dataset,
#     batch_size=batch_size,
#     sampler=val_sampler,
#     num_workers=num_workers,
#     collate_fn=stack_samples,
#     generator=generator,
# )

# test_dataloader = DataLoader(
#     dataset=test_dataset,
#     batch_size=batch_size,
#     sampler=test_sampler,
#     num_workers=num_workers,
#     collate_fn=stack_samples,
#     generator=generator,
# )

# # Create a GeoDataFrame for the polygons
# gdf = gpd.GeoDataFrame(columns=["geometry", "color", "edge", "alpha"])
# gdf_tiles = gpd.GeoDataFrame(columns=["geometry", "color", "edge", "alpha"])
# gdf_splits = gpd.GeoDataFrame(columns=["geometry", "color", "edge", "alpha"])
# gdf_batch = gpd.GeoDataFrame(columns=["geometry", "color", "edge", "alpha"])

# # Create a Cartopy map
# fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()})

# # Add basemap
# ax.add_feature(cfeature.BORDERS, color="gray")
# ax.add_feature(cfeature.COASTLINE, color="gray")
# ax.set_extent([2.5, 8.5, 49.5, 54.5])  # Adjust the extent based on your data

# request = cimgt.OSM()
# ax.add_image(request, 7, interpolation="spline36")

# gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)
# gl.xlabel_style = {"size": 14}
# gl.ylabel_style = {"size": 14}

# # Plot tiles
# for x in range(3, 8, 1):
#     for y in range(50, 54, 1):
#         minx, maxx, miny, maxy = x, x + 1, y, y + 1
#         lon_lat_list = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]
#         polygon_geom = Polygon(lon_lat_list)
#         gdf_tiles = pd.concat(
#             [
#                 gdf_tiles,
#                 gpd.GeoDataFrame(
#                     {
#                         "geometry": [polygon_geom],
#                         "color": ["black"],
#                         "edge": ["black"],
#                         "alpha": 1,
#                     }
#                 ),
#             ],
#             ignore_index=True,
#         )

# # Set the GeoDataFrame
# gdf_tiles = gpd.GeoDataFrame(gdf_tiles, geometry="geometry")

# # Plot the GeoDataFrame
# gdf_tiles.plot(
#     ax=ax, color="none", edgecolor=gdf_tiles["edge"], alpha=gdf_tiles["alpha"]
# )

# # Plot splits
# def prepare_plot_splits(minx_l, maxx_l, miny_l, maxy_l, color):
#     gdf_splits = gpd.GeoDataFrame()  # Initialize gdf_splits

#     for i in range(20):
#         minx, maxx, miny, maxy = minx_l[i], maxx_l[i], miny_l[i], maxy_l[i]
#         lon_lat_list = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]
#         polygon_geom = Polygon(lon_lat_list)
#         gdf_splits = gpd.GeoDataFrame(
#             pd.concat(
#                 [
#                     gdf_splits,
#                     gpd.GeoDataFrame(
#                         {
#                             "geometry": [polygon_geom],
#                             "color": [color],
#                             "edge": ["none"],
#                             "alpha": 0.2,
#                         }
#                     ),
#                 ],
#                 ignore_index=True,
#             )
#         )
#     return gdf_splits

# # Set borders of regions selected for training split with this seed
# minx_l = [3, 4, 5, 6, 7, 3, 4.3, 5.3, 6, 7, 3, 4, 5.3, 6, 7, 3, 4, 5, 6, 7]
# maxx_l = [4, 4.7, 6, 7, 8, 4, 5, 6, 6.7, 8, 3.7, 5, 6, 7, 8, 3.7, 5, 6, 7, 8]
# miny_l = [53, 53, 53.3, 53.3, 53, 52, 52, 52, 52, 52, 51, 51, 51, 51.3, 51.3, 50, 50, 50, 50, 50.3]
# maxy_l = [53.7, 54, 54, 54, 53.7, 52.7, 53, 53, 53, 52.7, 52, 51.7, 52, 52, 52, 51, 50.7, 50.7, 50.7, 51]
# gdf_splits = prepare_plot_splits(minx_l, maxx_l, miny_l, maxy_l, "blue")

# # Set borders of regions selected for validation split with this seed
# minx_l = [ 3, 4.7, 5.5, 6.5, 7, 3, 4, 5, 6.7, 7, 3.7, 4, 5, 6.5, 7.5, 3.7, 4, 5, 6, 7.5, ]
# maxx_l = [ 3.5, 5, 6, 7, 7.5, 3.5, 4.3, 5.3, 7, 7.5, 4, 4.5, 5.3, 7, 8, 4, 4.5, 5.5, 6.5, 8, ]
# miny_l = [ 53.7, 53, 53, 53, 53.7, 52.7, 52.5, 52.5, 52, 52.7, 51, 51.7, 51.5, 51, 51, 50, 50.7, 50.7, 50.7, 50, ]
# maxy_l = [ 54, 53.5, 53.3, 53.3, 54, 53, 53, 53, 52.5, 53, 51.5, 52, 52, 51.3, 51.3, 50.5, 51, 51, 51, 50.3, ]
# gdf_splits = prepare_plot_splits(minx_l, maxx_l, miny_l, maxy_l, "red")

# # Set borders of regions selected for validation test with this seed
# minx_l = [ 3.5, 4.7, 5, 6, 7.5, 3.5, 4, 5, 6.7, 7.5, 3.7, 4.5, 5, 6, 7, 3.7, 4.5, 5.5, 6.5, 7, ]
# maxx_l = [4, 5, 5.5, 6.5, 8, 4, 4.3, 5.3, 7, 8, 4, 5, 5.3, 6.5, 7.5, 4, 5, 6, 7, 7.5]
# miny_l = [ 53.7, 53.5, 53, 53, 53.7, 52.7, 52, 52, 52.5, 52.7, 51.5, 51.7, 51, 51, 51, 50.5, 50.7, 50.7, 50.7, 50, ]
# maxy_l = [ 54, 54, 53.3, 53.3, 54, 53, 52.5, 52.5, 53, 53, 52, 52, 51.5, 51.3, 51.3, 51, 51, 51, 51, 50.3, ]
# gdf_splits = prepare_plot_splits(minx_l, maxx_l, miny_l, maxy_l, "green")

# # Set the GeoDataFrame
# gdf_splits = gpd.GeoDataFrame(gdf_splits, geometry="geometry")

# # Plot the GeoDataFrame
# gdf_splits.plot(
#     ax=ax,
#     color=gdf_splits["color"],
#     edgecolor=gdf_splits["edge"],
#     alpha=gdf_splits["alpha"],
# )


# # Plot batches
# def prepare_plot_batches(dataloader, color, condition, gdf_batch):
#     for j, batch in enumerate(dataloader):
#         if j == condition:  # It is the id of the batch
#             for i in range(len(batch["bbox"])):
#                 minx, maxx, miny, maxy, _, _ = batch["bbox"][i]
#                 lon_lat_list = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]
#                 polygon_geom = Polygon(lon_lat_list)
#                 gdf_batch = gpd.GeoDataFrame(
#                     pd.concat(
#                         [
#                             gdf_batch,
#                             gpd.GeoDataFrame(
#                                 {
#                                     "geometry": [polygon_geom],
#                                     "color": [color],
#                                     "edge": [color],
#                                     "alpha": 0.5,
#                                 }
#                             ),
#                         ],
#                         ignore_index=True,
#                     )
#                 )
#     return gdf_batch

# # Initialize the GeoDataFrame
# gdf_batch = gpd.GeoDataFrame()

# # Prepare gdf for plotting training batch
# gdf_batch = prepare_plot_batches(train_dataloader, "blue", 50, gdf_batch)

# # Prepare gdf for plotting validation batch
# gdf_batch = prepare_plot_batches(val_dataloader, "red", 5, gdf_batch)

# # Prepare gdf for plotting test batch
# gdf_batch = prepare_plot_batches(test_dataloader, "green", 20, gdf_batch)

# # Set the GeoDataFrame
# gdf_batch = gpd.GeoDataFrame(gdf_batch, geometry="geometry")

# # Plot the GeoDataFrame
# gdf_batch.plot(
#     ax=ax,
#     color=gdf_batch["color"],
#     edgecolor=gdf_batch["edge"],
#     alpha=gdf_batch["alpha"],
# )

# # Set axis labels
# ax.set_xlabel("Longitude")
# ax.set_ylabel("Latitude")


# plt.savefig(
#     experiment_dir + r"/fig2.jpg",
#     dpi=600,
#     bbox_inches="tight",
# )
# plt.close()


### Set up tensorboard (only works in jupyter? To be rewritten to work in VSC?) ###
# See this for VSC tensorboard: https://stackoverflow.com/questions/63938552/how-to-run-tensorboard-in-vscode

# # Jupyter noteboook
# %load_ext tensorboard
# # %reload_ext tensorboard

# # Python
# tb_log_directory = os.path.join(path_prefix, "logs")
# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', tb_log_directory])
# url = tb.launch()

# # Clear old tensorboard files
# folder_path = r"C:\Users\Dominika\AppData\Local\Temp\.tensorboard-info"
# extension = ".info"
# for filename in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, filename)
#     try:
#         if os.path.isfile(file_path) and filename.endswith(extension):
#             os.remove(file_path)
#     except Exception as e:
#         print(f"Error deleting {file_path}: {e}")
#         print(f"Error deleting {file_path}: {e}")
# print("Deletion done")

# %tensorboard --logdir ../logs/ --host localhost #jupyter notebook


### Run training in different configurations ###

# Set-up parameters

# Constant parameters
in_channels = 2  # Number of channels in input image
ignore_index = None  # Optional integer class index to ignore in the loss and metrics
lr_patience = 10  # Patience for learning rate scheduler
min_delta_lr = 0.01
patience = 15  # Patience for early stopping
min_delta = (
    0.01  # when change in val_loss less than that for appropriate number of epochs
)
# early stopping will be triggered
max_epochs = 300  # maximum number of epochs

gpu_id = 0
device = torch.device(f"cuda:{gpu_id}")
num_dataloader_workers = 0

param_const = [
    in_channels,
    ignore_index,
    lr_patience,
    min_delta_lr,
    patience,
    min_delta,
    max_epochs,
    gpu_id,
    device,
    num_dataloader_workers,
]

# Modifiable parameters
param_opts = [
    [16, 64, None, "resnet34", None, "mae", 0.00776601, 0.9, 0.99, 0.01],
]
patch_size = param_opts[0][1]
batch_size = param_opts[0][0]

# Iterate through the list of seeds and run the training with constant set of parameters but different seed
# for seed in [0, 1, 10, 42, 123, 1234, 12345, 54321, 999, 9999, 7436, 9703, 4502, 9648, 3606, 9570, 573, 7993, 9491, 6684]:
for seed in [54321]:
    for i in list(range(len(param_opts))):
        # Set-up training
        trainer, task, datamodule = training_setup(
            seed, param_const, param_opts[i], dataset, experiment_dir, experiment_name
        )

        # Start timer
        start_time = time.time()

        # Run training
        _ = trainer.fit(model=task, datamodule=datamodule)

        # End timer
        end_time = time.time()

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        print("Training took {:.2f} seconds".format(elapsed_time))

        # Copy best checkpoint and rename it to include training parameters
        dir_best = "/mnt/d/ML_model_data_paper/best_models"

        new_name = "seed_{}_{}.ckpt".format(
            seed, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        dir_cop = os.path.join(experiment_dir, new_name)

        shutil.copy(trainer.checkpoint_callback.best_model_path, dir_cop)

print("Training done")
