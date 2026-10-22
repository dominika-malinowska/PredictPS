"""
This script can be used to run model testing and prediction.
"""

# check Python version (3.9 needed)
# !python --version # jupyter
import platform

print(platform.python_version())

# Import modules
import os
import warnings
from datetime import datetime

import torch

from torchgeo.datasets import (
    BoundingBox,
    stack_samples,
)
from torchgeo.datasets.splits import random_bbox_splitting
from torchgeo.samplers import GridGeoSampler

from torch.utils.data import DataLoader

from lightning.pytorch import seed_everything

import timeit

from ps_density.torchgeo_classes_def import RegressionTask, PixelwiseRegressionTask
from ps_density.torchgeo_fun_to_read_input import (
    scale_coh,
    plot_input,
    RasterDataset_imgs,
    RasterDataset_msks,
    RasterDatasetOptical_imgs,
)
from ps_density.model_setup import test_pred_setup
from ps_density.paths import load_paths
from ps_density.prediction_plots import (
    georreferenced_chip_generator,
    merge_georeferenced_chips,
    plot_input_pred_patch,
)


# Check if torch with cuda enabled and working
# torch.zeros(1).cuda()
torch.cuda.is_available()

# Ignore warning about number of warnings
# See https://github.com/Lightning-AI/lightning/issues/10182
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")


### Set-up model parameters ###

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


### Read dataset ###
# Set-up paths
_cfg = load_paths()
root = str(_cfg.data_root)

# Path for saving results (such as model checkpoints)
path_prefix = str(_cfg.test.output.revision_results_dir)

experiment_name = _cfg.test.output.experiment_name
experiment_dir = os.path.join(path_prefix, experiment_name)
# print(experiment_dir)

os.makedirs(experiment_dir, exist_ok=True)

# Import images and masks
imgs_ltc = RasterDataset_imgs(
    paths=str(_cfg.test.input.coherence_summer_dir),
    transforms=scale_coh,
)
imgs_OSM = RasterDataset_imgs(paths=str(_cfg.test.input.osm_dir))
msks = RasterDataset_msks(paths=str(_cfg.test.input.mask_egms_dir))

print(
    "imgs info: ",
    "crs: ",
    imgs_ltc.crs,
    "res: ",
    imgs_ltc.res,
    "length: ",
    len(imgs_ltc),
)
print(
    "imgs OSM info: ",
    "crs: ",
    imgs_OSM.crs,
    "res: ",
    imgs_OSM.res,
    "length: ",
    len(imgs_OSM),
)
print("msks info: ", "crs: ", msks.crs, "res: ", msks.res, "length: ", len(msks))


### Pre-proces images so that they are combined into one dataset ###
# Combine long-term coherence with infrastructure map
imgs = imgs_ltc & imgs_OSM
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


# ### Model variability - print results from the 20 seeds ###

# # List all trained models with varied seed
# best_models_path = os.path.join(root, "best_models_15122023")

# models_seeds = os.listdir(best_models_path)
# print(
#     "number of models considered: ", len(models_seeds), "\n Model seeds: ", models_seeds
# )


# # Iterate throught the results from the twenty different seeds and print the testing results
# for model_seed in models_seeds:
#     model = PixelwiseRegressionTask.load_from_checkpoint(
#         checkpoint_path=os.path.join(best_models_path, "{}".format(model_seed))
#     )

#     seed = int(model_seed.split("_")[1])

#     trainer, datamodule = test_pred_setup(
#         seed, param_const, param_opts[0], dataset, experiment_dir, experiment_name
#     )

#     print("Model from checkpoint: ", model_seed)
#     trainer.test(model=model, datamodule=datamodule)


### Print results from one specific model ###
# Read trained model
model = PixelwiseRegressionTask.load_from_checkpoint(
    checkpoint_path=str(_cfg.test.checkpoints.netherlands_20231214)
)
plots_folder_name = "test_pred_54321_20231214_2in"

# Test updated model
model = PixelwiseRegressionTask.load_from_checkpoint(
    checkpoint_path=str(_cfg.test.checkpoints.netherlands_updated_osm)
)
plots_folder_name = "test_pred_OSM_updated"


# Set-up seed
seed = 54321
seed_everything(seed, workers=True)

trainer, datamodule = test_pred_setup(
    seed, param_const, param_opts[0], dataset, experiment_dir, experiment_name
)

trainer.test(model=model, datamodule=datamodule)

### Generate plots ###

generator = torch.Generator().manual_seed(seed)
(
    train_dataset,
    val_dataset,
    test_dataset,
) = random_bbox_splitting(dataset, [0.7, 0.15, 0.15], generator=generator)

sampler_pred = GridGeoSampler(test_dataset, patch_size, patch_size)

dataloader_pred = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    sampler=sampler_pred,
    num_workers=0,
    collate_fn=stack_samples,
    generator=generator,
)

# Generate prediction chips
# disable randomness, dropout, etc...
model.eval()

# Generate geotiff
pixel_size = test_dataset.res
crs = test_dataset.crs.to_epsg()

start = timeit.default_timer()  # Measuring the time
chips_generator = georreferenced_chip_generator(dataloader_pred, model, crs, pixel_size)
print("The time taken to predict was: ", timeit.default_timer() - start)

# Save chips as geotiff
start = timeit.default_timer()  # Measuring the time
file_name = os.path.join(
    root,
    "merged_prediction_fixed_stiching_{}.tif".format(
        datetime.now().strftime("%Y%m%d_%H%M%S")
    ),
)

overlap = 8
bounds = (int(3), int(50), int(8), int(54))

# merge_georeferenced_chips(chips_generator, file_name)
merge_georeferenced_chips(chips_generator, file_name, overlap, bounds)
print(
    "The time taken to generate a georrefenced image and save it was: ",
    timeit.default_timer() - start,
)

# Generate plots for paper

# Read optical data
opt = RasterDatasetOptical_imgs(paths=str(_cfg.test.input.google_satellite_dir))
print(opt)

outputdir_plots = os.path.join(str(_cfg.test.output.plots_review_dir), plots_folder_name)
# Create directory if it doesn't exist
os.makedirs(outputdir_plots, exist_ok=True)

plot_axis = True
plot_titles = False
plot_bar = True


# Plot all patches (!TIME CONSUMING!)
# df_mae_test = plot_input_pred_patch(
#     dataloader_pred, model, opt, batch_size, plot_axis, plot_bar, plot_titles, outputdir_plots
# )
# df_mae_test.to_csv(
#     outputdir_plots + "\\df_mae_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M%S")), index=False
# )
# print("Mean MAE", df_mae_test.mae.mean())

# Plot specific patch
df_mae_test = plot_input_pred_patch(
    dataloader_pred,
    model,
    opt,
    batch_size,
    plot_axis,
    plot_bar,
    plot_titles,
    outputdir_plots,
    specific_batch=0,
    specific_patch=1,
)


df_mae_test = plot_input_pred_patch(
    dataloader_pred,
    model,
    opt,
    batch_size,
    True,
    False,
    True,
    outputdir_plots,
    specific_batch=0,
    specific_patch=13,
)

df_mae_test = plot_input_pred_patch(
    dataloader_pred,
    model,
    opt,
    batch_size,
    True,
    False,
    False,
    outputdir_plots,
    specific_batch=2,
    specific_patch=6,
)

df_mae_test = plot_input_pred_patch(
    dataloader_pred,
    model,
    opt,
    batch_size,
    True,
    False,
    False,
    outputdir_plots,
    specific_batch=33,
    specific_patch=0,
)

df_mae_test = plot_input_pred_patch(
    dataloader_pred,
    model,
    opt,
    batch_size,
    True,
    True,
    False,
    outputdir_plots,
    specific_batch=49,
    specific_patch=0,
)


# Fig. 1 vs 2 inputs

df_mae_test = plot_input_pred_patch(
    dataloader_pred,
    model,
    opt,
    batch_size,
    True,
    False,
    True,
    outputdir_plots,
    specific_batch=49,
    specific_patch=8,
    box_coords=[40, 25, 24, 20],
)


df_mae_test = plot_input_pred_patch(
    dataloader_pred,
    model,
    opt,
    batch_size,
    True,
    True,
    False,
    outputdir_plots,
    specific_batch=2,
    specific_patch=15,
    box_coords=[25, 15, 39, 10],
)


# Fig. discussion
df_mae_test = plot_input_pred_patch(
    dataloader_pred,
    model,
    opt,
    batch_size,
    True,
    False,
    True,
    outputdir_plots,
    specific_batch=14,
    specific_patch=4,
)

df_mae_test = plot_input_pred_patch(
    dataloader_pred,
    model,
    opt,
    batch_size,
    True,
    False,
    False,
    outputdir_plots,
    specific_batch=4,
    specific_patch=11,
    box_coords=[0.2, 42, 24, 21.8],
)

df_mae_test = plot_input_pred_patch(
    dataloader_pred,
    model,
    opt,
    batch_size,
    True,
    True,
    False,
    outputdir_plots,
    specific_batch=3,
    specific_patch=0,
    box_coords=[20, 0, 30, 15],
)


df_mae_test = plot_input_pred_patch(
    dataloader_pred,
    model,
    opt,
    batch_size,
    True,
    False,
    True,
    outputdir_plots,
    specific_batch=47,
    specific_patch=15,
    box_coords=[24, 44, 39.2, 19.2],
)

df_mae_test = plot_input_pred_patch(
    dataloader_pred,
    model,
    opt,
    batch_size,
    True,
    True,
    False,
    outputdir_plots,
    specific_batch=25,
    specific_patch=9,
    box_coords=[13, 20, 29, 31],
)


### Generate plots for papers for one model ###

# Create an intersection dataset with images and mask
dataset_1in = imgs_ltc & msks


# 1 input
model_1in = PixelwiseRegressionTask.load_from_checkpoint(
    checkpoint_path=str(_cfg.test.checkpoints.one_channel_20231218)
)
param_const_1in = [
    1,
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

plots_folder_name_1in = "test_pred_54321_20231218_1in"

trainer_1in, datamodule_1in = test_pred_setup(
    seed, param_const_1in, param_opts[0], dataset_1in, experiment_dir, experiment_name
)

trainer_1in.test(model=model_1in, datamodule=datamodule_1in)

### Generate plots ###

(
    train_dataset_1in,
    val_dataset_1in,
    test_dataset_1in,
) = random_bbox_splitting(dataset_1in, [0.7, 0.15, 0.15], generator=generator)

sampler_pred_1in = GridGeoSampler(test_dataset_1in, patch_size, patch_size)

dataloader_pred_1in = DataLoader(
    dataset=test_dataset_1in,
    batch_size=batch_size,
    sampler=sampler_pred_1in,
    num_workers=0,
    collate_fn=stack_samples,
    generator=generator,
)

# Generate prediction chips
# disable randomness, dropout, etc...
model_1in.eval()


outputdir_plots_1in = os.path.join(
    str(_cfg.test.output.plots_review_dir), plots_folder_name_1in
)
# Create directory if it doesn't exist
os.makedirs(outputdir_plots_1in, exist_ok=True)

plot_axis = True
plot_titles = False
plot_bar = False

# Plot specific patch
# Fig. 12

df_mae_test = plot_input_pred_patch(
    dataloader_pred_1in,
    model_1in,
    opt,
    batch_size,
    True,
    False,
    True,
    outputdir_plots_1in,
    specific_batch=49,
    specific_patch=8,
    box_coords=[40, 25, 24, 20],
)


df_mae_test = plot_input_pred_patch(
    dataloader_pred_1in,
    model_1in,
    opt,
    batch_size,
    True,
    True,
    False,
    outputdir_plots_1in,
    specific_batch=2,
    specific_patch=15,
    box_coords=[25, 15, 39, 10],
)


### Generalisation performance - apply model to unseen data ###

# Prepare dataset of images and mask
imgs_pred_coh = RasterDataset_imgs(
    paths=str(_cfg.test.rome.coherence_dir), transforms=scale_coh
)
imgs_OSM_pred = RasterDataset_imgs(paths=str(_cfg.test.rome.osm_dir))
imgs_pred = imgs_pred_coh & imgs_OSM_pred
msks_pred = RasterDataset_msks(paths=str(_cfg.test.rome.mask_dir))

dataset_to_pred = imgs_pred & msks_pred

print(dataset_to_pred)


# Run interference with on model

experiment_name_pred = experiment_name + "_Rome"
trainer, datamodule_pred = test_pred_setup(
    seed,
    param_const,
    param_opts[0],
    dataset_to_pred,
    experiment_dir,
    experiment_name_pred,
)

prediction_output = trainer.predict(model=model, datamodule=datamodule_pred)

mae_pred = [prediction_output["MAE"] for prediction_output in prediction_output]
stacked_tensor = torch.stack(mae_pred, dim=0)
mean_tensor = torch.mean(stacked_tensor, dim=0)
print("MEA of each batch: ", mae_pred)
print("Overall MAE: ", mean_tensor)

RegressionTask.prediction_epoch_end(prediction_output)


# Run interference with 20 models

# # Iterate throught the results from the twenty different seeds and print the testing results
# for model_seed in models_seeds:
#     model = PixelwiseRegressionTask.load_from_checkpoint(
#         checkpoint_path=os.path.join(best_models_path, "{}".format(model_seed))
#     )

#     seed = int(model_seed.split("_")[1])

#     experiment_name_pred = experiment_name + "_Rome"

#     trainer, datamodule_pred = test_pred_setup(
#         seed,
#         param_const,
#         param_opts[0],
#         dataset_to_pred,
#         experiment_dir,
#         experiment_name_pred,
#     )

#     print("Model from checkpoint: ", model_seed)
#     prediction_output = trainer.predict(model=model, datamodule=datamodule_pred)
#     print(RegressionTask.prediction_epoch_end(prediction_output))


### Generate plots ###
sampler_interf = GridGeoSampler(dataset_to_pred, patch_size, patch_size)

# Create dataloader
dataloader_interf = DataLoader(
    dataset=dataset_to_pred,
    batch_size=batch_size,
    sampler=sampler_interf,
    num_workers=0,
    collate_fn=stack_samples,
    generator=generator,
)

# Generate prediction chips
# disable randomness, dropout, etc...
model.eval()


# Generate geotiff
pixel_size_interf = dataset_to_pred.res
crs_size_interf = dataset_to_pred.crs.to_epsg()

start = timeit.default_timer()  # Measuring the time
chips_generator_interf = georreferenced_chip_generator(
    dataloader_interf, model, crs_size_interf, pixel_size_interf
)
print("The time taken to predict was: ", timeit.default_timer() - start)

# Save chips as geotiff
start = timeit.default_timer()  # Measuring the time
file_name_interf = os.path.join(
    root,
    "merged_prediction_fixed_stiching_Rome_{}.tif".format(
        datetime.now().strftime("%Y%m%d_%H%M%S")
    ),
)

bounds_Rome = (
    12.0999999999999996,
    41.6233300000000028,
    12.953329999999999,
    42.0499999999999972,
)


merge_georeferenced_chips(
    chips_generator_interf, file_name_interf, overlap, bounds_Rome
)

print(
    "The time taken to generate a georrefenced image and save it was: ",
    timeit.default_timer() - start,
)

# Plots for paper
# Read optical data
opt_rome = RasterDatasetOptical_imgs(paths=str(_cfg.test.rome.google_satellite_dir))
print(opt_rome)

outputdir_plots_interf = os.path.join(
    str(_cfg.test.rome.plots_review_dir), plots_folder_name_1in
)
# Create directory if it doesn't exist
os.makedirs(outputdir_plots_interf, exist_ok=True)

# Plot specific patch
df_mae_test = plot_input_pred_patch(
    dataloader_interf,
    model,
    opt_rome,
    batch_size,
    True,
    False,
    True,
    outputdir_plots_interf,
    specific_batch=13,
    specific_patch=11,
)


df_mae_test = plot_input_pred_patch(
    dataloader_interf,
    model,
    opt_rome,
    batch_size,
    True,
    True,
    False,
    outputdir_plots_interf,
    specific_batch=5,
    specific_patch=6,
)


# ### Print model layers
# from skorch import NeuralNetRegressor

# # task = PixelwiseRegressionTask(
# #     model=segm_model,
# #     backbone=backbone,
# #     weights=weights,
# #     in_channels=in_channels,
# #     loss=loss,
# #     ignore_index=ignore_index,
# #     learning_rate=learning_rate,
# #     beta1=beta1,
# #     beta2=beta2,
# #     wd=wd,
# #     learning_rate_schedule_patience=learning_rate_schedule_patience,
# # )

# model = NeuralNetRegressor(module=task)

# print(model.initialize())
