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

from ps_density.model_setup import training_setup
from ps_density.paths import load_paths
from ps_density.torchgeo_fun_to_read_input import (
    scale_coh,
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
_cfg = load_paths()
path_prefix = str(_cfg.train.output.training_dir)

experiment_name = _cfg.train.output.experiment_name
experiment_dir = os.path.join(path_prefix, experiment_name)
print(experiment_dir)

os.makedirs(experiment_dir, exist_ok=True)

# Import images and mask and generate dataset for training

# Import images and masks
imgs = RasterDataset_imgs(
    paths=str(_cfg.train.input.coherence_summer_dir), transforms=scale_coh
)

imgs_OSM = RasterDataset_imgs(paths=str(_cfg.train.input.osm_dir))


msks = RasterDataset_msks(paths=str(_cfg.train.input.mask_egms_dir))

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
        new_name = "seed_{}_{}.ckpt".format(
            seed, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        dir_cop = os.path.join(experiment_dir, new_name)

        shutil.copy(trainer.checkpoint_callback.best_model_path, dir_cop)

print("Training done")
