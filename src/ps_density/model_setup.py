"""
This script contains functions for setting up the training, testing 
and prediction of the model.
"""

# Import modules
import os

import torch
from torch import Tensor

from lightning.pytorch import seed_everything

from torchgeo.datasets import (
    stack_samples,
)

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

from torch.utils.data import Dataset

import datetime

from helper_functions.torchgeo_classes_def import PixelwiseRegressionTask, CustomGeoDataModule


### Define functions for training, testing and prediction set-up ###
def general_setup(
    seed: int = 0,
    dataset: Dataset[dict[str, Tensor]] = None,
    batch_size: int = 16,
    patch_size: int = 64,
    nb_btch_epch: int = None,
    num_dataloader_workers: int = 0,
    experiment_dir: str = None,
    experiment_name: str = None,
    min_delta: float = 0.0,
    patience: int = 0,
    max_epochs: int = 1,
    gpu_id=0,
):
    """
    The function sets up the general training/testing/prediction environment.
    It sets up the datamodule, model checkpoint, early stopping, learning rate

    Arguments:
        seed (int): seed for reproducibility
        dataset (Dataset[dict[str, Tensor]]): dataset to be used
        batch_size (int): batch size
        patch_size (int): patch size
        nb_btch_epch (int): number of batches per epoch
        num_dataloader_workers (int): number of workers for dataloader
        experiment_dir (str): directory for storing the experiment
        experiment_name (str): name of the experiment
        min_delta (float): minimum delta for early stopping
        patience (int): patience for early stopping
        max_epochs (int): maximum number of epochs
        gpu_id (int): GPU id

    Returns:
        trainer (pl.Trainer): trainer object
        datamodule (CustomGeoDataModule): datamodule object
    """

    seed_everything(seed, workers=True)

    datamodule = CustomGeoDataModule(
        dataset_class=type(dataset),
        dataset_in=dataset,
        batch_size=batch_size,
        patch_size=patch_size,
        length=nb_btch_epch,
        num_workers=num_dataloader_workers,
        collate_fn=stack_samples,
        seed=seed,  # pass seed for generator needed for splitting into t/v/t
    )

    # Set up Model Checkpoint so that the last and the best is stored
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=experiment_dir,
        save_top_k=1,
        save_last=True,
    )

    # Set up early stopping
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=min_delta,
        patience=patience,
    )

    # Set up learning rate monitor to report every epoch
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Create directory for storing tensorboard logs
    logger_dir = (
        experiment_name
        + "_"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        + "_"
        + "seed"
        + str(seed)
    )
    os.mkdir(os.path.join(".", "logs", logger_dir))

    # Set up tensorboard logger. Set default_hp_metric to false so that metrics defined
    # in the semantic segmentation task (Acc and Jacc) are logged
    tb_logger = TensorBoardLogger(
        save_dir="../logs/", name=logger_dir, default_hp_metric=False
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=[tb_logger],
        default_root_dir=experiment_dir,
        min_epochs=3,
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[gpu_id],
        precision="16-mixed",  # to set up mixed precision
        #         log_every_n_steps=patience,
        deterministic=True,
    )

    return trainer, datamodule


def training_setup(
    seed: int = 0,
    param_const: list = [],
    param_opts: list = [],
    dataset: Dataset[dict[str, Tensor]] = None,
    experiment_dir: str = "",
    experiment_name: str = "",
) -> tuple[pl.Trainer, PixelwiseRegressionTask, CustomGeoDataModule]:
    """
    The function sets up the training environment.

    Parameters:
        seed (int): seed for reproducibility
        param_const (list): list of constants
        param_opts (list): list of options
        dataset (Dataset[dict[str, Tensor]]): dataset to be used
        experiment_dir (str): directory for storing the experiment
        experiment_name (str): name of the experiment

    Returns:
        trainer (pl.Trainer): trainer object
        task (PixelwiseRegressionTask): task object
        datamodule (CustomGeoDataModule): datamodule object
    """

    seed_everything(seed, workers=True)

    # free CUDA memory
    torch.cuda.empty_cache()

    (
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
    ) = param_const

    (
        batch_size,
        patch_size,
        nb_btch_epch,
        backbone,
        weights,
        loss,
        learning_rate,
        beta1,
        beta2,
        wd,
    ) = param_opts

    trainer, datamodule = general_setup(
        seed,
        dataset,
        batch_size,
        patch_size,
        nb_btch_epch,
        num_dataloader_workers,
        experiment_dir,
        experiment_name,
        min_delta,
        patience,
        max_epochs,
        gpu_id,
    )

    # Set up the task with paramters
    task = PixelwiseRegressionTask(
        model="unet",
        backbone=backbone,
        weights=weights,
        in_channels=in_channels,
        loss=loss,
        ignore_index=ignore_index,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        wd=wd,
        lr_patience=lr_patience,
        min_delta=min_delta_lr,
    )

    return trainer, task, datamodule


def test_pred_setup(
    seed: int = 0,
    param_const: list = [],
    param_opts: list = [],
    dataset: Dataset[dict[str, Tensor]] = None,
    experiment_dir: str = "",
    experiment_name: str = "",
) -> tuple[pl.Trainer, CustomGeoDataModule]:
    """
    This function sets up the testing and prediction environment.

    Arguments:
        seed (int): seed for reproducibility
        param_const (list): list of constants
        param_opts (list): list of options
        dataset (Dataset[dict[str, Tensor]]): dataset to be used
        experiment_dir (str): directory for storing the experiment
        experiment_name (str): name of the experiment

    Returns:
        trainer (pl.Trainer): trainer object
        datamodule (CustomGeoDataModule): datamodule object
    """

    seed_everything(seed, workers=True)

    # free CUDA memory
    torch.cuda.empty_cache()

    (
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
    ) = param_const

    (
        batch_size,
        patch_size,
        nb_btch_epch,
        backbone,
        weights,
        loss,
        learning_rate,
        beta1,
        beta2,
        wd,
    ) = param_opts

    trainer, datamodule = general_setup(
        seed,
        dataset,
        batch_size,
        patch_size,
        nb_btch_epch,
        num_dataloader_workers,
        experiment_dir,
        experiment_name,
        min_delta,
        patience,
        max_epochs,
        gpu_id,
    )

    return trainer, datamodule
