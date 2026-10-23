"""Lightning regression task and TorchGeo datamodule for pixel-wise PS density."""

# Import modules
import os
from typing import Optional, Union, Any, cast

import matplotlib.pyplot as plt

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import segmentation_models_pytorch as smp
import timm

from lightning.pytorch import LightningModule
from lightning.pytorch import seed_everything

from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
)

from torchvision.models._api import WeightsEnum

import kornia.augmentation as K

from torchgeo.datasets import (
    GeoDataset,
    unbind_samples,
)
from torchgeo.datasets.splits import random_bbox_splitting
from torchgeo.datamodules import GeoDataModule
from torchgeo.samplers import (
    GridGeoSampler,
    RandomGeoSampler,
)
from torchgeo.models import FCN, get_weight
from torchgeo.trainers import utils
from torchgeo.transforms import AugmentationSequential

from torch.utils.data import DataLoader, Dataset


# Copied from TorchGeo so that plot functions and other details can be changed.
class RegressionTask(LightningModule):
    """
    This class is a PyTorch Lightning module for training models on regression datasets.
    It supports any available `Timm model <https://huggingface.co/docs/timm/index>`
    as an architecture choice.
    To see a list of available models, you can do:

    .. code-block:: python

            import timm
            print(timm.list_models())
    """

    target_key: str = "label"

    def plot(self, sample):
        """
        This function plots the image, mask and prediction.

        Args:
            sample: the sample to be plotted

        Returns:
            fig: the figure object
        """

        image1 = sample["image"][0]
        mask = sample["mask"]
        prediction = sample["prediction"]

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(3 * 5, 5))
        axs[0].imshow(image1.squeeze().numpy(), cmap="turbo", vmin=0, vmax=1)
        axs[0].axis("off")
        axs[1].imshow(mask, vmin=0, vmax=80)
        axs[1].axis("off")
        axs[2].imshow(prediction.detach().numpy(), vmin=0, vmax=80)
        axs[2].axis("off")

        axs[0].set_title("Image")
        axs[1].set_title("Mask")
        axs[2].set_title("Prediction")

        plt.tight_layout()

        return fig

    def config_model(self) -> None:
        """
        Configures the model based on kwargs parameters.

        Args:
            None

        Returns:
            None
        """
        # Create model
        weights = self.hyperparams["weights"]
        self.model = timm.create_model(
            self.hyperparams["model"],
            num_classes=self.hyperparams["num_classes"],
            in_chans=self.hyperparams["in_channels"],
            pretrained=weights is True,
        )

        # Load weights
        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            self.model = utils.load_state_dict(self.model, state_dict)

        # Freeze backbone and unfreeze classifier head
        if self.hyperparams.get("freeze_backbone", False):
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True

    def config_task(self) -> None:
        """
        Configures the task based on kwargs parameters.

        Args:
            None

        Returns:
            None
        """
        self.config_model()

        self.loss: nn.Module
        if self.hyperparams["loss"] == "mse":
            self.loss = nn.MSELoss()
        elif self.hyperparams["loss"] == "mae":
            self.loss = nn.L1Loss()
        elif self.hyperparams["loss"] == "huber":
            self.loss = nn.HuberLoss(reduction="mean", delta=1.0)
        else:
            raise ValueError(
                f"Loss type '{self.hyperparams['loss']}' is not valid. "
                f"Currently, supports 'mse' or 'mae' loss."
            )

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize a new LightningModule for training simple regression models.

        Keyword Args:
            model: Name of the timm model to use
            weights: Either a weight enum, the string representation of a weight enum,
                True for ImageNet weights, False or None for random weights,
                or the path to a saved model state dict.
            num_outputs: Number of prediction outputs
            in_channels: Number of input channels to model
            learning_rate: Learning rate for optimizer
            lr_patience: Patience for learning rate scheduler
            freeze_backbone: Freeze the backbone network to linear probe
                the regression head. Does not support FCN models.
            freeze_decoder: Freeze the decoder network to linear probe
                the regression head. Does not support FCN models.
                Only applicable to PixelwiseRegressionTask.

        .. versionchanged:: 0.4
            Change regression model support from torchvision.models to timm

        .. versionadded:: 0.5
           The *freeze_backbone* and *freeze_decoder* parameters.
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()
        self.hyperparams = cast(dict[str, Any], self.hparams)
        self.config_task()

        self.train_metrics = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
                #                 "MAPE": MeanAbsolutePercentageError(),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.prediction_metrics = self.train_metrics.clone(prefix="prediction_")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        batch = args[0]
        batch_idx = args[1]

        # Data augumentation
        # see https://github.com/pytorch/vision/issues/566
        # and https://github.com/microsoft/torchgeo/discussions/1417
        transforms = AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation([90, 90], p=0.5),
            data_keys=["image", "mask"],
        )
        batch = transforms(batch)

        x = batch["image"]
        y = batch[self.target_key]
        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        loss: Tensor = self.loss(y_hat, y.to(torch.float))
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True
        )  # logging to TensorBoard
        self.train_metrics(y_hat, y.to(torch.float))

        return loss

    def on_train_epoch_end(self) -> None:
        """Logs epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
        self.train_metrics.reset()

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch[self.target_key]
        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        y_hat_hard = y_hat.squeeze(dim=1)

        if batch_idx < 10:
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)
            summary_writer = self.logger.experiment
            summary_writer.add_figure(
                f"image/val/{batch_idx}", fig, global_step=self.global_step
            )

            plt.close()

        loss: Tensor = self.loss(y_hat, y.to(torch.float))
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.val_metrics(y_hat, y.to(torch.float))

        return loss

    def on_validation_epoch_end(self) -> None:
        """Logs epoch level validation metrics."""
        self.log_dict(
            self.val_metrics.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute test loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch[self.target_key]
        y_hat = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        loss = self.loss(y_hat, y.to(torch.float))
        self.log("test_loss", loss)
        self.test_metrics(y_hat, y.to(torch.float))

        y_hat_hard = y_hat.squeeze(dim=1)

        if batch_idx < 10:
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)
            summary_writer = self.logger.experiment
            summary_writer.add_figure(
                f"image/test/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()

        return loss

    def on_test_epoch_end(self) -> None:
        """Logs epoch level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the predictions.

        Args:
            batch: the output of your DataLoader

        Returns:
            predicted values
        """
        #         batch = args[0]
        #         x = batch["image"]
        #         y_hat: Tensor = self(x)

        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch[self.target_key]
        y_hat: Tensor = self(x)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        mae_calc = nn.L1Loss()
        mae = mae_calc(y_hat, y.to(torch.float))
        self.prediction_metrics(y_hat, y.to(torch.float))
        return {"y_pred": y_hat, "y_true": y, "MAE": mae}

    def prediction_epoch_end(self, outputs):
        y_preds = torch.cat([output["y_pred"] for output in outputs])
        y_trues = torch.cat([output["y_true"] for output in outputs]).to(torch.float)

        mae_calc = nn.L1Loss()
        mae = mae_calc(y_preds, y_trues)

        # Log the metrics
        print(f"Mean Absolute Error: {mae.item()}")

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            learning rate dictionary
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hyperparams["learning_rate"],
            betas=(self.hyperparams["beta1"], self.hyperparams["beta2"]),
            weight_decay=self.hyperparams["wd"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hyperparams["lr_patience"],
                    threshold=self.hyperparams["min_delta"],
                ),
                "monitor": "val_loss",
            },
        }


# Copied from TorchGeo so that plot functions and other details can be changed.
class PixelwiseRegressionTask(RegressionTask):
    """LightningModule for pixelwise regression of images.

    Supports `Segmentation Models Pytorch
    <https://github.com/qubvel/segmentation_models.pytorch>`_
    as an architecture choice in combination with any of these
    `TIMM backbones <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_.

    .. versionadded:: 0.5
    """

    target_key: str = "mask"

    def config_model(self) -> None:
        """Configures the model based on kwargs parameters."""
        weights = self.hyperparams["weights"]

        if self.hyperparams["model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hyperparams["backbone"],
                encoder_weights="imagenet" if weights is True else None,
                in_channels=self.hyperparams["in_channels"],
                classes=1,
                activation="identity",
                #                 decoder_attention_type='scse'
            )
        elif self.hyperparams["model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hyperparams["backbone"],
                encoder_weights="imagenet" if weights is True else None,
                in_channels=self.hyperparams["in_channels"],
                classes=1,
            )
        elif self.hyperparams["model"] == "fcn":
            self.model = FCN(
                in_channels=self.hyperparams["in_channels"],
                classes=1,
                num_filters=self.hyperparams["num_filters"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hyperparams['model']}' is not valid. "
                f"Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if self.hyperparams["model"] != "fcn":
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hyperparams.get("freeze_backbone", False) and self.hyperparams[
            "model"
        ] in ["unet", "deeplabv3+"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hyperparams.get("freeze_decoder", False) and self.hyperparams[
            "model"
        ] in ["unet", "deeplabv3+"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False


class CustomGeoDataModule(GeoDataModule):
    """Train/val/test split of a TorchGeo intersection dataset.

    ``random_bbox_splitting`` keeps 70/15/15 of each GeoTIFF. Training uses
    ``RandomGeoSampler``; validation, test, and predict use ``GridGeoSampler``.
    """

    def __init__(
        self,
        dataset_class: type[GeoDataset],
        dataset_in: Dataset[dict[str, Tensor]],
        batch_size: int = 1,
        patch_size: Union[int, tuple[int, int]] = 64,
        length: Optional[int] = None,
        length_tr: Optional[int] = None,
        length_val: Optional[int] = None,
        length_pred: Optional[int] = None,
        num_workers: int = 0,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Build a datamodule around an existing intersection dataset.

        Args:
            dataset_class: TorchGeo dataset class (unused; the instance is ``dataset_in``).
            dataset_in: Combined image/mask dataset.
            batch_size: Chips per batch.
            patch_size: Chip size in pixels.
            length: Sampler length passed to GeoDataModule.
            length_tr: Number of training chips drawn by RandomGeoSampler.
            length_val: Unused; validation uses a grid sampler.
            length_pred: Unused; predict uses a grid sampler.
            num_workers: DataLoader workers.
            seed: Split and sampler seed.
            **kwargs: Extra GeoDataModule arguments.
        """
        # BaseDataModule is normalizing images in all batches with mean = 0 and std =255.
        # That doesn't make sense when images aren't RGB (i.e. in range 0-255)
        # Thus, change std to 1 to avoid the normalization
        self.std = torch.tensor(1)

        # Define input dataset
        self.dataset_in = dataset_in
        self.seed = seed
        seed_everything(self.seed, workers=True)
        self.generator = torch.Generator().manual_seed(self.seed)

        def on_after_batch_transfer(
            self, batch: dict[str, Tensor], dataloader_idx: int
        ) -> dict[str, Tensor]:
            """Apply batch augmentations to the batch after it is transferred to the device.

            Args:
                batch: A batch of data that needs to be altered or augmented.
                dataloader_idx: The index of the dataloader to which the batch belongs.

            Returns:
                A batch of data.
            """
            if self.trainer:
                if self.trainer.training:
                    split = "train"
                elif self.trainer.validating or self.trainer.sanity_checking:
                    split = "val"
                elif self.trainer.testing:
                    split = "test"
                elif self.trainer.predicting:
                    split = "predict"

                aug = self._valid_attribute(f"{split}_aug", "aug")
                batch = aug(batch)

            return batch

        super().__init__(
            dataset_class, batch_size, patch_size, length, num_workers, **kwargs
        )
        self.length_tr = length_tr
        self.length_val = length_val
        self.length_pred = length_pred

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        #         self.dataset = self.dataset_class(**self.kwargs)
        self.dataset = self.dataset_in

        # random_bbox_splitting: A 0.8/0.2 split will put 80% of each file in one dataset
        # and the other 20% in the other dataset. Every file will occur in both datasets.
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_bbox_splitting(
            self.dataset,
            [0.7, 0.15, 0.15],
            generator=self.generator,
        )

        self.predict_dataset = self.dataset

        # Define samplers used for different sateges. For trainging, validation, and test use random geo sampler,
        # so that the patches are randomly selected from the full spatial extend.
        if stage in ["fit"]:
            self.train_batch_sampler = RandomGeoSampler(
                self.train_dataset, self.patch_size, self.length_tr
            )

        if stage in ["fit", "validate"]:
            #             self.val_sampler = RandomGeoSampler(
            #                 self.val_dataset, self.patch_size, self.length_val
            #             )

            self.val_sampler = GridGeoSampler(
                # Usually the stride should be slightly smaller than the chip size such that each chip
                # has some small overlap with surrounding chips.
                self.val_dataset,
                self.patch_size,
                self.patch_size - 8,
            )

        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                # Usually the stride should be slightly smaller than the chip size such that each chip
                # has some small overlap with surrounding chips.
                self.test_dataset,
                self.patch_size,
                self.patch_size - 8,
            )
        # For prediction, use grid geo sampler?
        if stage in ["predict"]:
            self.predict_sampler = GridGeoSampler(
                self.dataset,
                self.patch_size,
                self.patch_size,
            )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            generator=self.generator,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            generator=self.generator,
        )

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            generator=self.generator,
        )

    def predict_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.batch_size,
            sampler=self.predict_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            generator=self.generator,
            #             drop_last = True
        )
