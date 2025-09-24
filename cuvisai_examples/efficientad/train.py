import argparse
import itertools
import random

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    AUROC,
    ROC,
    Accuracy,
    AveragePrecision,
    PrecisionRecallCurve,
)
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchvision.transforms.functional import equalize
from tqdm import tqdm
import yaml

from cuvisai_examples.efficientad.data import EfficientADCuvisDataset
from cuvisai_examples.efficientad.model import EfficientAdModel


def reduce_tensor_elems(tensor: torch.Tensor, m: int = 2**24) -> torch.Tensor:
    """Reduce tensor elements.

    This function flatten n-dimensional tensors,  selects m elements from it
    and returns the selected elements as tensor. It is used to select
    at most 2**24 for torch.quantile operation, as it is the maximum
    supported number of elements.
    https://github.com/pytorch/pytorch/blob/b9f81a483a7879cd3709fd26bcec5f1ee33577e6/aten/src/ATen/native/Sorting.cpp#L291.

    Args:
        tensor (torch.Tensor): input tensor from which elements are selected
        m (int): number of maximum tensor elements.
            Defaults to ``2**24``

    Returns:
            Tensor: reduced tensor
    """
    tensor = torch.flatten(tensor)
    if len(tensor) > m:
        # select a random subset with m elements.
        perm = torch.randperm(len(tensor), device="cpu")
        idx = perm[:m]
        tensor = tensor[idx]
    return tensor


class EfficientAD_lightning(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_size = config["model"]["model_size"]
        self.channel_size = config["model"]["channel_size"]
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.in_channels = config["model"]["in_channels"]
        self.model = EfficientAdModel(
            teacher_out_channels=384,
            in_channels=self.in_channels,
            model_size=config["model"]["model_size"],
            use_imgnet_penalty=config["model"]["use_imgnet_penalty"],
        )
        self.student = self.model.student
        self.teacher = self.model.teacher
        self.autoencoder = self.model.autoencoder

        self.load_pretrain_teacher()

        if config["seed"] != "random":
            self.set_seed(config["seed"])
        # self.training = True
        self.save_hyperparameters()
        # Metrics
        self.auroc = AUROC(task="binary")
        self.roc = ROC(task="binary")
        self.prc = PrecisionRecallCurve(task="binary")
        self.ap = AveragePrecision(task="binary")
        # Dictionary-based metrics for better organization
        self.thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Use attributes for thresholded metrics to avoid device issues
        for t in self.thresholds:
            setattr(self, f"acc_at_{t}", Accuracy(task="binary", threshold=t))
            setattr(self, f"dice_at_{t}", DiceScore(num_classes=2, include_background=False))
            setattr(self, f"iou_at_{t}", MeanIoU(num_classes=2, include_background=False))
        self.dice_bg = DiceScore(num_classes=2, include_background=True)
        self.iou_bg = MeanIoU(num_classes=2, include_background=True)
        self.has_masks = False

        self.labels = []
        self.scores = []
        self.images_logged = []

    # ---- Device-aware, called once per stage ----
    def setup(self, stage: str):
        self.teacher.eval()

        for p in self.teacher.parameters():
            p.requires_grad = False

    def training_step(self, batch, batch_idx):
        """Train the model for one step."""
        loss_st, loss_ae, loss_stae = self.model.forward(
            image_batch=batch["image"], imagenet_batch=batch["imgnet_img"], image_ae_aug=batch["image_ae"]
        )
        loss_total = loss_st + loss_ae + loss_stae
        self.log("train/st", loss_st.item(), on_epoch=True, prog_bar=True)
        self.log("train/ae", loss_ae.item(), on_epoch=True, prog_bar=True)
        self.log("train/stae", loss_stae.item(), on_epoch=True, prog_bar=True)
        self.log("train/loss", loss_total.item(), on_epoch=True, prog_bar=True)

        return loss_total

    def on_validation_start(self) -> None:
        """Calculate the feature map quantiles of the validation dataset and push to the model."""
        self.teacher.eval()
        map_norm_quantiles = self.map_norm_quantiles(self.trainer.val_dataloaders)
        # Model exposes set_quantiles(...) — use it to set computed quantiles.
        self.model.set_quantiles(
            map_norm_quantiles["qa_st"],
            map_norm_quantiles["qb_st"],
            map_norm_quantiles["qa_ae"],
            map_norm_quantiles["qb_ae"],
        )

    def on_test_start(self):
        self.teacher.eval()

    def validation_step(self, batch: dict[str, str | torch.Tensor], batch_idx, *args, **kwargs):
        """Perform the validation step of EfficientAd returns anomaly maps for the input image batch.

        Args:
          batch (dict[str, str | torch.Tensor]): Input batch
          args: Additional arguments.
          kwargs: Additional keyword arguments.

        Returns:
          Dictionary containing anomaly maps.
        """
        del args, kwargs  # These variables are not used.

        res = self.model(batch["image"], return_all_maps=True)

        amap = res["anomaly_map"].detach().squeeze(0)
        map_st = res["map_st"]
        map_ae = res["map_ae"]
        score = torch.max(amap)[None]
        label = batch["label"]
        gt_mask = batch["mask"]
        # Metrics
        self.auroc.update(score, label)
        self.roc.update(score, label)
        self.prc.update(score, label)
        self.ap.update(score, label)
        # Update attribute-based accuracy metrics
        for threshold in self.thresholds:
            metric = getattr(self, f"acc_at_{threshold}")
            metric.update(score, label)

        # Segmentation Metrics
        # if     INTACT        or      DEFECT       and    MASK EXISTS
        if (label.item() == 0) or (label.item() == 1 and gt_mask.max().item() > 0):
            self.has_masks = True

            # Update background metrics
            pred_bg_formatted, target_bg_formatted = self._format_for_dice_score(amap > 0.5, gt_mask)
            self.dice_bg.update(pred_bg_formatted, target_bg_formatted)
            self.iou_bg.update(pred_bg_formatted, target_bg_formatted)

            # Update attribute-based segmentation metrics
            for threshold in self.thresholds:
                pred_formatted, target_formatted = self._format_for_dice_score(amap > threshold, gt_mask)
                getattr(self, f"dice_at_{threshold}").update(pred_formatted, target_formatted)
                getattr(self, f"iou_at_{threshold}").update(pred_formatted, target_formatted)

        # Image Logging
        image = batch["image"].detach().squeeze().cpu()
        pred = amap.detach().squeeze().cpu()
        # reverte the normalization of the images to display the input image correctly
        image = np.transpose(
            np.transpose(image, (1, 2, 0)) * np.array(self.config["std"]) + np.array(self.config["mean"]),
            (2, 0, 1),
        )
        if (
            self.current_epoch == 0
            and label.item() == 1
            and len(self.images_logged) < 4
            and batch["defect"] not in self.images_logged
        ):  # and gt_mask.max() > 0:  # Just once
            if self.in_channels == 6:
                self.logger.experiment.add_image(f"image{batch_idx}_{batch['defect']}/0_rgb", image[:3].numpy())
                self.logger.experiment.add_image(f"image{batch_idx}_{batch['defect']}/1_ir", image[3:].numpy())
            else:
                self.logger.experiment.add_image(f"image{batch_idx}_{batch['defect']}/0_rgb", image.numpy())
        if (
            self.current_epoch % 2 == 0
            and label.item() == 1
            and len(self.images_logged) < 4
            and batch["defect"] not in self.images_logged
        ):  # and gt_mask.max() > 0:  # Every 10 epochs
            self.images_logged.append(batch["defect"])
            gt_mask = gt_mask.squeeze().cpu()

            masks_only = {
                f"image{batch_idx}_{batch['defect']}/mask_t/t={t:.1f}": (pred > t).unsqueeze(0).numpy()
                for t in [0.1, 0.2, 0.3, 0.4, 0.5]
            }
            masks_quant = {
                f"image{batch_idx}_{batch['defect']}/mask_q/q={q:.2f}": (pred > pred.quantile(q)).unsqueeze(0).numpy()
                for q in [0.9, 0.95, 0.98, 0.99]
            }
            eq_map = equalize((255.0 * (pred[None] - pred.min()) / (pred.max() - pred.min())).to(torch.uint8))
            self.logger.experiment.add_image(
                f"image{batch_idx}_{batch['defect']}/pred_equalized",
                eq_map.numpy(),
                global_step=self.global_step,
            )
            self.logger.experiment.add_image(
                f"image{batch_idx}_{batch['defect']}/2_pred_raw",
                pred.unsqueeze(0).numpy(),
                global_step=self.global_step,
            )
            self.logger.experiment.add_image(
                f"image{batch_idx}_{batch['defect']}/3_pred_st",
                map_st.detach().squeeze(0).cpu().numpy(),
                global_step=self.global_step,
            )
            self.logger.experiment.add_image(
                f"image{batch_idx}_{batch['defect']}/5_pred_ae",
                map_ae.detach().squeeze(0).cpu().numpy(),
                global_step=self.global_step,
            )
            self.logger.experiment.add_image(
                f"image{batch_idx}_{batch['defect']}/4_gt_mask",
                gt_mask.unsqueeze(0).numpy(),
                global_step=self.global_step,
            )
            for k, v in masks_quant.items():
                self.logger.experiment.add_image(k, v, global_step=self.global_step)

            for k, v in masks_only.items():
                self.logger.experiment.add_image(k, v, global_step=self.global_step)

        return res["anomaly_map"]

    def on_validation_epoch_end(self) -> None:
        """Called after every validation epoch. Logs all accumulated data and clears buffers."""
        self.log("val_im/AU-ROC", self.auroc, on_epoch=True, prog_bar=True)
        self.log("val_im/AP", self.ap, on_epoch=True)
        # Log attribute-based accuracy metrics
        for threshold in self.thresholds:
            metric = getattr(self, f"acc_at_{threshold}")
            self.log(f"val_im/Acc_{threshold}", metric, on_epoch=True)
        roc_fig, _ = self.roc.plot(score=True)
        self.logger.experiment.add_figure("curve/ROC", roc_fig, global_step=self.global_step)
        prc_fig, _ = self.prc.plot(score=True)
        self.logger.experiment.add_figure("curve/PrecisionRecall", prc_fig, global_step=self.global_step)
        plt.close(roc_fig)
        plt.close(prc_fig)
        self.roc.reset()
        self.prc.reset()
        if self.has_masks:
            # Log attribute-based segmentation metrics
            self.log("val_IoU/t=0.5_BG", self.iou_bg.compute(), on_epoch=True)
            self.log("val_F1/t=0.5_BG", self.dice_bg, on_epoch=True)

            for threshold in self.thresholds:
                iou_metric = getattr(self, f"iou_at_{threshold}")
                dice_metric = getattr(self, f"dice_at_{threshold}")
                self.log(f"val_IoU/t={threshold}", iou_metric, on_epoch=True)
                self.log(f"val_F1/t={threshold}", dice_metric, on_epoch=True)
        self.images_logged = []

    @torch.no_grad()
    def map_norm_quantiles(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """Calculate 90% and 99.5% quantiles of the student(st) and autoencoder(ae).

        Args:
            dataloader (DataLoader): Dataloader of the respective dataset.

        Returns:
            dict[str, torch.Tensor]: Dictionary of both the 90% and 99.5% quantiles
            of both the student and autoencoder feature maps.
        """
        maps_st = []
        maps_ae = []

        for batch in tqdm(
            dataloader,
            desc="Calculate Validation Dataset Quantiles",
            position=0,
            leave=False,
        ):
            for img, label in zip(batch["image"], batch["label"], strict=True):
                if label == 0:  # only use good images of validation set!
                    output = self.model(img.to(self.device), normalize=False, return_all_maps=True)
                    map_st = output["map_st"]
                    map_ae = output["map_ae"]
                    maps_st.append(map_st)
                    maps_ae.append(map_ae)

        qa_st, qb_st = self._get_quantiles_of_maps(maps_st)
        qa_ae, qb_ae = self._get_quantiles_of_maps(maps_ae)
        return {"qa_st": qa_st, "qa_ae": qa_ae, "qb_st": qb_st, "qb_ae": qb_ae}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            itertools.chain(self.student.parameters(), self.autoencoder.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # # Add learning rate scheduler as per implementation_plan_II
        # def lr_lambda(step):
        #     total_steps = self.config.get("max_steps", 70000)
        #     threshold = int(total_steps * self.config.get("lr_reduce_at_pct", 0.95))
        #     return 0.1 if step >= threshold else 1.0

        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #         "frequency": 1,
        #     },
        # }
        return optimizer

    def _format_for_dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Format tensors for DiceScore metric.

        DiceScore expects binary masks with proper shape for binary segmentation.
        The prediction should be converted to int8/long type and target should be int.

        Args:
            pred (torch.Tensor): Prediction tensor (binary mask)
            target (torch.Tensor): Target tensor (ground truth mask)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Formatted prediction and target tensors
        """
        # Ensure pred is in correct format for DiceScore
        # DiceScore expects int type tensors for binary segmentation
        pred_formatted = pred.int()
        target_formatted = target.int()

        return pred_formatted, target_formatted

    def _get_quantiles_of_maps(self, maps: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate 90% and 99.5% quantiles of the given anomaly maps.

        If the total number of elements in the given maps is larger than 16777216
        the returned quantiles are computed on a random subset of the given
        elements.

        Args:
            maps (list[torch.Tensor]): List of anomaly maps.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Two scalars - the 90% and the 99.5% quantile.
        """
        assert len(maps) > 0, "The list of maps must not be empty."
        maps_flat = reduce_tensor_elems(torch.cat(maps))
        qa = torch.quantile(maps_flat, q=0.9).to(self.device)
        qb = torch.quantile(maps_flat, q=0.995).to(self.device)
        return qa, qb

    def load_pretrain_teacher(self):
        # map_location="cpu" avoids device mismatch at load time
        state = torch.load(self.config["model"]["checkpoints"], map_location="cpu")
        self.teacher.load_state_dict(state)

        # Don’t move to CUDA manually; Lightning will do model.to(device).
        # Don’t set eval here permanently; do it in forward/hooks.

        # Freeze params
        for p in self.teacher.parameters():
            p.requires_grad = False

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def forward(self, image):
        res = self.model.forward(image["image"], return_all_maps=True)
        res["mask"] = image["mask"]
        res["label"] = image["label"]
        return res

    def on_train_epoch_start(self):
        self.teacher.eval()

    def on_train_start(self) -> None:
        self.teacher.eval()

        channel_mean_std = self.teacher_channel_mean_std(self.trainer.train_dataloader)
        self.model.set_teacher_stats(channel_mean_std["mean"], channel_mean_std["std"])

    @torch.no_grad()
    def teacher_channel_mean_std(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """
        Compute per-channel mean and std of teacher feature maps over the dataset.
        Returns:
            {"mean": (1, C, 1, 1), "std": (1, C, 1, 1)}  on self.device
        """
        # teacher should always be in eval mode
        self.teacher.eval()

        channel_sum = None  # (C,)
        channel_sum_sqr = None  # (C,)
        count = 0  # scalar

        for batch in tqdm(dataloader, desc="Calculate teacher channel mean & std", position=0, leave=True):
            imgs = batch["image"].to(self.device, non_blocking=True)
            y = self.model.teacher(imgs)  # (B, C, H, W)
            y = y.float()  # ensure fp32 for stable squares

            # initialize accumulators
            if channel_sum is None:
                _, num_channels, _, _ = y.shape
                channel_sum = torch.zeros(num_channels, dtype=torch.float64, device=y.device)
                channel_sum_sqr = torch.zeros(num_channels, dtype=torch.float64, device=y.device)

            # per-batch reductions
            channel_sum += y.sum(dim=(0, 2, 3)).to(torch.float64)
            channel_sum_sqr += (y * y).sum(dim=(0, 2, 3)).to(torch.float64)
            count += y.shape[0] * y.shape[2] * y.shape[3]

        if count == 0:
            raise ValueError("Empty dataloader: no images to compute statistics.")

        # mean and std per channel
        channel_mean = (channel_sum / count).to(torch.float32)  # (C,)
        channel_var = (channel_sum_sqr / count - channel_mean.double() ** 2).clamp_min(1e-12)
        channel_std = channel_var.sqrt().to(torch.float32)  # (C,)

        # reshape to (1, C, 1, 1)
        channel_mean = channel_mean[None, :, None, None]
        channel_std = channel_std[None, :, None, None]

        return {"mean": channel_mean, "std": channel_std}


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: set train batch size to 1, val batch size to 2, workers to 0",
    )
    args = parser.parse_args()
    return args


def parse_args(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return config


def train(config):
    enable_debug = config.get("debug", False)

    train_data = EfficientADCuvisDataset(
        config["datasets"]["train"]["root"],
        mode="train",
        imagenet_dir=config["datasets"]["imagenet"]["root"],
        imagenet_file_ending=config["datasets"]["imagenet"]["file_ending"],
        mean=config["mean"],
        std=config["std"],
        normalize=config["normalize"],
        max_img_shape=config["max_img_shape"],
        white_percentage=config["white_percentage"],
        channels=config["channels"],
        max_data_load=2 if enable_debug else -1,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config["model"]["batch_size"],
        shuffle=True,
        num_workers=0 if enable_debug else 4,
        persistent_workers=False if enable_debug else True,
    )

    test_data = EfficientADCuvisDataset(
        config["datasets"]["eval"]["root"],
        mode="test",
        imagenet_dir=config["datasets"]["imagenet"]["root"],
        imagenet_file_ending=config["datasets"]["imagenet"]["file_ending"],
        mean=config["mean"],
        std=config["std"],
        normalize=config["normalize"],
        max_img_shape=config["max_img_shape"],
        white_percentage=config["white_percentage"],
        channels=config["channels"],
        max_data_load=2 if enable_debug else -1,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=config["model"]["batch_size"],
        shuffle=False,
        num_workers=0 if enable_debug else 4,
    )

    # create custom callback to save a model checkpoint for every epoch
    checkpoint_callback = ModelCheckpoint(
        monitor="val_im/AU-ROC",  # Metric to monitor
        dirpath=config["ckpt_dir"] + "/" + config["name"],  # Directory to save checkpoints
        filename=config["name"] + "-{epoch:02d}-{val_im/AU-ROC:.2f}",  # Filename format
        save_top_k=-1,  # Save all checkpoints
        mode="max",
        verbose=True,
    )

    logger = TensorBoardLogger(save_dir=config["logger_dir"], log_graph=True, name=config["name"])
    if "ckpt" in config:
        model = EfficientAD_lightning.load_from_checkpoint(config["ckpt"], config=config)
    else:
        model = EfficientAD_lightning(config)
    trainer = L.Trainer(
        logger=logger,
        max_steps=config["max_steps"],
        benchmark=False if enable_debug else True,
        precision="16-mixed",
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback],
        limit_train_batches=1 if enable_debug else 1.0,
        limit_val_batches=2 if enable_debug else 1.0,
        fast_dev_run=True if enable_debug else False,
    )

    trainer.fit(model, train_loader, test_loader)


def main():
    args = get_arguments()
    config = parse_args(args)
    config["debug"] = args.debug
    train(config)


if __name__ == "__main__":
    main()
