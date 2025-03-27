import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L
from tqdm import tqdm
from torchmetrics.classification import AUROC, ROC, PrecisionRecallCurve, AveragePrecision, Accuracy
from torchmetrics.segmentation import DiceScore, MeanIoU
from EfficientAD_torch_model import EfficientAdModel
from torchvision.transforms.functional import equalize


def reduce_tensor_elems(tensor: torch.Tensor, m: int = 2 ** 24) -> torch.Tensor:
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
        self.model_size = config['Model']['model_size']
        self.channel_size = config['Model']['channel_size']
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay',1e-5)
        self.in_channels = config['Model']['in_channels']
        self.model = EfficientAdModel(384, in_channels=self.in_channels, model_size=config["Model"]["model_size"], use_imgNet_penalty=config["Model"]["use_imgNet_penalty"])
        self.student = self.model.student
        self.student = self.student.cuda()
        self.teacher = self.model.teacher
        self.teacher = self.teacher.cuda()
        self.load_pretrain_teacher()
        self.ae = self.model.ae
        self.labels = []
        self.scores = []
        self.ae = self.ae.cuda()
        self.channel_mean, self.channel_std = None, None
        self.batch_size = config['Model']['batch_size']
        if config["seed"] != "random":
            self.set_seed(config['seed'])
        self.training = True
        self.save_hyperparameters()
        # Metrics
        self.auroc = AUROC(task='binary')
        self.roc = ROC(task='binary')
        self.prc = PrecisionRecallCurve(task='binary')
        self.ap = AveragePrecision(task='binary')
        self.acc1 = Accuracy(task='binary', threshold=0.1)
        self.acc2 = Accuracy(task='binary', threshold=0.2)
        self.acc3 = Accuracy(task='binary', threshold=0.3)
        self.acc4 = Accuracy(task='binary', threshold=0.4)
        self.acc5 = Accuracy(task='binary', threshold=0.5)
        self.dice1 = DiceScore(num_classes=2, include_background=False)
        self.dice2 = DiceScore(num_classes=2, include_background=False)
        self.dice3 = DiceScore(num_classes=2, include_background=False)
        self.dice4 = DiceScore(num_classes=2, include_background=False)
        self.dice5 = DiceScore(num_classes=2, include_background=False)
        self.iou1 = MeanIoU(num_classes=2, include_background=False)
        self.iou2 = MeanIoU(num_classes=2, include_background=False)
        self.iou3 = MeanIoU(num_classes=2, include_background=False)
        self.iou4 = MeanIoU(num_classes=2, include_background=False)
        self.iou5 = MeanIoU(num_classes=2, include_background=False)
        self.dice_bg = DiceScore(num_classes=2, include_background=True)
        self.iou_bg = MeanIoU(num_classes=2, include_background=True)
        self.has_masks = False
        self.images_logged = []

    def training_step(self, batch, batch_idx):
        """Train the model for one step."""
        loss_st, loss_ae, loss_stae = self.model.forward(batch['image'], batch['imgNet_img'])
        loss_total = loss_st + loss_ae + loss_stae
        self.log("train/st", loss_st.item(), on_epoch=True, prog_bar=True)
        self.log("train/ae", loss_ae.item(), on_epoch=True, prog_bar=True)
        self.log("train/stae", loss_stae.item(), on_epoch=True, prog_bar=True)
        self.log("train/loss", loss_total.item(), on_epoch=True, prog_bar=True)

        return loss_total

    def on_validation_start(self) -> None:
        """Calculate the feature map quantiles of the validation dataset and push to the model."""
        map_norm_quantiles = self.map_norm_quantiles(self.trainer.val_dataloaders)
        self.model.quantiles.update(map_norm_quantiles)

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
        amap = res['anomaly_map'].detach().squeeze(0)
        map_st = res['map_st']
        map_ae = res['map_ae']
        score = torch.max(amap)[None]
        label = batch['label']
        gt_mask = batch['mask']
        # Metrics
        self.auroc.update(score, label)
        self.roc.update(score, label)
        self.prc.update(score, label)
        self.ap.update(score, label)
        self.acc1.update(score, label)
        self.acc2.update(score, label)
        self.acc3.update(score, label)
        self.acc4.update(score, label)
        self.acc5.update(score, label)
        # Segmentation Metrics
        # if     INTACT        or      DEFECT       and    MASK EXISTS
        if (label.item() == 0) or (label.item() == 1 and gt_mask.max().item() > 0):
            self.has_masks = True
            self.dice_bg.update(amap > 0.5, gt_mask)
            self.dice1.update(amap > 0.1, gt_mask)
            self.dice2.update(amap > 0.2, gt_mask)
            self.dice3.update(amap > 0.3, gt_mask)
            self.dice4.update(amap > 0.4, gt_mask)
            self.dice5.update(amap > 0.5, gt_mask)
            self.iou_bg.update(amap > 0.5, gt_mask)
            self.iou1.update(amap > 0.1, gt_mask)
            self.iou2.update(amap > 0.2, gt_mask)
            self.iou3.update(amap > 0.3, gt_mask)
            self.iou4.update(amap > 0.4, gt_mask)
            self.iou5.update(amap > 0.5, gt_mask)

        # Image Logging
        image = batch['image'].detach().squeeze().cpu()
        pred = amap.detach().squeeze().cpu()
        # reverte the normalization of the images to display the input image correctly
        image = np.transpose(np.transpose(image, (1, 2, 0)) * np.array(self.config["std"]) + np.array(self.config["mean"]), (2, 0, 1))
        if self.current_epoch == 0 and label.item() == 1 and len(self.images_logged) < 4 and batch["defect"] not in self.images_logged:  # and gt_mask.max() > 0:  # Just once
            if self.in_channels == 6:

                self.logger.experiment.add_image(f'image{batch_idx}_{batch["defect"]}/0_rgb', image[:3].numpy())
                self.logger.experiment.add_image(f'image{batch_idx}_{batch["defect"]}/1_ir', image[3:].numpy())
            else:
                self.logger.experiment.add_image(f'image{batch_idx}_{batch["defect"]}/0_rgb', image.numpy())
        if self.current_epoch % 2 == 0 and label.item() == 1 and len(self.images_logged) < 4 and batch["defect"] not in self.images_logged:  # and gt_mask.max() > 0:  # Every 10 epochs
            self.images_logged.append(batch["defect"])
            gt_mask = gt_mask.squeeze().cpu()

            masks_only = {f'image{batch_idx}_{batch["defect"]}/mask_t/t={t:.1f}': (pred > t).unsqueeze(0).numpy() for t in [0.1, 0.2, 0.3, 0.4, 0.5]}
            masks_quant = {f'image{batch_idx}_{batch["defect"]}/mask_q/q={q:.2f}': (pred > pred.quantile(q)).unsqueeze(0).numpy() for q in [0.9, 0.95, 0.98, 0.99]}
            eq_map = equalize((255.0 * (pred[None] - pred.min()) / (pred.max() - pred.min())).to(torch.uint8))
            self.logger.experiment.add_image(f'image{batch_idx}_{batch["defect"]}/pred_equalized', eq_map.numpy(), global_step=self.global_step)
            self.logger.experiment.add_image(f'image{batch_idx}_{batch["defect"]}/2_pred_raw', pred.unsqueeze(0).numpy(), global_step=self.global_step)
            self.logger.experiment.add_image(f'image{batch_idx}_{batch["defect"]}/3_pred_st', map_st.detach().squeeze(0).cpu().numpy(), global_step=self.global_step)
            self.logger.experiment.add_image(f'image{batch_idx}_{batch["defect"]}/5_pred_ae', map_ae.detach().squeeze(0).cpu().numpy(), global_step=self.global_step)
            self.logger.experiment.add_image(f'image{batch_idx}_{batch["defect"]}/4_gt_mask', gt_mask.unsqueeze(0).numpy(), global_step=self.global_step)
            for k, v in masks_quant.items():
                self.logger.experiment.add_image(k, v, global_step=self.global_step)

            for k, v in masks_only.items():
                self.logger.experiment.add_image(k, v, global_step=self.global_step)

        return res["anomaly_map"]

    def on_validation_epoch_end(self) -> None:
        """Called after every validation epoch. Logs all accumulated data and clears buffers."""
        self.log('val_im/AU-ROC', self.auroc, on_epoch=True, prog_bar=True)
        self.log('val_im/AP', self.ap, on_epoch=True)
        self.log('val_im/Acc_0.1', self.acc1, on_epoch=True)
        self.log('val_im/Acc_0.2', self.acc2, on_epoch=True)
        self.log('val_im/Acc_0.3', self.acc3, on_epoch=True)
        self.log('val_im/Acc_0.4', self.acc4, on_epoch=True)
        self.log('val_im/Acc_0.5', self.acc5, on_epoch=True)
        roc_fig, _ = self.roc.plot(score=True)
        self.logger.experiment.add_figure('curve/ROC', roc_fig, global_step=self.global_step)
        prc_fig, _ = self.prc.plot(score=True)
        self.logger.experiment.add_figure('curve/PrecisionRecall', prc_fig, global_step=self.global_step)
        plt.close(roc_fig)
        plt.close(prc_fig)
        self.roc.reset()
        self.prc.reset()
        if self.has_masks:
            self.log('val_IoU/t=0.5_BG', self.iou_bg.compute(), on_epoch=True)
            self.log('val_IoU/t=0.1', self.iou1, on_epoch=True)
            self.log('val_IoU/t=0.2', self.iou2, on_epoch=True)
            self.log('val_IoU/t=0.3', self.iou3, on_epoch=True)
            self.log('val_IoU/t=0.4', self.iou4, on_epoch=True)
            self.log('val_IoU/t=0.5', self.iou5, on_epoch=True)
            self.log('val_F1/t=0.5_BG', self.dice_bg, on_epoch=True)
            self.log('val_F1/t=0.1', self.dice1, on_epoch=True)
            self.log('val_F1/t=0.2', self.dice2, on_epoch=True)
            self.log('val_F1/t=0.3', self.dice3, on_epoch=True)
            self.log('val_F1/t=0.4', self.dice4, on_epoch=True)
            self.log('val_F1/t=0.5', self.dice5, on_epoch=True)
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

        for batch in tqdm(dataloader, desc="Calculate Validation Dataset Quantiles", position=0, leave=False):
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
        optimizer = torch.optim.Adam(itertools.chain(self.student.parameters(),
                                                     self.ae.parameters()),
                                     lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

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
        maps_flat = reduce_tensor_elems(torch.cat(maps))
        qa = torch.quantile(maps_flat, q=0.9).to(self.device)
        qb = torch.quantile(maps_flat, q=0.995).to(self.device)
        return qa, qb

    def load_pretrain_teacher(self):
        self.teacher.load_state_dict(torch.load(self.config["Model"]["checkpoints"]))
        self.teacher = self.teacher.cuda()
        self.teacher.eval()
        for parameters in self.teacher.parameters():
            parameters.requires_grad = False

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def forward(self, image):
        res = self.model.forward(image["image"], return_all_maps=True)
        res["mask"] = image["mask"]
        res["label"] = image["label"]
        return res

    def on_train_start(self) -> None:
        channel_mean_std = self.teacher_channel_mean_std(self.trainer.train_dataloader)
        self.model.mean_std.update(channel_mean_std)

    @torch.no_grad()
    def teacher_channel_mean_std(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """Calculate channel-wise mean and std of teacher model activations.

        Computes running mean and standard deviation of teacher model feature maps
        over the full dataset.

        Args:
            dataloader (DataLoader): Dataloader for the dataset.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - ``mean``: Channel-wise means of shape ``(1, C, 1, 1)``
                - ``std``: Channel-wise standard deviations of shape
                  ``(1, C, 1, 1)``

        Raises:
            ValueError: If no data is provided (``n`` remains ``None``).
        """
        arrays_defined = False
        n: torch.Tensor | None = None
        chanel_sum: torch.Tensor | None = None
        chanel_sum_sqr: torch.Tensor | None = None

        for batch in tqdm(dataloader, desc="Calculate teacher channel mean & std", position=0, leave=True):
            y = self.model.teacher(batch["image"].to(self.device))
            if not arrays_defined:
                _, num_channels, _, _ = y.shape
                n = torch.zeros((num_channels,), dtype=torch.int64, device=y.device)
                chanel_sum = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                chanel_sum_sqr = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                arrays_defined = True

            n += y[:, 0].numel()
            chanel_sum += torch.sum(y, dim=[0, 2, 3])
            chanel_sum_sqr += torch.sum(y ** 2, dim=[0, 2, 3])

        if n is None:
            msg = "The value of 'n' cannot be None."
            raise ValueError(msg)

        channel_mean = chanel_sum / n

        channel_std = (torch.sqrt((chanel_sum_sqr / n) - (channel_mean ** 2))).float()[None, :, None, None]
        channel_mean = channel_mean.float()[None, :, None, None]

        return {"mean": channel_mean, "std": channel_std}
