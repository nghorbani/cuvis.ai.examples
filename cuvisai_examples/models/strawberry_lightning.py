import itertools
import pytorch_lightning as pl
import torchvision.transforms
import torch
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import AveragePrecision, Accuracy, ROC, AUROC
import torch.nn.functional as F

from cuvisai_examples.registry import MODELS
from .strawberry_unet import FreshTwin2DUNet

try:
    from StrawberryClassification.gpu_pca import IncrementalPCAonGPU as IncPca
except Exception:
    IncPca = None


class GeneralizedDiceLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-6, generalize=False, mode=None):
        super().__init__()
        self.epsilon = epsilon
        self.generalize = generalize
        self.mode = mode

    def forward(self, inputs, targets):
        if inputs.dim() != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {inputs.shape}")
        B, C, H, W = inputs.shape
        if targets.shape != inputs.shape:
            targets = F.one_hot(targets.long(), num_classes=C)
            targets = targets.permute(0, 3, 1, 2).float()
        inputs = inputs.float()
        targets = targets.float()
        inputs_flat = inputs.view(B, C, -1)
        targets_flat = targets.view(B, C, -1)
        if self.mode == "advanced":
            gt_sum = targets_flat.sum(-1)
            class_weights = 1.0 / (gt_sum**2 + self.epsilon)
        else:
            class_weights = torch.tensor([0, 1, 1], device=inputs.device)
        intersection = (inputs_flat * targets_flat).sum(-1)
        union = (inputs_flat + targets_flat).sum(-1)
        if self.generalize:
            denominator = (class_weights * union).sum(1)
            numerator = (class_weights * intersection).sum(1)
        else:
            denominator = union.sum(1)
            numerator = intersection.sum(1)
        dice_score = 2 * numerator / (denominator + self.epsilon)
        loss = 1 - dice_score
        return loss.mean()


@MODELS.register("strawberry.Lightning")
class StrawberryLightning(pl.LightningModule):
    def __init__(
        self,
        pca_channels=8,
        cube_channels=224,
        num_classes=4,
        learning_rate=1e-3,
        weight_decay=1e-5,
        image_size=(200, 200),
        data_loader=None,
    ):
        super().__init__()
        self.pca_channels = pca_channels
        self.cube_channels = cube_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.model = FreshTwin2DUNet(
            in_channels=self.pca_channels, num_classes=self.num_classes, pca=None
        )
        self.roc = ROC(task="multiclass", num_classes=self.num_classes)
        self.auroc = AUROC(task="multiclass", num_classes=self.num_classes)
        self.ap = AveragePrecision(task="multiclass", num_classes=self.num_classes)
        self.acc1 = Accuracy(
            task="multiclass", num_classes=self.num_classes, threshold=0.1
        )
        self.acc2 = Accuracy(
            task="multiclass", num_classes=self.num_classes, threshold=0.2
        )
        self.acc3 = Accuracy(
            task="multiclass", num_classes=self.num_classes, threshold=0.3
        )
        self.acc4 = Accuracy(
            task="multiclass", num_classes=self.num_classes, threshold=0.4
        )
        self.acc5 = Accuracy(
            task="multiclass", num_classes=self.num_classes, threshold=0.5
        )
        self.dice = DiceScore(
            num_classes=self.num_classes,
            include_background=False,
            input_format="one-hot",
        )
        self.dice_bg = DiceScore(num_classes=self.num_classes, include_background=True)
        self.dice1 = DiceScore(
            num_classes=self.num_classes,
            include_background=False,
            input_format="one-hot",
        )
        self.dice2 = DiceScore(
            num_classes=self.num_classes,
            include_background=False,
            input_format="one-hot",
        )
        self.dice3 = DiceScore(
            num_classes=self.num_classes,
            include_background=False,
            input_format="one-hot",
        )
        self.dice4 = DiceScore(
            num_classes=self.num_classes,
            include_background=False,
            input_format="one-hot",
        )
        self.dice5 = DiceScore(
            num_classes=self.num_classes,
            include_background=False,
            input_format="one-hot",
        )
        self.iou_bg = MeanIoU(num_classes=self.num_classes, include_background=True)
        self.iou1 = MeanIoU(num_classes=self.num_classes, include_background=False)
        self.iou2 = MeanIoU(num_classes=self.num_classes, include_background=False)
        self.iou3 = MeanIoU(num_classes=self.num_classes, include_background=False)
        self.iou4 = MeanIoU(num_classes=self.num_classes, include_background=False)
        self.iou5 = MeanIoU(num_classes=self.num_classes, include_background=False)
        self.to_PIL = torchvision.transforms.ToPILImage()
        self.val_loss = []
        self.train_loss = []
        self.save_imgs = False
        self.data_loader = data_loader
        self.image_height, self.image_width = image_size
        self.pca = None
        if IncPca is not None:
            try:
                self.pca = IncPca(n_components=self.pca_channels)
                self.model.pca = self.pca
            except Exception:
                self.pca = None

    def setup(self, stage):
        if (
            self.pca is not None
            and getattr(self.pca, "n_samples_seen_", 0) == 0
            and self.data_loader is not None
        ):
            for batch in self.data_loader:
                pca_image = (
                    batch["image"]
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .reshape(-1, self.cube_channels)
                )
                self.pca.partial_fit(pca_image)

    def training_step(self, batch, batch_idx):
        res = self.model.forward(batch["image"])
        pred = torch.softmax(res, dim=0)
        one_hot_gt = torch.nn.functional.one_hot(
            batch["mask"].type(torch.long), num_classes=self.num_classes
        ).movedim(-1, 1)
        loss = GeneralizedDiceLoss(generalize=True)(pred.unsqueeze(0), one_hot_gt)
        self.train_loss.append(loss.item())
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        res = self.model.forward(batch["image"])
        gt_mask = batch["mask"]
        pred = torch.softmax(res, dim=0).unsqueeze(0)
        one_hot_gt = torch.nn.functional.one_hot(
            gt_mask.type(torch.long), num_classes=self.num_classes
        ).movedim(-1, 1)
        loss = GeneralizedDiceLoss(generalize=True)(pred, one_hot_gt).item()
        self.val_loss.append(loss)
        prediction = torch.argmax(res, dim=0, keepdim=True)
        one_hot_pred = torch.nn.functional.one_hot(
            prediction.type(torch.long), num_classes=self.num_classes
        ).movedim(-1, 1)
        self.dice.update(one_hot_pred, one_hot_gt)
        self.dice_bg.update(one_hot_pred > 0.5, one_hot_gt)
        self.dice1.update(one_hot_pred > 0.1, one_hot_gt)
        self.dice2.update(one_hot_pred > 0.2, one_hot_gt)
        self.dice3.update(one_hot_pred > 0.3, one_hot_gt)
        self.dice4.update(one_hot_pred > 0.4, one_hot_gt)
        self.dice5.update(one_hot_pred > 0.5, one_hot_gt)
        self.iou_bg.update(prediction > 0.5, gt_mask)
        self.iou1.update(prediction > 0.1, gt_mask)
        self.iou2.update(prediction > 0.2, gt_mask)
        self.iou3.update(prediction > 0.3, gt_mask)
        self.iou4.update(prediction > 0.4, gt_mask)
        self.iou5.update(prediction > 0.5, gt_mask)
        self.ap.update(res.unsqueeze(0), gt_mask)
        self.acc1.update(res.unsqueeze(0), gt_mask)
        self.acc2.update(res.unsqueeze(0), gt_mask)
        self.acc3.update(res.unsqueeze(0), gt_mask)
        self.acc4.update(res.unsqueeze(0), gt_mask)
        self.acc5.update(res.unsqueeze(0), gt_mask)
        self.roc.update(res.unsqueeze(0), gt_mask)
        self.auroc.update(res.unsqueeze(0), gt_mask)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_im/AP", self.ap, on_epoch=True)
        self.log("val_im/Acc_0.1", self.acc1, on_epoch=True)
        self.log("val_im/Acc_0.2", self.acc2, on_epoch=True)
        self.log("val_im/Acc_0.3", self.acc3, on_epoch=True)
        self.log("val_im/Acc_0.4", self.acc4, on_epoch=True)
        self.log("val_im/Acc_0.5", self.acc5, on_epoch=True)
        self.log("val_IoU/t=0.5_BG", self.iou_bg.compute(), on_epoch=True)
        self.log("val_IoU/t=0.1", self.iou1, on_epoch=True)
        self.log("val_IoU/t=0.2", self.iou2, on_epoch=True)
        self.log("val_IoU/t=0.3", self.iou3, on_epoch=True)
        self.log("val_IoU/t=0.4", self.iou4, on_epoch=True)
        self.log("val_IoU/t=0.5", self.iou5, on_epoch=True)
        self.log("val_F1/dice", self.dice, on_epoch=True)
        self.log("val_F1/t=0.1", self.dice1, on_epoch=True)
        self.log("val_F1/t=0.2", self.dice2, on_epoch=True)
        self.log("val_F1/t=0.3", self.dice3, on_epoch=True)
        self.log("val_F1/t=0.4", self.dice4, on_epoch=True)
        self.log("val_F1/t=0.5", self.dice5, on_epoch=True)
        self.log("val_ROC/AUROC", self.auroc, on_epoch=True)
        self.roc.reset()
        self.val_loss = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            itertools.chain(self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", patience=2
                ),
                "monitor": "train/epoch_loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }
