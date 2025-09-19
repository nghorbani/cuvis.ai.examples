import itertools
import torch
import pytorch_lightning as pl
from torchmetrics.classification import AUROC, ROC, PrecisionRecallCurve, AveragePrecision, Accuracy
from torchmetrics.segmentation import DiceScore, MeanIoU
from cuvisai_examples.registry import MODELS


@MODELS.register("efficientad.MediumLightning")
class EfficientADLightning(pl.LightningModule):
    def __init__(self, backbones_type: str = "efficientad.MediumBackbones", in_channels: int = 6, learning_rate: float = 1e-4, weight_decay: float = 1e-5, checkpoints: str | None = None, use_imgNet_penalty: bool = False):
        super().__init__()
        backbones_cls = MODELS.get(backbones_type)
        self.backbones = backbones_cls(in_channels=in_channels)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_imgNet_penalty = use_imgNet_penalty

        self.student = self.backbones.student
        self.teacher = self.backbones.teacher
        self.ae = self.backbones.ae

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

    def forward(self, batch):
        return {"anomaly_map": torch.zeros((1, 1, batch["image"].shape[-2], batch["image"].shape[-1]), device=self.device)}

    def training_step(self, batch, batch_idx):
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(itertools.chain(self.student.parameters(), self.ae.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
