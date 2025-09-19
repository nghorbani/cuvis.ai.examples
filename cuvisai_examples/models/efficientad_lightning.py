import itertools
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics.classification import AUROC, ROC, PrecisionRecallCurve, AveragePrecision, Accuracy
from torchmetrics.segmentation import DiceScore, MeanIoU
from cuvisai_examples.registry import MODELS


def _reduce_tensor_elems(tensor: torch.Tensor, m: int = 2 ** 24) -> torch.Tensor:
    t = tensor.flatten()
    if t.numel() > m:
        idx = torch.randperm(t.numel(), device=t.device)[:m]
        t = t[idx]
    return t


@MODELS.register("efficientad.MediumLightning")
class EfficientADLightning(pl.LightningModule):
    def __init__(self, backbones_type: str = "efficientad.MediumBackbones", in_channels: int = 6, learning_rate: float = 1e-4, weight_decay: float = 1e-5, checkpoints: str | None = None, use_imgNet_penalty: bool = False, loss: dict | None = None, preprocessing: dict | None = None):
        super().__init__()
        backbones_cls = MODELS.get(backbones_type)
        self.backbones = backbones_cls(in_channels=in_channels)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_imgNet_penalty = use_imgNet_penalty

        loss = loss or {}
        self.st_weight = float(loss.get("st_weight", 1.0))
        self.ae_weight = float(loss.get("ae_weight", 1.0))
        self.imgnet_penalty_weight = float(loss.get("imgnet_penalty_weight", 0.0))

        pre = preprocessing or {}
        self.compute_teacher_stats = bool(pre.get("compute_teacher_stats", True))
        self.compute_percentile_quantiles = bool(pre.get("compute_percentile_quantiles", True))

        self.student = self.backbones.student
        self.teacher = self.backbones.teacher
        self.ae = self.backbones.ae

        self.register_buffer("teacher_mean", torch.zeros(1, 384, 1, 1))
        self.register_buffer("teacher_std", torch.ones(1, 384, 1, 1))
        self.register_buffer("qa_st", torch.tensor(0.0))
        self.register_buffer("qb_st", torch.tensor(0.0))
        self.register_buffer("qa_ae", torch.tensor(0.0))
        self.register_buffer("qb_ae", torch.tensor(0.0))

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

    def _teacher_feats(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t = self.teacher(x)
            t = (t - self.teacher_mean) / (self.teacher_std + 1e-6)
        return t

    def _maps(self, x: torch.Tensor) -> dict:
        img_h, img_w = x.shape[-2:]
        t = self._teacher_feats(x)
        s = self.student(x)
        dist_st = (t - s[:, : t.shape[1], :, :]).pow(2)
        map_st = dist_st.mean(dim=1, keepdim=True)

        recon = self.ae(x, (img_h, img_w))
        dist_ae = (x - recon).pow(2).mean(dim=1, keepdim=True)
        norm_st = map_st / (self.qb_st + 1e-6)
        norm_ae = map_ae = dist_ae / (self.qb_ae + 1e-6)
        anomaly_map = 0.5 * (norm_st + norm_ae)
        return {"map_st": map_st, "map_ae": map_ae, "anomaly_map": anomaly_map}

    def forward(self, batch):
        return self._maps(batch["image"])

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        maps = self._maps(x)
        loss_st = maps["map_st"].mean()
        img_h, img_w = x.shape[-2:]
        recon = self.ae(x, (img_h, img_w))
        loss_ae = (x - recon).pow(2).mean()

        loss = self.st_weight * loss_st + self.ae_weight * loss_ae

        if self.use_imgNet_penalty and "imgNet_img" in batch and batch["imgNet_img"] is not None:
            imgnet = batch["imgNet_img"].to(x.dtype).to(x.device)
            r2 = (imgnet - self.ae(imgnet, imgnet.shape[-2:])).pow(2).mean()
            loss = loss + self.imgnet_penalty_weight * (-r2)

        self.log("train/st", loss_st, on_epoch=True, prog_bar=True)
        self.log("train/ae", loss_ae, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_start(self) -> None:
        if self.compute_teacher_stats:
            self._compute_teacher_mean_std(self.trainer.train_dataloader)

    @torch.no_grad()
    def _compute_teacher_mean_std(self, dl: DataLoader) -> None:
        n = None
        s1 = None
        s2 = None
        for b in dl:
            y = self.teacher(b["image"].to(self.device))
            if n is None:
                c = y.shape[1]
                n = torch.zeros((c,), dtype=torch.int64, device=self.device)
                s1 = torch.zeros((c,), dtype=torch.float32, device=self.device)
                s2 = torch.zeros((c,), dtype=torch.float32, device=self.device)
            n += y[:, 0].numel()
            s1 += y.sum(dim=[0, 2, 3])
            s2 += (y ** 2).sum(dim=[0, 2, 3])
        mean = s1 / n
        var = s2 / n - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        self.teacher_mean = mean.view(1, -1, 1, 1)
        self.teacher_std = std.view(1, -1, 1, 1)

    def on_validation_start(self) -> None:
        if self.compute_percentile_quantiles and self.trainer.val_dataloaders is not None:
            self._compute_quantiles(self.trainer.val_dataloaders)

    @torch.no_grad()
    def _compute_quantiles(self, dl: DataLoader) -> None:
        maps_st = []
        maps_ae = []
        for batch in dl or []:
            for img, label in zip(batch["image"], batch["label"], strict=True):
                if label == 0:
                    res = self._maps(img.unsqueeze(0).to(self.device))
                    maps_st.append(res["map_st"])
                    maps_ae.append(res["map_ae"])
        if len(maps_st) > 0:
            ms = torch.cat(maps_st, dim=0).to(self.device)
            ma = torch.cat(maps_ae, dim=0).to(self.device)
            msf = _reduce_tensor_elems(ms)
            maf = _reduce_tensor_elems(ma)
            self.qa_st = torch.quantile(msf, q=0.9)
            self.qb_st = torch.quantile(msf, q=0.995)
            self.qa_ae = torch.quantile(maf, q=0.9)
            self.qb_ae = torch.quantile(maf, q=0.995)

    def configure_optimizers(self):
        return torch.optim.Adam(itertools.chain(self.student.parameters(), self.ae.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
