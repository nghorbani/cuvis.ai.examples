import torch
import pytorch_lightning as pl
from torch import nn
from cuvisai_examples.registry import MODELS

@MODELS.register("strawberry.LightningStub")
class StrawberryLightningStub(pl.LightningModule):
    def __init__(self, in_channels: int = 10, num_classes: int = 4, lr: float = 1e-3, weight_decay: float = 0.0):
        super().__init__()
        self.net = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = batch["image"].float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        y = torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.long, device=x.device)
        logits = self(x)
        if logits.dim() == 3:
            logits = logits.unsqueeze(0)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"].float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        y = torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.long, device=x.device)
        logits = self(x)
        if logits.dim() == 3:
            logits = logits.unsqueeze(0)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
