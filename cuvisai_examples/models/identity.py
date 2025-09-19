import torch
import torch.nn as nn
import pytorch_lightning as pl
from cuvisai_examples.registry import MODELS


@MODELS.register("IdentityModel")
class IdentityModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Identity()
        self._dummy = nn.Parameter(torch.zeros((), device=self.device if hasattr(self, "device") else None))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = batch["image"].float()
        _ = self(x)
        loss = self._dummy * 0.0
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)
