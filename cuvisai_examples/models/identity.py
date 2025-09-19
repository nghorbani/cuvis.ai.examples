import torch
import torch.nn as nn
import pytorch_lightning as pl
from cuvisai_examples.registry import MODELS


@MODELS.register("IdentityModel")
class IdentityModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Identity()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = batch["image"].float()
        _ = self(x)
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)
