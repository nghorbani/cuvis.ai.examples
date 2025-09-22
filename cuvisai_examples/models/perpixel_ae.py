import torch
import torch.nn as nn
import pytorch_lightning as pl
from cuvisai_examples.registry import MODELS


@MODELS.register("perpixel_ae.Autoencoder")
class PerPixelAutoencoder(pl.LightningModule):
    def __init__(
        self,
        in_features: int = 6,
        hidden_dims=(64, 32),
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        dims = [
            in_features,
            *hidden_dims,
            hidden_dims[-1],
            *reversed(hidden_dims[:-1]),
            in_features,
        ]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = batch["image"].float()
        recon = self(x)
        loss = torch.nn.functional.mse_loss(recon, x)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"].float()
        recon = self(x)
        loss = torch.nn.functional.mse_loss(recon, x)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
