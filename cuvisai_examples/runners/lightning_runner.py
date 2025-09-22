import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from cuvisai_examples.registry import RUNNERS


@RUNNERS.register("LightningRunner")
class LightningRunner:
    def __init__(
        self,
        logger: str = "tensorboard",
        save_dir: str = "./work_dirs",
        **trainer_kwargs,
    ):
        self.logger = (
            TensorBoardLogger(save_dir=save_dir) if logger == "tensorboard" else None
        )
        self.trainer = pl.Trainer(logger=self.logger, **trainer_kwargs)

    def fit(self, model, train_loader, val_loader=None):
        self.trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

    def test(self, model, test_loader=None):
        self.trainer.test(model=model, dataloaders=test_loader)
