import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from cuvisai_examples.registry import RUNNERS


@RUNNERS.register("LightningRunner")
class LightningRunner:
    def __init__(
        self,
        logger: str = "tensorboard",
        save_dir: str = "./work_dirs",
        checkpoint: dict | None = None,
        **trainer_kwargs,
    ):
        self.logger = (
            TensorBoardLogger(save_dir=save_dir) if logger == "tensorboard" else None
        )
        
        callbacks = []
        if checkpoint:
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint.get("dirpath", save_dir),
                filename=checkpoint.get("filename", "best-{epoch:02d}-{step}"),
                monitor=checkpoint.get("monitor", "val/auroc"),
                mode=checkpoint.get("mode", "max"),
                save_top_k=checkpoint.get("save_top_k", 1),
                save_last=checkpoint.get("save_last", True),
                verbose=checkpoint.get("verbose", True),
            )
            callbacks.append(checkpoint_callback)
        
        self.trainer = pl.Trainer(
            logger=self.logger, 
            callbacks=callbacks,
            **trainer_kwargs
        )

    def fit(self, model, train_loader, val_loader=None):
        self.trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

    def test(self, model, test_loader=None):
        self.trainer.test(model=model, dataloaders=test_loader)
