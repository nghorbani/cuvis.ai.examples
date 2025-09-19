import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from cuvisai_examples.registry import DATASETS, MODELS, RUNNERS, build_from_cfg


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    model = build_from_cfg(cfg.model, MODELS)
    train_ds = build_from_cfg(cfg.datasets.train, DATASETS)
    val_ds = build_from_cfg(cfg.datasets.val, DATASETS) if "val" in cfg.datasets else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.dataloader.batch_size,
            num_workers=cfg.dataloader.num_workers,
        )

    runner = build_from_cfg(cfg.runner, RUNNERS)
    runner.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
