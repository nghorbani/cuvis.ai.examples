import os
from dotenv import load_dotenv
load_dotenv(override=True)

import logging
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from cuvisai_examples.registry import DATASETS, MODELS, RUNNERS, build_from_cfg


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    level_name = str(getattr(cfg, "log_level", "INFO"))
    if hasattr(cfg, "logging") and getattr(cfg.logging, "verbose", False):
        level_name = "DEBUG"
    level = getattr(logging, level_name.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt)

    os.makedirs(cfg.work_dir, exist_ok=True)
    log_file = os.path.join(cfg.work_dir, "train.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)
    logging.info(f"Logging initialized at level={level_name.upper()} | file={log_file}")

    model = build_from_cfg(cfg.model, MODELS)
    train_ds = build_from_cfg(cfg.datasets.train, DATASETS)
    val_ds = build_from_cfg(cfg.datasets.val, DATASETS) if "val" in cfg.datasets else None

    shuffle = len(train_ds) > 0
    logging.info(f"Train dataset size={len(train_ds)} batch_size={cfg.dataloader.batch_size} num_workers={cfg.dataloader.num_workers} shuffle={shuffle}")
    if val_ds is not None:
        logging.info(f"Val dataset size={len(val_ds)} batch_size={cfg.dataloader.batch_size} num_workers={cfg.dataloader.num_workers}")
    if len(train_ds) == 0:
        logging.warning("Train dataset is empty; training will no-op. Check dataset paths and filters.")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=shuffle,
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
