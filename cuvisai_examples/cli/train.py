import os
from dotenv import load_dotenv

import logging
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from cuvisai_examples.registry import DATASETS, MODELS, RUNNERS, build_from_cfg

load_dotenv(override=True)


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
    val_ds = (
        build_from_cfg(cfg.datasets.val, DATASETS) if "val" in cfg.datasets else None
    )

    def _split_info(ds):
        p = getattr(ds, "dataset_dir", None)
        n_npz = len(getattr(ds, "npz_paths", [])) if hasattr(ds, "npz_paths") else None
        n = len(ds)
        return p, n_npz, n

    t_dir, t_npz, t_n = _split_info(train_ds)
    logging.info(
        f"Train dir={os.path.abspath(t_dir) if t_dir else 'N/A'} npz_files={t_npz if t_npz is not None else 'N/A'} size={t_n}"
    )

    if val_ds is not None:
        v_dir, v_npz, v_n = _split_info(val_ds)
        logging.info(
            f"Val   dir={os.path.abspath(v_dir) if v_dir else 'N/A'} npz_files={v_npz if v_npz is not None else 'N/A'} size={v_n}"
        )
        try:
            if t_dir and v_dir and os.path.abspath(t_dir) == os.path.abspath(v_dir):
                logging.warning(
                    "Train and Val directories are identical; splits may be overlapping and metrics biased."
                )
        except Exception:
            pass

    shuffle = len(train_ds) > 0
    logging.info(
        f"Train dataset size={len(train_ds)} batch_size={cfg.dataloader.batch_size} num_workers={cfg.dataloader.num_workers} shuffle={shuffle}"
    )
    if val_ds is not None:
        logging.info(
            f"Val dataset size={len(val_ds)} batch_size={cfg.dataloader.batch_size} num_workers={cfg.dataloader.num_workers}"
        )
    if len(train_ds) == 0:
        logging.warning(
            "Train dataset is empty; training will no-op. Check dataset dirs and filters."
        )

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
