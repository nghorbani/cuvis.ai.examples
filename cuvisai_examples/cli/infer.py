from dotenv import load_dotenv

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from cuvisai_examples.registry import DATASETS, MODELS, RUNNERS, build_from_cfg

load_dotenv(override=True)


@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig):
    model = build_from_cfg(cfg.model, MODELS)
    if "checkpoint" in cfg and cfg.checkpoint:
        if hasattr(model, "load_from_checkpoint"):
            model = model.load_from_checkpoint(cfg.checkpoint)
    ds = build_from_cfg(cfg.test_dataset, DATASETS)
    loader = DataLoader(
        ds, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.num_workers
    )
    runner = build_from_cfg(cfg.runner, RUNNERS)
    runner.test(model, loader)


if __name__ == "__main__":
    main()
