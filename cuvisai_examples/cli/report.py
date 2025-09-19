import os
from dotenv import load_dotenv
load_dotenv(override=True)

import hydra
from omegaconf import DictConfig
from cuvisai_examples.registry import EVALUATORS, REPORTERS, build_from_cfg


@hydra.main(version_base=None, config_path="../configs", config_name="report")
def main(cfg: DictConfig):
    evaluator = build_from_cfg(cfg.eval, EVALUATORS)
    metrics = {"status": "ok"}
    reporter = build_from_cfg(cfg.reporting, REPORTERS)
    reporter.save_metrics(metrics)


if __name__ == "__main__":
    main()
