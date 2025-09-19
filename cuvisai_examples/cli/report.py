import hydra
from omegaconf import DictConfig
from cuvisai_examples.registry import EVALUATORS, REPORTERS, build_from_cfg


@hydra.main(version_base=None, config_path="../configs", config_name="report")
def main(cfg: DictConfig):
    evaluator = build_from_cfg(cfg.eval, EVALUATORS)
    metrics = evaluator.evaluate(cfg.inputs)
    reporter = build_from_cfg(cfg.reporter, REPORTERS)
    reporter.save(metrics, cfg.outputs)


if __name__ == "__main__":
    main()
