from .registry import Registry
from .build import build_from_cfg

DATASETS = Registry("datasets")
MODELS = Registry("models")
TRANSFORMS = Registry("transforms")
RUNNERS = Registry("runners")
EVALUATORS = Registry("evaluators")
REPORTERS = Registry("reporters")

__all__ = [
    "Registry",
    "build_from_cfg",
    "DATASETS",
    "MODELS",
    "TRANSFORMS",
    "RUNNERS",
    "EVALUATORS",
    "REPORTERS",
]
