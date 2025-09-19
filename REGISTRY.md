Registries

- Purpose: map string keys in configs to Python classes/factories so components can be declared and constructed dynamically.
- Usage:
  - Register a class:
    from cuvisai_examples.registry import DATASETS
    @DATASETS.register("MyDataset")
    class MyDataset(...):
        ...
  - Build from config:
    from cuvisai_examples.registry import build_from_cfg, DATASETS
    ds = build_from_cfg({"type": "MyDataset", "params": {"arg": 1}}, DATASETS)
- Available registries: DATASETS, MODELS, TRANSFORMS, RUNNERS, EVALUATORS, REPORTERS
