# cuvisai-examples architecture and extensibility

Overview
- Hydra/OmegaConf configs declare components.
- Registries map string keys to Python classes/factories.
- build_from_cfg(cfg, REG) instantiates cfg.type with cfg.params.

Registries and build flow
- Registries: DATASETS, MODELS, RUNNERS, EVALUATORS, REPORTERS.
- Register:
  from cuvisai_examples.registry import MODELS
  @MODELS.register("my_family.MyModel")
  class MyModel: ...
- Build:
  from cuvisai_examples.registry import build_from_cfg
  obj = build_from_cfg({"type":"my_family.MyModel","params":{"hidden":128}}, MODELS)

Config layout and overrides
- Configs live under cuvisai_examples/configs
  - dataset/, model/, runner/, eval/, reporting/
- Example train defaults:
  defaults:
    - dataset: efficientad_train_val
    - model: efficientad/medium
    - runner: lightning
    - eval: efficientad
    - reporting: efficientad
    - _self_
- Override via CLI:
  uv run cuvisai-train model=efficientad/medium trainer.max_epochs=2
  uv run cuvisai-train datasets.train.params.path=/abs/data datasets.val.params.path=/abs/data

Environment variables
- .env auto-loaded (override=True) via python-dotenv.
- Use ${oc.env:VAR, default} in YAML.
- Common: HF_TOKEN, WORK_DIR, DATA_CUBES_DIR, IMAGENET_DIR.

Adding components

Models
- Lightning recommended.
  from cuvisai_examples.registry import MODELS
  @MODELS.register("my_family.BigNet")
  class BigNet(pl.LightningModule):
      def __init__(self, width=256, loss: dict | None=None): ...
- Config:
  type: my_family.BigNet
  params:
    width: 256
    loss:
      ce_weight: 1.0
      dice_weight: 0.0

Datasets
- PyTorch Dataset registered under DATASETS.
  from cuvisai_examples.registry import DATASETS
  @DATASETS.register("my_data.Train")
  class MyTrain(Dataset): ...
- Train/val config:
  train:
    type: my_data.Train
    params:
      path: ${oc.env:DATA_DIR, ./data}/train
  val:
    type: my_data.Val
    params:
      path: ${oc.env:DATA_DIR, ./data}/val

Runners
- Wrap training engine; default is Lightning.
  from cuvisai_examples.registry import RUNNERS
  @RUNNERS.register("lightning")
  class LightningRunner: ...
- Config:
  type: lightning
  params:
    accelerator: cuda
    devices: 1
    precision: 32
    max_epochs: ${trainer.max_epochs}
    default_root_dir: ${work_dir}

Evaluators and Reporters
- Implement minimal interfaces and register with EVALUATORS/REPORTERS.
- Provide YAML under configs/eval and configs/reporting.

Programmatic usage
from cuvisai_examples.registry import build_from_cfg, MODELS, DATASETS, RUNNERS
from torch.utils.data import DataLoader
model = build_from_cfg(cfg.model, MODELS)
train_ds = build_from_cfg(cfg.datasets.train, DATASETS)
val_ds = build_from_cfg(cfg.datasets.val, DATASETS)
runner = build_from_cfg(cfg.runner, RUNNERS)
runner.fit(model, DataLoader(train_ds, batch_size=4), DataLoader(val_ds, batch_size=4))

Troubleshooting
- Key not found: ensure module import triggers @register call.
- Zero-length DataLoader: verify dataset paths and filters.
- Hydra override errors: check key paths; quote complex values.
