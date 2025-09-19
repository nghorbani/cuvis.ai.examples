# Migration guide

What changed
- Installable library with CLIs:
  - cuvisai-train, cuvisai-infer, cuvisai-report
- Hydra/OmegaConf configs under cuvisai_examples/configs
- Component registries (datasets, models, runners, evaluators, reporters)
- .env support via python-dotenv (override=True)
- GPU default (CUDA) with CPU override

Install and run
- uv sync
- cp .env.example .env
- Train:
  uv run cuvisai-train model=efficientad/medium dataset=efficientad_train_val trainer.max_epochs=1
- Infer:
  uv run cuvisai-infer model=efficientad/medium dataset=efficientad
- Report:
  uv run cuvisai-report eval=efficientad reporting=efficientad

Porting scripts

1) Identify components
- Dataset -> torch Dataset registered in DATASETS
- Model -> class in MODELS (Lightning if trainable)
- Training loop -> RUNNERS (Lightning runner recommended)
- Evaluation/Reporting -> EVALUATORS/REPORTERS

2) Create configs
- dataset/<name>_train_val.yaml with train/val sections
- model/<family>/variant.yaml with type and params
- runner/lightning.yaml
- eval/<name>.yaml, reporting/<name>.yaml
- Reference in configs/train.yaml defaults

3) Replace argparse with Hydra
- Use @hydra.main in CLIs
- CLI overrides:
  datasets.train.params.path=/abs/path trainer.max_epochs=10

4) Env vars
- Move hard-coded paths to ${oc.env:VAR, default} and .env

5) Losses/metrics
- Expose knobs under model.params.loss
- Compute/log in Lightning steps

EfficientAD specifics
- Torch-only trainer removed; Lightning is canonical.
- Loss weights and preprocessing flags configurable:
  - model.params.loss.st_weight, ae_weight, imgnet_penalty_weight
  - model.params.preprocessing.compute_teacher_stats, compute_percentile_quantiles
- NPZ-backed dataset supported; cuvis optional for .npz data.

Common pitfalls
- Registry key mismatch: config type must match @register("key")
- Missing imports: ensure packages import modules where registration happens
- Hydra run dir: outputs under work_dir; set WORK_DIR env or config
