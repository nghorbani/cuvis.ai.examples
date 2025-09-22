![image](https://raw.githubusercontent.com/cubert-hyperspectral/cuvis.sdk/main/branding/logo/banner.png)

# cuvis.ai.examples

## Prerequisites
To run any of these examples, you need to have cuvis.ai installed in your environment. Please refer to the [cuvis.ai installation instructions](https://github.com/cubert-hyperspectral/cuvis.ai?tab=readme-ov-file#installation) to do so.
Please be also aware, that some examples might have their own prerequisites which can be found in the respective `README.md` and `requirements.txt`
## Inventory
For each example pretrained model weights, an extensive dataset and a step-by-step guide are provided.

### EfficientAD
Train a spatial and spectral aware anomaly detection algorithm and infer measurements using cuvis.ai. 

### Strawberry classification
Train a UNet to classify strawberries and find bruises on them.

## Install (uv)
- uv sync

## CLIs
- Train
  - uv run cuvisai-train model=efficientad/medium dataset=efficientad_train_val trainer.max_epochs=1
  - uv run cuvisai-train model=perpixel_ae dataset=perpixel_ae_train_val trainer.max_epochs=1
  - uv run cuvisai-train model=strawberry dataset=strawberry_train_val trainer.max_epochs=1
- Infer
  - uv run cuvisai-infer model=efficientad/medium dataset=efficientad
- Report
  - uv run cuvisai-report eval=efficientad reporting=efficientad

### Accelerator defaults
- The default trainer accelerator is CUDA (with easy override). CPU training will be very slow.
- Force GPU: add trainer.accelerator=cuda (and optionally trainer.devices=1)
  - Example: uv run cuvisai-train model=efficientad/medium dataset=efficientad_train_val trainer.max_epochs=1 trainer.accelerator=cuda
- Force CPU: trainer.accelerator=cpu

### Verbose logging and sanity checks
- Logs are printed to console and saved to ${work_dir}/train.log.
- Control verbosity via:
  - log_level=INFO|DEBUG
  - or logging.verbose=true (sets DEBUG)
- Lightning progress bar remains enabled; after each epoch, one-line summaries are printed and persist:
  - Train: train/loss, train/st, train/ae
  - Val: val/auroc, val/ap

- Example:
  - uv run cuvisai-train model=efficientad/medium dataset=efficientad_train_val trainer.max_epochs=1 trainer.accelerator=cpu logging.verbose=true
- You will see:
  - Dataset init summary: counts, NPZ/cu3s selection, ImageNet availability, normalization settings
  - Training start: computing teacher mean/std start/end
  - Validation start: percentile quantiles start/end with computed qa/qb
  - Skips are explicit (flags disabled or no val loader)

### EfficientAD configurable losses and preprocessing
- Loss weights:
  - model.params.loss.st_weight: student-teacher loss weight (default 1.0)
  - model.params.loss.ae_weight: autoencoder reconstruction loss weight (default 1.0)
  - model.params.loss.imgnet_penalty_weight: ImageNet penalty weight (default 0.0; requires model.params.use_imgNet_penalty=true)
- Preprocessing flags:
  - model.params.preprocessing.compute_teacher_stats: compute teacher feature mean/std on train start (default true)
  - model.params.preprocessing.compute_percentile_quantiles: compute percentile thresholds on val start using good samples (default true)
- Example:
  - uv run cuvisai-train model=efficientad/medium dataset=efficientad_train_val trainer.max_epochs=1 model.params.loss.ae_weight=0.5 model.params.preprocessing.compute_percentile_quantiles=false

## Environment variables
- Copy .env.example to .env and edit values as needed:
  - cp .env.example .env
- We load .env automatically in all CLIs and tools (dotenv override=True).
- Important vars:
  - HF_TOKEN: Hugging Face read token for private datasets
  - HF_REPO_ID: Dataset repo id (default: nghorbani/Hyperspektral-Small)
  - HF_LOCAL_DIR: Local download directory (default: data/Hyperspektral-Small)
  - WORK_DIR: Default work dir for outputs (default: ./work_dirs/exp)


Notes
- Hydra/OmegaConf configs live under cuvisai_examples/configs. Override any key via CLI.
- work_dir defaults to ./work_dirs/exp; set WORK_DIR env var or work_dir in configs to change.

## Deprecated
- EfficientAD/train_cuvis.py, PerPixelAE/train_cuvis.py, StrawberryClassification/train.py are deprecated. Use the CLIs above.

## Architecture and Extensibility
- See REGISTRY.md for how the registry system works and how to add models, datasets, losses, runners, evaluators, and reporters.
- See MIGRATION.md to port old scripts to the new Hydra/registry-based library.
- EfficientAD: only the Lightning training path is supported; torch-only trainer is removed. Backbones are provided via efficientad_torch and consumed by efficientad_lightning.

## Sample data (Hugging Face)
- Prepare:
  - cp .env.example .env && edit HF_TOKEN
  - uv sync
- Download:
  - uv run python tools/download_hf.py
  - Uses HF_TOKEN, HF_REPO_ID, HF_LOCAL_DIR from .env

## Using the provided Dropbox sample
- Download and extract to ./data:
  - curl -L "https://www.dropbox.com/scl/fi/l1ols1x6zoqq1yc5ehcrj/data.zip?rlkey=v2wi5cbxg2xue9tav4ebk3bil&amp;dl=1" -o data.zip
  - unzip -q data.zip -d ./data
- Note: EfficientAD now supports NPZ-backed loading. The sample extracts under:
  - ./data/Hyperspektral-Small/bedding_dataset/{train,val} with .npz files (no cuvis required)
- Run EfficientAD for 10 epochs on CPU:
  - uv run cuvisai-train model=efficientad/medium dataset=efficientad_train_val trainer.max_epochs=10 trainer.accelerator=cpu dataloader.batch_size=1 DATA_CUBES_DIR=./data/Hyperspektral-Small/bedding_dataset IMAGENET_DIR=./data/ImageNet_6_channel

## Reports
- EfficientAD: cuvisai-report reporting=efficientad eval=efficientad work_dir=./work_dirs/effad_report_smoke
- PerPixelAE: cuvisai-report reporting=perpixel_ae eval=perpixel_ae work_dir=./work_dirs/ppae_report_smoke
- Strawberry: cuvisai-report reporting=strawberry eval=strawberry work_dir=./work_dirs/strawberry_report_smoke

## EfficientAD setup
- cuvis Python SDK must be installed to read .cu3s session files.
- Configure data locations via .env:
  - DATA_CUBES_DIR=path/to/cubes
  - IMAGENET_DIR=path/to/ImageNet_6_channel
- Or override via CLI:
  - uv run cuvisai-train DATA_CUBES_DIR=/abs/cubes IMAGENET_DIR=/abs/ImageNet_6_channel ...

### Troubleshooting: No training batches
- If you see "Total length of DataLoader is zero" or "Trainer.fit stopped: No training batches":
  - Verify DATA_CUBES_DIR points to a directory containing .cu3s files
  - Verify cuvis is installed and importable in the environment
  - Ensure dataset mode and filters match your data layout

### Validation metrics
- During validation, EfficientAD logs:
  - val/auroc: AUROC computed from the continuous anomaly_map vs. binary masks (mask&gt;0)
  - val/ap: Average Precision on the same predictions/targets
- Metrics appear in the console/progress bar and are written to ${work_dir}/train.log.
- Ensure your validation dataset provides pixel masks for defects to enable these metrics.

