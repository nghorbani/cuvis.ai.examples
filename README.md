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

## Sample data (Hugging Face)
- Prepare:
  - cp .env.example .env && edit HF_TOKEN
  - uv sync
- Download:
  - uv run python tools/download_hf.py
  - Uses HF_TOKEN, HF_REPO_ID, HF_LOCAL_DIR from .env
## Real-data smoke runs (CPU)
- EfficientAD
  - cuvisai-train model=efficientad/medium dataset=efficientad_train_val trainer.max_epochs=1 trainer.accelerator=cpu dataloader.batch_size=1 dataset.train.params.path=./data/Hyperspektral-Small dataset.val.params.path=./data/Hyperspektral-Small
- PerPixelAE
  - cuvisai-train model=perpixel_ae dataset=perpixel_ae_train_val trainer.max_epochs=1 trainer.accelerator=cpu dataloader.batch_size=1 dataset.train.params.path=./data/Hyperspektral-Small dataset.val.params.path=./data/Hyperspektral-Small
- Strawberry
  - cuvisai-train model=strawberry dataset=strawberry_train_val trainer.max_epochs=1 trainer.accelerator=cpu dataloader.batch_size=1 dataset.train.params.root_dir=./data/Hyperspektral-Small dataset.val.params.root_dir=./data/Hyperspektral-Small

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
  - uv run cuvisai-train dataset.train.params.path=/abs/cubes dataset.train.params.imageNet_path=/abs/ImageNet_6_channel ...

### Troubleshooting: No training batches
- If you see "Total length of DataLoader is zero" or "Trainer.fit stopped: No training batches":
  - Verify DATA_CUBES_DIR points to a directory containing .cu3s files
  - Verify cuvis is installed and importable in the environment
  - Ensure dataset mode and filters match your data layout

