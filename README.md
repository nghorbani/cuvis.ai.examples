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

## Install
- pip install -e .

## CLIs
- Train
  - cuvisai-train model=efficientad/medium dataset=efficientad_train_val trainer.max_epochs=1
  - cuvisai-train model=perpixel_ae dataset=perpixel_ae_train_val trainer.max_epochs=1
  - cuvisai-train model=strawberry dataset=strawberry_train_val trainer.max_epochs=1
- Infer
  - cuvisai-infer model=efficientad/medium dataset=efficientad
- Report
  - cuvisai-report eval=efficientad reporting=efficientad

Notes
- Hydra/OmegaConf configs live under cuvisai_examples/configs. Override any key via CLI.
- work_dir defaults to ./work_dirs/exp; set WORK_DIR env var or work_dir in configs to change.

## Deprecated
- EfficientAD/train_cuvis.py, PerPixelAE/train_cuvis.py, StrawberryClassification/train.py are deprecated. Use the CLIs above.

