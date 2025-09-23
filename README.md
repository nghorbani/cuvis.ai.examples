![image](https://raw.githubusercontent.com/cubert-hyperspectral/cuvis.sdk/main/branding/logo/banner.png)

# cuvis.ai.examples

## Prerequisites
To run any of these examples, you need to have cuvis.ai installed in your environment. Please refer to the [cuvis.ai installation instructions](https://github.com/cubert-hyperspectral/cuvis.ai?tab=readme-ov-file#installation) to do so.
Please be also aware, that some examples might have their own prerequisites which can be found in the respective `README.md` and `requirements.txt`

## Downloading Data and Models
This project includes a tool to download datasets and models from Hugging Face for the examples.

### Setup
1. Ensure you have a Hugging Face token with read permissions for the required repositories.
2. Set the `HF_TOKEN` in your `.env` file (next to `.env.example` as a reference).
3. Install the project dependencies: `uv pip install -e .` (or `uv sync` if using lockfile).

### Usage
To download data and models for a specific example:
```
uv run get-data <name> --output_dir ./data
```
For example:
```
uv run get-data efficientad --output_dir ./data
```
This downloads the data and model to `./data/efficientad/data` and `./data/efficientad/model` respectively.

### Adding New Configurations
To add more download configurations, edit `hf_config.yaml`. Add new entries like:
```yaml
new_example:
  data: some_user/some_dataset_repo
  model: some_user/some_model_repo
```
You can add multiple keys (e.g., additional data sources) as needed.
## Inventory
For each example pretrained model weights, an extensive dataset and a step-by-step guide are provided.

### EfficientAD
Train a spatial and spectral aware anomaly detection algorithm and infer measurements using cuvis.ai. 

### Strawberry classification
Train a UNet to classify strawberries and find bruises on them.
