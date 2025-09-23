from pathlib import Path
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
import fire


def download_hf(name, output_dir="./data"):
    """
    Download datasets and models from Hugging Face based on config.

    :param name: Config name, e.g. efficientad
    :param output_dir: Output directory
    """
    load_dotenv(override=True)
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; please export it or add to .env", file=sys.stderr)
        raise SystemExit(1)

    config_path = Path("hf_config.yaml")
    if not config_path.exists():
        print(f"Config file {config_path} not found", file=sys.stderr)
        raise SystemExit(1)

    cfg = OmegaConf.load(config_path)
    if name not in cfg:
        print(f"No config entry for '{name}'", file=sys.stderr)
        raise SystemExit(1)

    repos = OmegaConf.to_container(cfg[name], resolve=True)  # dict
    for key, repo_id in repos.items():
        repo_type = "dataset" if key == "data" else "model"
        cur_out_dir = Path(output_dir) / name / key
        cur_out_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(cur_out_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"Downloaded {repo_id} to: {cur_out_dir.resolve()}")


def main():
    fire.Fire(download_hf)


if __name__ == "__main__":
    main()
