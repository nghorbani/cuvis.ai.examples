from pathlib import Path
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

try:
    from cuvisai_examples.configs import load_config  # provided by library
except Exception:
    load_config = None


def main():
    load_dotenv(override=True)
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; please export it or add to .env", file=sys.stderr)
        raise SystemExit(1)

    repo_id = "nghorbani/Hyperspektral-Small"
    local_dir = "data/Hyperspektral-Small"

    if load_config is not None:
        try:
            cfg = load_config(name="train")
            repo_id = OmegaConf.select(cfg, "hf.repo_id", default=repo_id)
            local_dir = OmegaConf.select(cfg, "hf.local_dir", default=local_dir)
        except Exception:
            pass

    out = Path(local_dir)
    out.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out),
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"Downloaded {repo_id} to: {out.resolve()}")


if __name__ == "__main__":
    main()
