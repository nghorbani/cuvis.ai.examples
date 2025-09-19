from pathlib import Path
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, HfFolder

def main():
    load_dotenv(override=True)
    out = Path(os.environ.get("HF_LOCAL_DIR", "data/Hyperspektral-Small"))
    out.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    if token:
        HfFolder.save_token(token)
    snapshot_download(
        repo_id=os.environ.get("HF_REPO_ID", "nghorbani/Hyperspektral-Small"),
        repo_type="dataset",
        local_dir=str(out),
        allow_patterns=None,
        ignore_patterns=None,
        token=token,
    )
    print("Downloaded to:", out.resolve())

if __name__ == "__main__":
    main()
