from pathlib import Path
import os
from huggingface_hub import snapshot_download, HfFolder

def main():
    out = Path("data/Hyperspektral-Small")
    out.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    if token:
        HfFolder.save_token(token)
    snapshot_download(repo_id="nghorbani/Hyperspektral-Small", local_dir=str(out), allow_patterns=None, ignore_patterns=None, token=token)
    print("Downloaded to:", out.resolve())

if __name__ == "__main__":
    main()
