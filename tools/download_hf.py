from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    out = Path("data/Hyperspektral-Small")
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="nghorbani/Hyperspektral-Small", local_dir=str(out), allow_patterns=None, ignore_patterns=None)
    print("Downloaded to:", out.resolve())

if __name__ == "__main__":
    main()
