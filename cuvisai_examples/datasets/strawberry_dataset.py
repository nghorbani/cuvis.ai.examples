from typing import List, Optional, Tuple
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import cv2 as cv

from cuvisai_examples.registry import DATASETS

try:
    import cuvis  # type: ignore
except Exception:
    cuvis = None


@DATASETS.register("StrawberryDataset")
class StrawberryDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        normalize: bool = False,
        white_path: Optional[str] = None,
        dark_path: Optional[str] = None,
        cube_size: Optional[List[int]] = None,
        cube_rgb_channels: Optional[List[int]] = None,
        strawberry_range: Tuple[int, int] = (0, 220),
        sides_to_exclude: Optional[List[int]] = None,
        days_to_exclude: Optional[List[int]] = None,
        obtain: Optional[dict] = None,
    ):
        if days_to_exclude is None:
            days_to_exclude = [28]
        if sides_to_exclude is None:
            sides_to_exclude = []
        if cube_rgb_channels is None:
            cube_rgb_channels = [4, 12, 25]
        if cube_size is None:
            cube_size = [200, 200]

        self.root_dir = Path(root_dir)

        if not self.root_dir.exists():
            if obtain and isinstance(obtain, dict) and "hf" in obtain:
                hf = obtain.get("hf") or {}
                repo_id = hf.get("repo_id")
                local_dir = hf.get("local_dir", str(self.root_dir))
                token = os.environ.get("HF_TOKEN")
                if repo_id and token:
                    try:
                        from huggingface_hub import snapshot_download

                        print(
                            f"Dataset dir missing; downloading from HF: repo_id={repo_id} to {local_dir}"
                        )
                        snapshot_download(
                            repo_id=repo_id,
                            repo_type="dataset",
                            local_dir=str(local_dir),
                            local_dir_use_symlinks=False,
                            token=token,
                        )
                    except Exception as e:
                        print(f"HF download failed: {e}")
                else:
                    print(
                        "Dataset dir missing and obtain.hf provided but repo_id or HF_TOKEN missing; skipping download."
                    )
            elif obtain and isinstance(obtain, dict) and "manual" in obtain:
                manual = obtain.get("manual") or {}
                download_url = manual.get("url")
                instructions = manual.get("instructions", "Please download the dataset manually")
                print(
                    f"Dataset not found at: {self.root_dir}\n"
                    f"{instructions}\n"
                    f"Expected location: {self.root_dir.resolve()}\n"
                    f"After downloading, run this script once more."
                    + (f"\nDownload URL: {download_url}" if download_url else "")
                )
                raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
            else:
                print(
                    f"Dataset not found at: {self.root_dir}\n"
                    f"Please download the dataset to: {self.root_dir.resolve()}\n"
                    f"After downloading, run this script once more."
                )
                raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.file_paths: List[Path] = []
        for path in self.root_dir.glob("*.cu3s"):
            name_splits = path.name.split("_")
            try:
                num = int(name_splits[1])
                side = int(name_splits[2])
                day = int(name_splits[3])
            except Exception:
                continue
            if strawberry_range[0] <= num <= strawberry_range[1]:
                if side not in sides_to_exclude and day not in days_to_exclude:
                    self.file_paths.append(path)

        self.images = []
        if cuvis is not None:
            for file_path in self.file_paths:
                try:
                    sess = cuvis.SessionFile(file_path)  # type: ignore
                    count = len(sess)
                    for index in range(count):
                        self.images.append([file_path, index])
                except Exception:
                    continue

        self.masks = {}
        for file_path in self.file_paths:
            self.masks[file_path] = (
                file_path.parent
                / "masks"
                / (file_path.stem + "_0000_Strawberry_swir_fasterRGB_mask.npy")
            )

        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.proc = None
        self.white_path = white_path
        self.dark_path = dark_path
        self.height = int(cube_size[0])
        self.width = int(cube_size[1])
        self.cube_rgb_channels = cube_rgb_channels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_path: Path = self.images[idx][0]
        index = self.images[idx][1]
        file_name = file_path.name.split("_")

        sess = cuvis.SessionFile(file_path)  # type: ignore
        mesu = sess.get_measurement(index)
        if "cube" not in mesu.data:
            if self.proc is None:
                self.proc = cuvis.ProcessingContext(sess)  # type: ignore
                if (
                    self.white_path
                    and self.dark_path
                    and Path(self.white_path).exists()
                    and Path(self.dark_path).exists()
                ):
                    self.proc.set_reference(
                        cuvis.SessionFile(self.white_path).get_measurement(0),
                        cuvis.ReferenceType.White,
                    )  # type: ignore
                    self.proc.set_reference(
                        cuvis.SessionFile(self.dark_path).get_measurement(0),
                        cuvis.ReferenceType.Dark,
                    )  # type: ignore
                    self.proc.processing_mode = cuvis.ProcessingMode.Reflectance  # type: ignore
            mesu = self.proc.apply(mesu)  # type: ignore

        cube = torch.from_numpy(mesu.data["cube"].array)  # C,H,W comes as H,W,C
        cube = cube.permute(2, 0, 1)  # C,H,W
        cube = cube / 10000.0
        if self.normalize and self.mean is not None and self.std is not None:
            cube = torchvision.transforms.Normalize(mean=self.mean, std=self.std)(cube)
        if cube.shape[1] != self.height or cube.shape[2] != self.width:
            cube = torchvision.transforms.Resize(size=[self.height, self.width])(cube)

        rgb = torch.zeros(3, self.height, self.width, dtype=cube.dtype)
        rgb[0] = cube[self.cube_rgb_channels[0]]
        rgb[1] = cube[self.cube_rgb_channels[1]]
        rgb[2] = cube[self.cube_rgb_channels[2]]

        mask_path = self.masks.get(file_path)
        if mask_path and os.path.exists(mask_path):
            mask_np = np.load(mask_path)
            mask = torch.tensor(
                cv.resize(
                    mask_np, (self.width, self.height), interpolation=cv.INTER_NEAREST
                )
            )
        else:
            mask = torch.zeros((self.height, self.width), dtype=torch.long)

        return {
            "image": cube,
            "mask": mask,
            "number": file_name[1],
            "side": file_name[2],
            "day": file_name[3],
            "rgb_image": rgb,
            "name": file_path.stem,
        }
