from typing import Optional, List
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv

from cuvisai_examples.registry import DATASETS

try:
    import cuvis
    from cuvis.cuvis_types import ProcessingMode
except Exception:
    cuvis = None
    ProcessingMode = None


@DATASETS.register("PerPixelAECuvisDataSet")
class PerPixelAECuvisDataSet(Dataset):
    def __init__(
        self,
        path: str = "data/cubes",
        mode: str = "train",
        max_img_shape: int = 1500,
        channels: str = "ALL",
        white_percentage: float = 0.55,
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        self.path = path
        self.mode = mode
        self.max_img_shape = max_img_shape
        self.channels = channels
        self.white_percentage = white_percentage
        self.normalize = normalize
        self.mean = mean
        self.std = std

        self.file_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(self.path)
            for f in files
            if f.lower().endswith(".cu3s")
        ]
        self.images = (
            [
                [file_path, index]
                for file_path in self.file_paths
                for index in range(len(cuvis.SessionFile(file_path)))  # type: ignore[attr-defined]
            ]
            if cuvis is not None
            else []
        )

        self.proc = None

    def __len__(self):
        return len(self.images)

    def _load_cube(self, file_path: str, index: int) -> torch.Tensor:
        sess = cuvis.SessionFile(file_path)  # type: ignore[operator]
        mesu = sess.get_measurement(index)
        if "cube" not in mesu.data:
            if self.proc is None:
                self.proc = cuvis.ProcessingContext(sess)  # type: ignore[operator]
                self.proc.processing_mode = ProcessingMode.Raw  # type: ignore[assignment]
            mesu = self.proc.apply(mesu)
        cube = mesu.data["cube"].array
        cube = cube[300:-300, 300:-300, :]
        cube = np.transpose(cube, (2, 0, 1)).astype(np.float32)
        cube = torch.from_numpy(cube)
        if self.white_percentage != 1:
            cube = cube * self.white_percentage
        cube = cube / 10000.0
        if self.normalize and self.mean is not None and self.std is not None:
            cube = torch.nn.functional.layer_norm(cube, cube.shape, torch.tensor(self.mean, dtype=cube.dtype, device=cube.device), torch.tensor(self.std, dtype=cube.dtype, device=cube.device))
        if cube.shape[1] > self.max_img_shape or cube.shape[2] > self.max_img_shape:
            h, w = cube.shape[1], cube.shape[2]
            scale = min(self.max_img_shape / h, self.max_img_shape / w)
            new_h, new_w = int(h * scale), int(w * scale)
            cube = torch.nn.functional.interpolate(cube.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        if self.channels == "RGB":
            cube = cube[:3, :, :]
        elif self.channels == "SWIR":
            cube = cube[3:, :, :]
        return cube

    def __getitem__(self, idx: int):
        file_path, index = self.images[idx]
        cube = self._load_cube(file_path, index)
        return {"image": cube, "label": torch.tensor(0, dtype=torch.long), "meta": {"path": file_path, "index": index}}
