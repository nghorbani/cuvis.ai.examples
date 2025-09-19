from typing import Optional, List
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import numpy as np
import cv2 as cv
import random
from functools import partial

from cuvisai_examples.registry import DATASETS

try:
    import cuvis
    from cuvis.cuvis_types import ProcessingMode
except Exception:
    cuvis = None
    ProcessingMode = None


@DATASETS.register("EfficientADCuvisDataSet")
class EfficientADCuvisDataSet(Dataset):
    def __init__(
        self,
        path: str = "data/cubes",
        mode: str = "train",
        imageNet_path: Optional[str] = None,
        imageNet_file_ending: str = ".npy",
        in_channels: int = 6,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        normalize: bool = True,
        max_img_shape: int = 1500,
        white_percentage: float = 0.55,
        channels: str = "ALL",
    ):
        self.path = path
        self.mode = mode
        self.imageNet_file_ending = imageNet_file_ending
        self.imageNet_path = imageNet_path
        self.in_channels = in_channels
        self.mean = mean
        self.std = std
        self.max_img_shape = max_img_shape
        self.normalize = normalize
        self.white_percentage = white_percentage
        self.channels = channels

        self.file_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(self.path)
            for f in files
            if f.lower().endswith(".cu3s")
        ]
        self.images = [
            [file_path, index]
            for file_path in self.file_paths
            for index in range(len(cuvis.SessionFile(file_path)))  # type: ignore[attr-defined]
        ] if cuvis is not None else []

        if imageNet_path is not None:
            self.imgNet_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(imageNet_path)
                for f in files
                if f.lower().endswith(self.imageNet_file_ending) and mode in os.path.join(root, f)
            ]
        else:
            self.imgNet_files = []

        if mode == "test":
            self.gt = {}
            for file_path in self.file_paths:
                if "_ok_ok_" not in file_path:
                    self.gt[file_path] = file_path.replace(".cu3s", "_0_RGB_mask.png")

        self.transform = lambda x: x

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
        cube = np.transpose(cube, (2, 0, 1))
        cube = torch.from_numpy(cube)
        if self.white_percentage != 1:
            cube = cube * self.white_percentage
        cube = cube / 10000.0
        if self.normalize and self.mean is not None and self.std is not None:
            cube = torchvision.transforms.Normalize(mean=self.mean, std=self.std)(cube)
        if cube.shape[1] > self.max_img_shape or cube.shape[2] > self.max_img_shape:
            cube = torchvision.transforms.Resize(size=self.max_img_shape - 1, max_size=self.max_img_shape)(cube)
        if self.channels == "RGB":
            cube = cube[:3, :, :]
        elif self.channels == "SWIR":
            cube = cube[3:, :, :]
        return cube

    def _load_imagenet(self) -> torch.Tensor:
        if self.imageNet_file_ending == ".npy":
            imgNet_img = np.load(random.choice(self.imgNet_files))
        else:
            imgNet_img = np.array(cv.imread(random.choice(self.imgNet_files)))
        imgNet_img = np.transpose(imgNet_img, (2, 0, 1))
        imgNet_img = (imgNet_img / 255).astype(np.float32)
        imgNet_img = torch.from_numpy(imgNet_img)
        if imgNet_img.shape[1] > 1000 or imgNet_img.shape[2] > 1000 or imgNet_img.shape[1] < 256 or imgNet_img.shape[2] < 256:
            imgNet_img = torchvision.transforms.Resize(size=500, max_size=1000)(imgNet_img)
        return imgNet_img

    def __getitem__(self, idx: int):
        file_path, index = self.images[idx]
        cube = self._load_cube(file_path, index)
        if self.mode == "train":
            imgNet_img = self._load_imagenet() if self.imgNet_files else torch.zeros_like(cube[:3])
            return self.transform({"image": cube, "imgNet_img": imgNet_img})
        else:
            if "_ok_ok_" in file_path:
                return {"image": cube, "label": torch.tensor(0, dtype=torch.long), "mask": torch.zeros(cube.shape[-2:]), "defect": "good"}
            else:
                defect = Path(file_path).parent.name
                if hasattr(self, "gt") and file_path in self.gt and os.path.exists(self.gt[file_path]):
                    mask = cv.imread(self.gt[file_path], cv.IMREAD_GRAYSCALE)[300:-300, 300:-300]
                    mask = torch.from_numpy(mask).unsqueeze(0)
                    mask_out = torchvision.transforms.Resize(size=cube.shape[1:], interpolation=torchvision.transforms.InterpolationMode.NEAREST)(mask).squeeze(0)
                else:
                    mask_out = torch.zeros(cube.shape[-2:])
                return {"image": cube, "label": torch.tensor(1, dtype=torch.long), "mask": mask_out, "defect": defect}
