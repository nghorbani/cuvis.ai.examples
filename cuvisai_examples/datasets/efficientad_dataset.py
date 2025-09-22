from typing import Optional, List
import os
from pathlib import Path
import logging
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

        self.npz_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(self.path)
            for f in files
            if f.lower().endswith(".npz")
        ]
        self.uses_npz = len(self.npz_paths) > 0

        if not self.uses_npz and cuvis is not None:
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
            ]
        else:
            self.file_paths = []
            self.images = list(range(len(self.npz_paths)))

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
            if self.uses_npz:
                for file_path in self.npz_paths:
                    if "_ok_ok_" not in file_path:
                        self.gt[file_path] = file_path.replace(".npz", "_0_RGB_mask.png")
            else:
                for file_path in self.file_paths:
                    if "_ok_ok_" not in file_path:
                        self.gt[file_path] = file_path.replace(".cu3s", "_0_RGB_mask.png")

        logging.getLogger(__name__).info(f"EfficientAD dataset init: mode={self.mode} path={self.path}")
        logging.getLogger(__name__).info(f"Found {len(self.npz_paths)} NPZ files")
        if not self.uses_npz:
            logging.getLogger(__name__).info(f"Found {len(self.file_paths)} CU3S files; cuvis={'available' if cuvis is not None else 'missing'}")
        logging.getLogger(__name__).info(f"ImageNet dir set: {self.imageNet_path is not None}; files={len(self.imgNet_files) if hasattr(self, 'imgNet_files') else 0}")
        if self.normalize:
            logging.getLogger(__name__).info(f"Normalization enabled; mean/std provided={self.mean is not None and self.std is not None}")
            if self.mean is None or self.std is None:
                logging.getLogger(__name__).warning("Normalization requested but mean/std not provided; skipping normalization.")
        if self.mode == "train" and not self.imgNet_files:
            logging.getLogger(__name__).warning("No ImageNet files found; imgNet penalty images will be unavailable.")

        self.transform = lambda x: x
        self.proc = None
        self._dbg = 0

    def __len__(self):
        return len(self.images)

    def _load_cube_from_npz(self, npz_path: str) -> torch.Tensor:
        with np.load(npz_path, allow_pickle=False) as npz:
            cube = npz["arr_0"]
        cube = cube[300:-300, 300:-300, :]
        cube = np.transpose(cube, (2, 0, 1))
        cube = torch.from_numpy(cube).to(torch.float32)
        if self.white_percentage != 1:
            cube.mul_(self.white_percentage)
        cube.div_(10000.0)
        if self.normalize and self.mean is not None and self.std is not None:
            cube = torchvision.transforms.Normalize(mean=self.mean, std=self.std)(cube)
        if cube.shape[1] > self.max_img_shape or cube.shape[2] > self.max_img_shape:
            cube = torchvision.transforms.Resize(size=self.max_img_shape - 1, max_size=self.max_img_shape)(cube)
        if self.channels == "RGB":
            cube = cube[:3, :, :]
        elif self.channels == "SWIR":
            cube = cube[3:, :, :]
        return cube

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
            if self._dbg < 3:
                logging.getLogger(__name__).debug(f"EfficientAD item {idx}: npz cube shape pre-return={tuple(cube.shape)} channels_sel={self.channels}")
                self._dbg += 1

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
        if self.uses_npz:
                        if self._dbg < 3:
                            logging.getLogger(__name__).debug(f"EfficientAD eval item {idx}: mask found for {npz_path} -> resized to {tuple(mask_out.shape)}")
                            self._dbg += 1

            npz_path = self.npz_paths[idx]
            cube = self._load_cube_from_npz(npz_path)
                        if self._dbg < 3:
                            logging.getLogger(__name__).debug(f"EfficientAD eval item {idx}: NO mask for {npz_path}, returning zeros with shape={tuple(mask_out.shape)}")
                            self._dbg += 1

            if self.mode == "train":
                imgNet_img = self._load_imagenet() if self.imgNet_files else torch.zeros((3, cube.shape[-2], cube.shape[-1]), dtype=cube.dtype)
                return self.transform({"image": cube, "imgNet_img": imgNet_img})
            else:
                if "_ok_ok_" in npz_path:
                    return {"image": cube, "label": torch.tensor(0, dtype=torch.long), "mask": torch.zeros(cube.shape[-2:]), "defect": "good"}
                if self._dbg < 3:
                    logging.getLogger(__name__).debug(f"EfficientAD item {idx}: cu3s cube shape pre-return={tuple(cube.shape)} channels_sel={self.channels}")
                    self._dbg += 1

                else:
                    defect = Path(npz_path).parent.name
                    if hasattr(self, "gt") and npz_path in self.gt and os.path.exists(self.gt[npz_path]):
                        mask = cv.imread(self.gt[npz_path], cv.IMREAD_GRAYSCALE)[300:-300, 300:-300]
                        mask = torch.from_numpy(mask).unsqueeze(0)
                        mask_out = torchvision.transforms.Resize(size=cube.shape[1:], interpolation=torchvision.transforms.InterpolationMode.NEAREST)(mask).squeeze(0)
                    else:
                        mask_out = torch.zeros(cube.shape[-2:])
                    return {"image": cube, "label": torch.tensor(1, dtype=torch.long), "mask": mask_out, "defect": defect}
        else:
                        if self._dbg < 3:
                            logging.getLogger(__name__).debug(f"EfficientAD eval item {idx}: mask found for {file_path} -> resized to {tuple(mask_out.shape)}")
                            self._dbg += 1

            file_path, index = self.images[idx]
            cube = self._load_cube(file_path, index)
                        if self._dbg < 3:
                            logging.getLogger(__name__).debug(f"EfficientAD eval item {idx}: NO mask for {file_path}, returning zeros with shape={tuple(mask_out.shape)}")
                            self._dbg += 1

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
