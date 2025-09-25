"""
EfficientAD Data Pipeline - Refactored Implementation

This module provides a modular data loading pipeline for EfficientAD training:

1. CubeDataset: Handles loading and preprocessing of cube files (.npz format)
2. ImageNetDataset: Manages ImageNet 6-channel data for pretraining penalty
3. EfficientADCuvisDataset: Combines both datasets and applies appropriate transforms

Key improvements:
- Separated concerns: cube loading vs ImageNet loading
- Explicit transform separation: geometric transforms vs color augmentation
- Efficient data handling with caching and minimal per-sample operations
- Backward compatible with existing test/validation code

Training mode returns:
- "image": Clean normalized cube (for student-teacher branch)
- "image_ae": Color-augmented cube (for autoencoder branch)
- "imgnet_img": Random ImageNet image (for pretraining penalty)

Test mode returns:
- "image": Normalized cube
- "label": 0 for good, 1 for defect
- "mask": Ground truth mask
- "defect": Defect type name
"""

from functools import partial
import os
from pathlib import Path
import random

import albumentations as A
import cv2 as cv
from loguru import logger
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2


class CubeDataset(Dataset):
    """
    Dataset for loading cube files (.npz format).
    Handles cube loading, cropping, normalization, and resizing.
    """

    def __init__(
        self,
        dataset_dir: str,
        mode: str = "train",
        mean: list = None,
        std: list = None,
        normalize: bool = True,
        max_img_shape: int = 1500,
        white_percentage: float = 0.55,
        channels: str = "ALL",
        max_data_load: int = -1,
    ):
        """
        Args:
            dataset_dir: Path to the cube files
            mode: 'train' or 'test' mode
            mean: List of means for each channel
            std: List of standard deviations for each channel
            normalize: Whether to normalize the input data
            max_img_shape: Maximum length of an image side
            white_percentage: Diffuse reflectance of the white target
            channels: Which channels to use ('RGB', 'SWIR', or 'ALL')
            max_data_load: Limit number of files to load (-1 for all)
        """
        self.path = dataset_dir
        self.mode = mode
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.max_img_shape = max_img_shape
        self.white_percentage = white_percentage
        self.channels = channels

        # Find all .npz files
        self.file_paths = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(self.path)
            for file in files
            if file.lower().endswith(".npz")
        ]

        if max_data_load > 0:
            logger.warning(
                f"max_data_load is set, so only a subset {max_data_load} of the dataset files {len(self.file_paths)} will be used!"
            )
            if self.mode == "train":
                self.file_paths = self.file_paths[:max_data_load]
            else:
                # For test mode, prioritize non-defect samples for debugging
                self.file_paths = [f for f in self.file_paths if "_ok_ok_" in f][:max_data_load]

        # For test mode, prepare ground truth masks
        if mode == "test":
            self.gt = {}
            for file_path in self.file_paths:
                if "_ok_ok_" in file_path:
                    continue  # no GT for good parts
                self.gt[file_path] = file_path.replace(".npz", "_0_RGB_mask.png")

        logger.info(f"CubeDataset: Found {len(self.file_paths)} {mode} cube files")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Load and preprocess a cube file."""
        file_path = self.file_paths[idx]

        # Load cube data
        cube = np.load(file_path)["arr_0"]
        cube = cube[300:-300, 300:-300, :]  # Crop borders
        cube = np.transpose(cube, (2, 0, 1))  # H×W×C to C×H×W
        cube = torch.from_numpy(cube).float()

        # Apply white percentage correction
        if self.white_percentage != 1:
            cube = cube * self.white_percentage

        # Normalize to [0, 1] range (100% reflectance = 10,000)
        cube = cube / 10000

        # Apply normalization if requested
        if self.normalize and self.mean is not None and self.std is not None:
            cube = torchvision.transforms.Normalize(mean=self.mean, std=self.std)(cube)

        # Resize if needed
        if cube.shape[1] > self.max_img_shape or cube.shape[2] > self.max_img_shape:
            cube = torchvision.transforms.Resize(size=self.max_img_shape - 1, max_size=self.max_img_shape)(cube)

        # Select channels
        if self.channels == "RGB":
            cube = cube[:3, :, :]
        elif self.channels == "SWIR":
            cube = cube[3:, :, :]

        if self.mode == "train":
            return {"image": cube, "file_path": file_path}
        else:
            # Test mode - return with labels and masks
            if "_ok_ok_" in file_path:
                return {
                    "image": cube,
                    "label": 0,
                    "mask": torch.zeros(cube.shape[-2:]),
                    "defect": "good",
                    "file_path": file_path,
                }
            else:
                defect = Path(file_path).parent.name
                if file_path in self.gt and os.path.exists(self.gt[file_path]):
                    # Load ground truth mask
                    mask = cv.imread(self.gt[file_path], cv.IMREAD_GRAYSCALE)[300:-300, 300:-300]
                    mask = torch.from_numpy(mask).unsqueeze(0).float()
                    mask = torchvision.transforms.Resize(
                        size=cube.shape[1:],
                        interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                    )(mask).squeeze(0)
                else:
                    logger.warning(f"NO GT DATA AVAILABLE for cube: {file_path}")
                    mask = torch.zeros(cube.shape[-2:])

                return {"image": cube, "label": 1, "mask": mask, "defect": defect, "file_path": file_path}


class ImageNetDataset(Dataset):
    """
    Dataset for loading ImageNet images (6-channel version).
    Supports both .npy and .jpeg formats.
    """

    def __init__(
        self,
        imagenet_dir: str,
        imagenet_file_ending: str = ".npy",
        mode: str = "train",
        max_data_load: int = -1,
    ):
        """
        Args:
            imagenet_dir: Path to ImageNet files
            imagenet_file_ending: File extension ('.npy' or '.jpeg')
            mode: 'train' or 'test' mode (used to filter subdirectories)
            max_data_load: Limit number of files to load (-1 for all)
        """
        self.imagenet_dir = imagenet_dir
        self.imagenet_file_ending = imagenet_file_ending
        self.mode = mode

        # Build file list
        self.imgnet_files = []
        if imagenet_dir is not None and os.path.exists(imagenet_dir):
            self.imgnet_files = [
                os.path.join(root, file)
                for root, dirs, files in os.walk(imagenet_dir)
                for file in files
                if file.lower().endswith(self.imagenet_file_ending) and mode in os.path.join(root, file)
            ]

        if max_data_load > 0 and len(self.imgnet_files) > max_data_load:
            logger.warning(
                f"max_data_load is set, so only a subset {max_data_load} of the ImageNet files {len(self.imgnet_files)} will be used!"
            )
            self.imgnet_files = self.imgnet_files[:max_data_load]

        logger.info(f"ImageNetDataset: Found {len(self.imgnet_files)} ImageNet files")

    def __len__(self):
        return len(self.imgnet_files)

    def __getitem__(self, idx):
        """Load and preprocess an ImageNet image."""
        file_path = self.imgnet_files[idx]

        if self.imagenet_file_ending == ".npy":
            imgnet_img = np.load(file_path)
        else:
            imgnet_img = np.array(cv.imread(file_path))

        # Convert to C×H×W format
        imgnet_img = np.transpose(imgnet_img, (2, 0, 1))
        imgnet_img = (imgnet_img / 255).astype(np.float32)
        imgnet_img = torch.from_numpy(imgnet_img)

        # Resize if needed (keep existing logic)
        h, w = imgnet_img.shape[1], imgnet_img.shape[2]
        if h > 1000 or w > 1000 or h < 256 or w < 256:
            imgnet_img = torchvision.transforms.Resize(size=500, max_size=1000)(imgnet_img)

        return imgnet_img

    def get_random_sample(self):
        """Get a random ImageNet sample."""
        if len(self.imgnet_files) == 0:
            raise ValueError("No ImageNet files available")
        idx = random.randint(0, len(self.imgnet_files) - 1)
        return self[idx]


class AlbumentationsTensorWrapper:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: C×H×W float tensor
        assert x.dim() == 3, "Expected CHW tensor"
        c, h, w = x.shape
        # Move to CPU numpy for Albumentations (H×W×C)
        np_img = x.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        out = self.aug(image=np_img)["image"]
        out_t = torch.from_numpy(out).permute(2, 0, 1).to(x.device).type_as(x)
        return out_t


class EfficientADCuvisDataset(Dataset):
    """
    Combined dataset that uses CubeDataset and ImageNetDataset.
    Returns appropriate data based on mode (train/test).
    """

    def __init__(
        self,
        dataset_dir: str,
        imagenet_dir: str,
        mode: str,
        imagenet_file_ending: str = ".npy",
        in_channels: int = 6,
        mean: list = None,
        std: list = None,
        normalize: bool = True,
        max_img_shape: int = 1500,
        white_percentage: float = 0.55,
        channels: str = "ALL",
        max_data_load: int = -1,
    ):
        """
        Combined dataset for EfficientAD training and testing.

        Args:
            dataset_dir: Path to cube files
            mode: 'train' or 'test' mode
            imagenet_dir: Path to ImageNet files
            imagenet_file_ending: ImageNet file extension
            in_channels: Number of input channels
            mean: Channel means for normalization
            std: Channel stds for normalization
            normalize: Whether to normalize
            max_img_shape: Maximum image dimension
            white_percentage: White reference correction
            channels: Channel selection ('RGB', 'SWIR', 'ALL')
            max_data_load: Limit files for debugging
        """
        self.mode = mode
        self.in_channels = in_channels

        # Initialize cube dataset
        self.cube_dataset = CubeDataset(
            dataset_dir=dataset_dir,
            mode=mode,
            mean=mean,
            std=std,
            normalize=normalize,
            max_img_shape=max_img_shape,
            white_percentage=white_percentage,
            channels=channels,
            max_data_load=max_data_load,
        )

        # Initialize ImageNet dataset (only for training)
        if mode == "train":
            self.imagenet_dataset = ImageNetDataset(
                imagenet_dir=imagenet_dir,
                imagenet_file_ending=imagenet_file_ending,
                mode=mode,
                max_data_load=max_data_load,
            )

            # Define transforms
            self.geom_transform = v2.Compose(
                [
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.RandomChoice(
                        [
                            v2.Lambda(partial(torch.rot90, k=0, dims=(-2, -1))),  # 0 deg
                            v2.Lambda(partial(torch.rot90, k=1, dims=(-2, -1))),  # 90 deg
                        ]
                    ),
                ]
            )

            # Color augmentation for AE branch (Albumentations - multi-channel safe)
            self.perceptual_transform = AlbumentationsTensorWrapper(
                A.Compose(
                    [
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                        # A.RandomGamma(gamma_limit=(90, 110), p=0.5),
                        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.5),
                        # A.GaussNoise(var_limit=(1e-5, 5e-4), mean=0.0, p=0.3),
                        # A.CoarseDropout(max_holes=2, max_height=0.1, max_width=0.1, fill_value=0.0, p=0.2),
                    ],
                    p=1.0,
                )
            )

    def __len__(self):
        return len(self.cube_dataset)

    def __getitem__(self, idx):
        """
        Get item for training or testing.

        For training mode, returns:
            - image: clean normalized cube (for student-teacher branch)
            - image_ae: augmented cube (for autoencoder branch)
            - imgnet_img: random ImageNet image (for pretraining penalty)

        For test mode, returns:
            - image: normalized cube
            - label: 0 for good, 1 for defect
            - mask: ground truth mask
            - defect: defect type name
        """
        # Get cube data
        cube_data = self.cube_dataset[idx]

        if self.mode == "train":
            # Get clean image
            clean_image = cube_data["image"]

            # Apply geometric transforms to both clean and AE images
            clean_image = self.geom_transform(clean_image)

            # Create augmented version for AE branch
            # First apply color augmentation, then geometric transforms
            ae_image = self.perceptual_transform(clean_image.clone())

            # Get random ImageNet image
            imgnet_img = self.imagenet_dataset.get_random_sample()

            return {"image": clean_image, "image_ae": ae_image, "imgnet_img": imgnet_img}
        else:
            # Test mode - return as-is from cube dataset
            return cube_data


# Backward compatibility - keep the old transform for reference
def get_legacy_transform():
    """Legacy transform for reference/testing."""
    return v2.Compose(
        [
            v2.Lambda(torch.as_tensor),
            v2.ToDtype(torch.float32, scale=False),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomChoice(
                [
                    v2.Lambda(partial(torch.rot90, k=0, dims=(-2, -1))),
                    v2.Lambda(partial(torch.rot90, k=1, dims=(-2, -1))),
                ]
            ),
        ]
    )
