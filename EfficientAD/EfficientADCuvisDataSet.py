import torchvision
from torch.utils.data import Dataset
import cuvis
import os
from cuvis.cuvis_types import ProcessingMode
import numpy as np
import cv2 as cv
import random
import torch
from torchvision.transforms import v2
from functools import partial
from pathlib import Path


class EfficientADCuvisDataSet(Dataset):
    """
    Dataset class to use the cuvis dataset with the EfficientAD model.
    """

    def __init__(self, path: str = "data/cubes", mode: str = "train", imageNet_path: str = "../data/ImageNet_6_channel",
                 imageNet_file_ending: str = '.npy', in_channels: int = 6, mean: list = None, std: list = None, normalize: bool = True, max_img_shape: int = 1500, white_percentage: float = 0.55,
                 channels: str = "ALL"):
        """
        :param path: Path to the session files. These must contain data cubes which are expected to be in reflectance mode. default: 'data/cubes'
        :param mode: Mode for which this dataset will be used. If this is 'train' the data will be prepared for training, otherwise it will fit the validation / inference process. default: 'train'
        :param imageNet_path: Path to the imageNet files needed for training. default: '../data/ImageNet_6_channel'
        :param imageNet_file_ending: File extension of the specified ImageNet dataset, either '.npy' or '.jpeg'. default: '.npy'
        :param in_channels: Number of input channels to the model. default: '6'
        :param mean: List of means for each channel of the input dataset. default: None
        :param std: List of standard deviations for each channel of the input dataset. default: None
        :param normalize: Whether to normalize the input data. default: True
        :param max_img_shape: Maximum length of an image side: default: 1500
        :param white_percentage: Diffuse reflectance of the white target used as reference for the reflectance calculation. default: 0.55
        :param channels: Which channels of the datacube to use. This can be 'RGB', 'SWIR' or 'ALL'. default: 'ALL'
        """
        self.path = path
        self.mode = mode
        self.imageNet_file_ending = imageNet_file_ending
        self.imageNet_path = imageNet_path
        self.file_paths = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(self.path)
            for file in files if file.lower().endswith(".cu3s")
        ]
        self.in_channels = in_channels
        self.images = [[file_path, index]
                       for file_path in self.file_paths
                       for index in range(len(cuvis.SessionFile(file_path)))]

        if imageNet_path is not None:
            self.imgNet_files = [
                os.path.join(root, file)
                for root, dirs, files in os.walk(imageNet_path)
                for file in files if file.lower().endswith(self.imageNet_file_ending) and mode in os.path.join(root, file)
            ]
        if mode == 'test':
            self.gt = {}
            for file_path in self.file_paths:
                if "_ok_ok_" not in file_path:
                    self.gt[file_path] = file_path.replace(".cu3s", "_0_RGB_mask.png")

        self.transform = v2.Compose([
            v2.Lambda(torch.as_tensor),
            v2.ToDtype(torch.float32, scale=False),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomChoice([
                v2.Lambda(partial(torch.rot90, k=0, dims=(-2, -1))),  # rotate 0 deg
                v2.Lambda(partial(torch.rot90, k=1, dims=(-2, -1)))  # rotate 90 deg
            ]),
        ])

        self.proc = None
        self.mean = mean
        self.std = std
        self.max_img_shape = max_img_shape
        self.normalize = normalize
        self.white_percentage = white_percentage
        self.channels = channels
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Gets and prepares the next item for the training loop.
        :param idx: index of the item to get
        :return: dict either containing the cube and an imageNet image for training or cube, label, mask and which defect is shown for validation / inference
        """
        file_path = self.images[idx][0]
        sess = cuvis.SessionFile(file_path)
        mesu = sess.get_measurement(self.images[idx][1])

        if "cube" not in mesu.data:
            if self.proc is None:
                # create processing context only once if there are session files without cubes
                self.proc = cuvis.ProcessingContext(sess)
                self.proc.processing_mode = ProcessingMode.Raw

            mesu = self.proc.apply(mesu)
        cube = mesu.data["cube"].array
        cube = cube[300:-300, 300:-300,:] # cut the border of the image to exclude the tray borders
        cube = np.transpose(cube, (2, 0, 1))  # transpose from H x W x C to C x H x W for torch
        cube = torch.from_numpy(cube)
        if self.white_percentage != 1:
            cube = cube * self.white_percentage
        cube = cube / 10000  # 100% reflectance equals 10.000, we divide by that to make 100% reflectance equal 1
        if self.normalize:
            cube = torchvision.transforms.Normalize(mean=self.mean, std=self.std)(cube)
        if cube.shape[1] > self.max_img_shape or cube.shape[2] > self.max_img_shape:
            cube = torchvision.transforms.Resize(size=self.max_img_shape - 1, max_size=self.max_img_shape)(cube)
        if self.channels == "RGB":
            cube = cube[:3,:,:]
        elif self.channels == "SWIR":
            cube = cube[3:,:,:]
        if self.mode == "train":

            if self.imageNet_file_ending == ".npy":
                imgNet_img = np.load(random.choice(self.imgNet_files))
            else:
                imgNet_img = np.array(cv.imread(random.choice(self.imgNet_files)))
            imgNet_img = np.transpose(imgNet_img, (2, 0, 1))  # transpose from H x W x C to C x H x W for torch
            imgNet_img = (imgNet_img / 255).astype(np.float32)
            imgNet_img = torch.from_numpy(imgNet_img)
            if imgNet_img.shape[1] > 1000 or imgNet_img.shape[2] > 1000 or imgNet_img.shape[1] < 256 or imgNet_img.shape[2] < 256:
                imgNet_img = torchvision.transforms.Resize(size=500, max_size=1000)(imgNet_img)

            return self.transform({"image": cube, "imgNet_img": imgNet_img})
        else:
            if "_ok_ok_" in file_path:
                return {"image": cube, "label": 0, "mask": torch.zeros(cube.shape[-2:], dtype=torch.bool), "defect": "good"}
            else:
                defect = Path(file_path).parent.name
                if os.path.exists(self.gt[file_path]):
                    mask = cv.imread(self.gt[file_path], cv.IMREAD_GRAYSCALE)[300:-300, 300:-300] # Crop the mask
                    mask = torch.from_numpy(mask)
                    mask = mask.unsqueeze(0)
                    mask_out = torchvision.transforms.Resize(size=cube.shape[1:], interpolation=torchvision.transforms.InterpolationMode.NEAREST)(mask).squeeze(0) # Resize it in the same way
                else:
                    print(f'NO GT DATA AVAILABLE for cube: {file_path}')
                    mask_out = torch.zeros(cube.shape[-2:], dtype=torch.bool)
                return {"image": cube, "label": 1, "mask": mask_out, "defect": defect}
