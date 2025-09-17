import torchvision
from torch.utils.data import Dataset
import cuvis
from pathlib import Path
import numpy as np
import torch
import cv2 as cv


class StrawberryDataset(Dataset):
    def __init__(self,
                 root_dir: Path,
                 mean: list = None,
                 std: list = None,
                 normalize: bool = False,
                 white_path: str = None,
                 dark_path: str = None,
                 cube_size: list = None,
                 cube_rgb_channels=None,
                 strawberry_range: tuple = (0, 220),
                 sides_to_exclude: list = None,
                 days_to_exclude: list = None, ):
        if days_to_exclude is None:
            days_to_exclude = [28]
        if sides_to_exclude is None:
            sides_to_exclude = []
        if cube_rgb_channels is None:
            cube_rgb_channels = [4, 12, 25]
        if cube_size is None:
            cube_size = [200, 200]
        self.root_dir = root_dir
        self.file_paths = []
        for path in Path(self.root_dir).glob("*.cu3s"):
            name_splits = path.name.split("_")
            if strawberry_range[0] <= int(name_splits[1]) <= strawberry_range[1]:
                if int(name_splits[2]) not in sides_to_exclude and int(name_splits[3]) not in days_to_exclude:
                    self.file_paths.append(path)

        self.images = [[file_path, index]
                       for file_path in self.file_paths
                       for index in range(len(cuvis.SessionFile(file_path)))]
        self.masks = {}
        for file_path in self.file_paths:
            self.masks[file_path] = file_path.parent / "masks" / (
                    file_path.stem + "_0000_Strawberry_swir_fasterRGB_mask.npy")
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.proc = None
        self.white_path = white_path
        self.dark_path = dark_path
        self.height = cube_size[0]
        self.width = cube_size[1]
        self.cube_rgb_channels = cube_rgb_channels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        file_path = self.images[idx][0]
        file_name = file_path.name.split("_")
        sess = cuvis.SessionFile(file_path)
        mesu = sess.get_measurement(self.images[idx][1])
        if "cube" not in mesu.data:
            if self.proc is None:
                # create processing context only once if there are session files without cubes
                self.proc = cuvis.ProcessingContext(sess)
                if Path(self.white_path).exists() and Path(self.dark_path).exists():
                    self.proc.set_reference(cuvis.SessionFile(self.white_path).get_measurement(0), cuvis.ReferenceType.White)
                    self.proc.set_reference(cuvis.SessionFile(self.dark_path).get_measurement(0), cuvis.ReferenceType.Dark)
                    self.proc.processing_mode = cuvis.ProcessingMode.Reflectance
            mesu = self.proc.apply(mesu)
        cube = torch.from_numpy(mesu.data["cube"].array).to("cuda")
        cube = cube.permute(2, 0, 1)  # transpose from H x W x C to C x H x W for torch
        cube = cube / 10000  # 100% reflectance equals 10000, we divide by that to make 100% reflectance equal 1
        if self.normalize:
            cube = torchvision.transforms.Normalize(mean=self.mean, std=self.std)(cube)
        if cube.shape[1] != self.height or cube.shape[2] != self.width:
            cube = torchvision.transforms.Resize(size=[self.height, self.width])(cube)
        rgb = torch.zeros(3, self.height, self.width)
        rgb[0] = cube[self.cube_rgb_channels[0]]
        rgb[1] = cube[self.cube_rgb_channels[1]]
        rgb[2] = cube[self.cube_rgb_channels[2]]

        mask = torch.tensor(cv.resize(np.load(self.masks[file_path]), (self.height, self.width), interpolation=cv.INTER_NEAREST), device="cuda")

        return {
            "image": cube,
            "mask": mask,
            "number": file_name[1],
            "side": file_name[2],
            "day": file_name[3],
            "rgb_image": rgb,
            "name": file_path.stem}
