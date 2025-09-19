from typing import Tuple
import torch
from torch.utils.data import Dataset
from cuvisai_examples.registry import DATASETS


@DATASETS.register("DummyDataset")
class DummyDataset(Dataset):
    def __init__(self, length: int = 8, shape: Tuple[int, int, int] = (3, 8, 8)):
        self.length = length
        self.shape = shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        x = torch.randn(*self.shape)
        y = torch.tensor(0, dtype=torch.long)
        return {"image": x, "label": y, "meta": {"idx": idx}}
