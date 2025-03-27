from pathlib import Path
import numpy as np
import cv2 as cv
from tqdm import tqdm

datasets = [["../data/ImageNet", ".JPEG"]]  # enter path to Imagenet dataset here

for dataset in datasets:
    path = Path(dataset[0])
    ext = dataset[1]
    files = [file for file in path.rglob(f"*{ext}")]

    print("found", len(files), "files in", dataset, "ext:", ext, "path:", path, "exists", path.exists())
    output_dir = path.parent / (path.name + "_6_channel")
    for file in tqdm(files):
        img = cv.imread(str(file))
        extended_img = np.zeros((img.shape[0], img.shape[1], 6), dtype='uint8')
        extended_img[:, :, :3] = img
        extended_img[:, :, 3:] = img

        relative = file.relative_to(path)
        new_path = output_dir / relative.with_suffix('.npy')
        new_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(new_path, extended_img)
