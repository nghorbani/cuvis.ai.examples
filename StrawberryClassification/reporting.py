from torch.utils.data import DataLoader
from Reporting.Report import Report
import os
import yaml
import argparse
import lightning as L
from pathlib import Path
from StrawberryLightning import StrawberryLightning
from StrawberryDataset import StrawberryDataset

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config', type=str, required=True)
    args = parser.parse_args()
    return args


def parse_args(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    args = get_arguments()
    config = parse_args(args)
    dataset = StrawberryDataset(Path(config["data_path"]),
                                white_path=config["white_path"],
                                dark_path=config["dark_path"],
                                cube_size=config["cube_size"],
                                strawberry_range=tuple(config["strawberry_range"]),
                                sides_to_exclude=config["sides_to_exclude"],
                                days_to_exclude=config["days_to_exclude"],
                                mean=config["mean"],
                                std=config["std"], )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"])
    model = StrawberryLightning.load_from_checkpoint(config["checkpoint_to_load"], config=config, data_loader=dataloader)
    trainer = L.Trainer(inference_mode=True, precision='16-mixed')
    rep = Report(config, model, trainer, Path("../data/StrawberryReporting/"), dataset, dataset_path=Path(config["data_path"]))
    rep.generate_report()
