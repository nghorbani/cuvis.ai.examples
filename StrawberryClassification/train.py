import argparse
import yaml
import lightning as L
from StrawberryLightning import StrawberryLightning
from torch.utils.data import DataLoader
from StrawberryDataset import StrawberryDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import torch


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config', type=str, required=True)
    args = parser.parse_args()
    return args


def parse_args(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = get_arguments()
    config = parse_args(args)

    full_dataset = StrawberryDataset(Path(config["data_path"]),
                                     white_path=config["white_path"],
                                     dark_path=config["dark_path"],
                                     cube_size=config["cube_size"],
                                     strawberry_range=tuple(config["strawberry_range"]),
                                     sides_to_exclude=config["sides_to_exclude"],
                                     days_to_exclude=config["days_to_exclude"],
                                     mean=config["mean"],
                                     std=config["std"],
                                     normalize=config["normalize"], )

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(full_dataset,
                                                          [train_size, test_size],
                                                          generator=torch.Generator().manual_seed(config["seed"]))

    train_loader = DataLoader(train_data,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              num_workers=config["num_workers"],
                              persistent_workers=config["num_workers"] > 0)

    test_loader = DataLoader(test_data,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             num_workers=config["num_workers"],
                             persistent_workers=config["num_workers"] > 0)

    checkpoint_callback = ModelCheckpoint(
        monitor="train/epoch_loss",  # Metric to monitor
        dirpath=config["ckpt_dir"] + "/" + config["name"],  # Directory to save checkpoints
        filename=config["best_ckpt"],  # Filename format
        save_top_k=-1,  # Save all checkpoints
        mode="min",
        verbose=True,
    )
    logger = TensorBoardLogger(save_dir=config["logger_dir"], log_graph=True, name=config['name'])

    model = StrawberryLightning(config,
                                DataLoader(full_dataset,
                                            batch_size=config["batch_size"],
                                            shuffle=False,
                                            num_workers=config["num_workers"],
                                            persistent_workers=config["num_workers"] > 0))

    trainer = L.Trainer(logger=logger,
                        max_steps=config["max_steps"],
                        benchmark=True,
                        precision='16-mixed',
                        gradient_clip_val=0.5,
                        callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, test_loader)
    # Force cleanup of datasets and dataloaders
    del train_loader, test_loader, full_dataset
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
