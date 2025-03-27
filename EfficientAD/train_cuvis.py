import argparse
import yaml
from torch.utils.data.dataloader import DataLoader
from EfficientADCuvisDataSet import EfficientADCuvisDataSet
from EfficientAD_lightning import EfficientAD_lightning
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config', type=str, required=True)
    args = parser.parse_args()
    return args


def parse_args(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return config


def train(config):
    train_data = EfficientADCuvisDataSet(config["Datasets"]["train"]["root"],
                                         mode="train",
                                         imageNet_path=config["Datasets"]["imagenet"]["root"],
                                         imageNet_file_ending=config["Datasets"]["imagenet"]["file_ending"],
                                         mean=config["mean"],
                                         std=config["std"],
                                         normalize=config["normalize"],
                                         max_img_shape=config["max_img_shape"],
                                         white_percentage=config["white_percentage"],
                                         channels=config["channels"])

    train_loader = DataLoader(train_data, batch_size=config["Model"]["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)

    test_data = EfficientADCuvisDataSet(config["Datasets"]["eval"]["root"],
                                        mode="test",
                                        imageNet_path=config["Datasets"]["imagenet"]["root"],
                                        imageNet_file_ending=config["Datasets"]["imagenet"]["file_ending"],
                                        mean=config["mean"],
                                        std=config["std"],
                                        normalize=config["normalize"],
                                        max_img_shape=config["max_img_shape"],
                                        white_percentage=config["white_percentage"],
                                        channels=config["channels"])

    test_loader = DataLoader(test_data, batch_size=config["Model"]["batch_size"], shuffle=False)

    # create custom callback to save a model checkpoint for every epoch
    checkpoint_callback = ModelCheckpoint(
        monitor="val_im/AU-ROC",  # Metric to monitor
        dirpath=config["ckpt_dir"] + "/" + config["name"],  # Directory to save checkpoints
        filename=config["name"] + "-{epoch:02d}-{val_im/AU-ROC:.2f}",  # Filename format
        save_top_k=-1,  # Save all checkpoints
        mode="max",
        verbose=True,
    )

    logger = TensorBoardLogger(save_dir=config["logger_dir"], log_graph=True, name=config['name'])
    if "ckpt" in config:
        model = EfficientAD_lightning.load_from_checkpoint(config["ckpt"], config=config)
    else:
        model = EfficientAD_lightning(config)
    trainer = L.Trainer(logger=logger, max_steps=config["max_steps"], benchmark=True, precision='16-mixed', gradient_clip_val=0.5, callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, test_loader)

    # TODO: find a way to cleanly exit cuvis


if __name__ == "__main__":
    args = get_arguments()
    config = parse_args(args)
    train(config)
