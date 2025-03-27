import pathlib

import lightning
import numpy as np

from EfficientADCuvisDataSet import EfficientADCuvisDataSet
import yaml
import torch
import lightning as L
from EfficientAD.EfficientAD_lightning import EfficientAD_lightning
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot as plt
import argparse
import os
import tqdm
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
import cv2 as cv
import glob


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config', type=str, required=True)
    args = parser.parse_args()
    return args


def parse_args(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return config


class Report:
    """
    Class to create a report for a given dataset folder
    
    Args:
        config(dict): parsed yaml configuration.
        model(torch.model): model to use to infer the data.
        trainer(lightning.Trainer): lightning.Trainer class to use.
        reporting_root_folder(pathlib.Path): root folder where reportings should be saved
    """

    def __init__(self, config: dict, model: torch.nn.Module, trainer: lightning.Trainer, reporting_root_folder: pathlib.Path):
        self.config = config
        self.model = model
        self.trainer = trainer
        self.mean = np.array(config['means'])
        self.std = np.array(config['stds'])
        self.plot_thresholds = config['plot_thresholds']
        self.reporting_root_folder = reporting_root_folder
        self.name = config["name"]
        self.reporting_run_folder = reporting_root_folder / self.name
        self.create_images = config['create_images'] if 'create_images' in config else True

    def generate_report(self):
        """
        Generates the report. This consists of an inference of all cubes given. It creates images for each cube, showing the RGB image, a SWIR representation and the model prediction as well as some threshold images.
        :return:
        """
        if not os.path.exists(self.reporting_run_folder):
            os.makedirs(self.reporting_run_folder)

        # dump the configuration used to create this report
        with open(self.reporting_run_folder / "reporting_config.yaml", "w") as f:
            yaml.dump(self.config, f)
        metrics = {}
        all_labels = []
        all_scores = []
        for dataset_path in config['datasets']:
            data_path = Path(dataset_path)
            cubes = glob.glob(str(data_path/ "*" / "*.cu3s"))
            cube_names = [Path(image).name for image in cubes]
            dataset_name = data_path.name

            # create dataset and infer the cubes
            dataset = EfficientADCuvisDataSet(config["datasets"][0],
                                              mode="test",
                                              mean=config["means"],
                                              std=config["stds"],
                                              normalize=config["normalize"],
                                              max_img_shape=config["max_img_shape"],)
            test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            pred = self.trainer.predict(self.model, test_loader)

            max_pred = 0
            labels = []
            scores = []
            for p in pred:
                labels.extend(p["label"])
                if p["label"] is not np.nan: all_labels.extend(p["label"])
                score = torch.max(p["anomaly_map"])
                # score = torch.topk(p["anomaly_map"].flatten(), k=500)[0].sum()
                scores.append(score)
                all_scores.append(score)
                if p["anomaly_map"].max().item() > max_pred:
                    max_pred = p["anomaly_map"].max().item()
            if self.create_images:
                for j, batch in enumerate(tqdm.tqdm(test_loader, desc=f"creating images")):
                    rel_path = Path(cubes[j]).relative_to(dataset_path)
                    target_folder = self.reporting_run_folder / dataset_name / rel_path.parent
                    if not target_folder.is_dir():
                        target_folder.mkdir(parents=True, exist_ok=True)
                    inference_image, _ = self.create_inference_png(batch, pred[j]['anomaly_map'].squeeze().detach().cpu().numpy(), labels[j], scores[j], cube_names[j])
                    inference_image.savefig(target_folder / (cube_names[j] + ".png"))
                    plt.close(inference_image)
            if 0 in labels and 1 in labels:
                auroc = roc_auc_score(labels, scores)
                fpr, tpr, roc_thresholds = roc_curve(labels, scores)
            else:
                auroc = torch.tensor(0)
                fpr = tpr = roc_thresholds = np.array(0)

            auroc_plot = self.plot_curve(fpr, tpr, f"ROC curve (AUC = {auroc:.2f})", "ROC-curve", "False Positive Rate", "True Positive Rate")
            auroc_plot.savefig(self.reporting_run_folder / (dataset_name + "_AUROC.png"), dpi=300, bbox_inches="tight")
            auroc_plot.close()
            metrics[dataset_name] = {"AU-ROC": auroc.item(), "FPR": fpr.tolist(), "TPR": tpr.tolist(), "ROC-thresholds": roc_thresholds.tolist()}
        if 0 in all_labels and 1 in all_labels:
            auroc = roc_auc_score(all_labels, all_scores)
            fpr, tpr, roc_thresholds = roc_curve(all_labels, all_scores)
        else:
            auroc = torch.tensor(0)
            fpr = tpr = roc_thresholds = np.array(0)
        auroc_plot = self.plot_curve(fpr, tpr, f"ROC curve (AUC = {auroc:.2f})", "ROC-curve", "False Positive Rate", "True Positive Rate")
        auroc_plot.savefig(self.reporting_run_folder / "overall_AUROC.png", dpi=300, bbox_inches="tight")
        auroc_plot.close()
        metrics["overall"] = {"AU-ROC": auroc.item(), "FPR": fpr.tolist(), "TPR": tpr.tolist(), "ROC-thresholds": roc_thresholds.tolist()}
        with open(self.reporting_run_folder / "metrics.yaml", "w") as f:
            yaml.dump(metrics, f)

    def create_inference_png(self, batch, pred, label, score, image_name):
        """
        Creates one image
        :param batch: input batch of the dataset class
        :param pred: prediction outcome of the model
        :param label: label of the image
        :param score: calculated anomalie score
        :param image_name: name of the image
        :return: pyplot figure and axis
        """
        nrows = 2 + len(self.plot_thresholds)
        fig_height = 6 + len(self.plot_thresholds)
        fig, ax = plt.subplots(nrows, 2, tight_layout=True, dpi=600, figsize=(10, fig_height))
        a_map = pred
        a_map = a_map + abs(a_map.min())

        # split image in RGB and SWIR, or double the first channels if it is a SWIR or RGB only model
        img = batch['image'].squeeze().detach().cpu().permute(1, 2, 0).numpy()
        if img.shape[2] > 3:
            rgb = img[:, :, :3]
            rgb = rgb[:, :, [2, 1, 0]]
            ir = img[:, :, 3:]
        else:
            rgb = img[:, :, [2, 1, 0]]
            ir = rgb

        # revert normalization to display images correctly
        if bool(self.config["normalize"]):
            if img.shape[2] > 3:
                rgb = rgb * self.std[:3][[2, 1, 0]] + self.mean[:3][[2, 1, 0]]
                ir = ir * self.std[3:] + self.mean[3:]
            elif self.config["channels"] == "SWIR":
                rgb = rgb * self.std[3:] + self.mean[3:]
                ir = ir * self.std[3:] + self.mean[3:]
            else:
                rgb = rgb * self.std[:3][[2, 1, 0]] + self.mean[:3][[2, 1, 0]]
                ir = ir * self.std[:3][[2, 1, 0]] + self.mean[:3][[2, 1, 0]]

        # clip reflectance value to have a nicer image
        rgb[rgb > 1.2] = 1.2
        rgb = rgb / 1.2
        ir[ir > 1.9] = 1.9
        ir = ir / 1.9

        ax[0][0].imshow((rgb * 255).astype(np.uint8), vmax=255, vmin=0)
        ax[0][1].imshow((ir * 255).astype(np.uint8), vmax=255, vmin=0)
        ax[0][0].set_title('RGB')
        ax[0][1].set_title('IR')
        ax[1][0].imshow((a_map * 255).astype(np.uint8), vmax=255, vmin=0)
        ax[1][0].set_title('anomaly_map')
        if self.config["overlay"] == "RGB":
            overlay = rgb.copy()
        else:
            overlay = ir.copy()
        mask = a_map > np.array(self.plot_thresholds).min()
        overlay[mask, 1] = a_map[mask]
        ax[1][1].imshow((overlay * 255).astype(np.uint8), vmax=255, vmin=0)
        ax[1][1].set_title('RGB overlay')

        # create threshold images and threshold overlays
        for i, threshold in enumerate(self.plot_thresholds):
            a_map_threshold = a_map.copy()
            a_map_threshold[a_map_threshold < threshold] = 0
            a_map_threshold[a_map_threshold >= threshold] = 1
            ax[2 + i][0].set_title(f'anomaly_map_threshold: {threshold}')
            if self.config["overlay"] == "RGB":
                overlay = rgb.copy()
            else:
                overlay = ir.copy()
            overlay[a_map_threshold == 1] = [1, 0, 0]
            ax[2 + i][0].imshow((a_map_threshold * 255).astype(np.uint8), vmax=255, vmin=0)
            ax[2 + i][1].imshow((overlay * 255).astype(np.uint8), vmax=255, vmin=0)
            ax[2 + i][1].set_title('RGB overlay with threshold')

        fig.suptitle(image_name)
        return fig, ax

    def plot_curve(self, x, y, label, title, x_label="", y_label="", color="navy", legend_position="lower right"):
        """
        creates a pyplot
        :param x: X-axis
        :param y: Y-axis
        :param label: label of the curve
        :param title: title of the curve
        :param x_label: label of the X-axis
        :param y_label: label of the Y-axis
        :param color: color of the curve
        :param legend_position: position of the legend
        :return: pyplot plot
        """
        fig = plt.figure()
        plt.plot(x, y, color=color, lw=2, label=label)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(loc=legend_position)
        plt.grid()
        return plt


if __name__ == "__main__":
    args = get_arguments()
    config = parse_args(args)
    model = EfficientAD_lightning.load_from_checkpoint(config["checkpoint_to_load"], config=config)
    trainer = L.Trainer(inference_mode=True, precision='16-mixed')
    rep = Report(config, model, trainer, Path("../data/EAD_reporting/"))
    rep.generate_report()
