import pathlib

import lightning
import numpy as np

from EfficientADCuvisDataSet import EfficientADCuvisDataSet
import yaml
import torch
import lightning as L
from EfficientAD_lightning import EfficientAD_lightning
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
import argparse
import os
import tqdm
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
import json
import cv2 as cv
import glob
from sklearn.metrics import roc_curve, auc
from collections import defaultdict

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
        self.create_roc = config['create_roc']
        self.annotations = json.load(open(config["annotations"])) if config["annotations"] != "" else None


    def plot_pixel_level_roc(self, anomaly_score_maps, groundtruth_masks, normalize=True):
        """
        Plot a pixel-level ROC curve and compute the overall AUC.

        This method computes a Receiver Operating Characteristic (ROC) curve by evaluating
        the anomaly score maps against binary ground truth masks. All non-zero pixel labels
        in the ground truth masks are treated as positives (foreground), and zeros as negatives (background).

        Parameters
        ----------
        anomaly_score_maps : List[np.ndarray]
            A list of 2D arrays containing pixel-wise anomaly scores for each image.

        groundtruth_masks : List[np.ndarray]
            A list of 2D arrays with ground truth segmentation masks. All non-zero pixels
            are considered foreground (label = 1), and zero pixels are considered background (label = 0).

        normalize : bool, optional (default=True)
            Whether to normalize each anomaly score map to the [0, 1] range before computing the ROC.

        Returns
        -------
        tuple
            A tuple containing:
            - fpr (np.ndarray): False Positive Rates.
            - tpr (np.ndarray): True Positive Rates.
            - roc_auc (float): Area Under the ROC Curve (AUC).

        Raises
        ------
        AssertionError
            If the number of anomaly score maps and ground truth masks differ,
            or if any score map and corresponding ground truth mask have different shapes.

        ValueError
            If the binary ground truth contains only one class (all 0s or all 1s), making ROC computation invalid.
        """
        assert len(anomaly_score_maps) == len(groundtruth_masks), "Mismatch in number of images"
        # In this scenario, convert pixels to binary values
        groundtruth_masks = groundtruth_masks > 0
        y_scores = []
        y_true = []

        for score_map, gt_mask in zip(anomaly_score_maps, groundtruth_masks):
            assert score_map.shape == gt_mask.shape, "Shape mismatch between score and groundtruth"

            if normalize:
                score_min, score_max = score_map.min(), score_map.max()
                score_map = (score_map - score_min) / (score_max - score_min + 1e-8)

            y_scores.append(score_map.flatten())
            y_true.append(gt_mask.flatten())

        y_scores = np.concatenate(y_scores)
        y_true = np.concatenate(y_true)

        if len(np.unique(y_true)) < 2:
            raise ValueError("ROC is undefined. Ground truth must have both 0 and 1 values.")

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plotting
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="blue")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Pixel-Level ROC Curve")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.reporting_run_folder}/AUROC.png', dpi=300, bbox_inches="tight")
        plt.close()
        return fpr, tpr, roc_auc

    def plot_per_class_pixel_roc(self, anomaly_score_maps, groundtruth_masks, normalize=True, class_map=None):
        """
        Optimized memory-efficient version of per-class pixel-level ROC and AUC computation.
        """
        assert len(anomaly_score_maps) == len(groundtruth_masks), "Mismatch in number of images"

        inverted_class_map = {v: k for k, v in class_map.items()} if class_map else {}

        class_scores = defaultdict(list)
        class_truths = defaultdict(list)
        present_classes = set()

        for score_map, gt_mask in zip(anomaly_score_maps, groundtruth_masks):
            assert score_map.shape == gt_mask.shape, "Score and mask shape mismatch"

            if normalize:
                score_min, score_max = np.min(score_map), np.max(score_map)
                denom = score_max - score_min
                if denom > 1e-8:
                    score_map = (score_map - score_min) / denom
                else:
                    score_map = np.zeros_like(score_map)  # handle constant maps

            for cls in np.unique(gt_mask):
                if cls == 0:
                    continue  # skip background
                present_classes.add(cls)
                mask = (gt_mask == cls)
                class_scores[cls].append(score_map[mask])
                class_truths[cls].append(np.ones(np.count_nonzero(mask)))
                # Add negative samples from other classes
                neg_mask = (gt_mask == 0)
                class_scores[cls].append(score_map[neg_mask])
                class_truths[cls].append(np.zeros(np.count_nonzero(neg_mask)))

        plt.figure(figsize=(7, 7))
        aucs = {}

        for cls in sorted(present_classes):
            if cls not in class_scores:
                continue

            y_scores = np.concatenate(class_scores[cls])
            y_true = np.concatenate(class_truths[cls])

            if len(np.unique(y_true)) < 2:
                print(f"Skipping class {cls}: insufficient positive/negative pixels")
                continue

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            aucs[cls] = roc_auc

            cls_label = inverted_class_map.get(cls, f"Class {cls}")
            plt.plot(fpr, tpr, label=f"{cls_label} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Per-Class Pixel-Level ROC Curve")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(self.reporting_run_folder, exist_ok=True)
        plt.savefig(f'{self.reporting_run_folder}/AUROC_Class.png', dpi=300, bbox_inches="tight")
        plt.close()

        return {inverted_class_map.get(k, f"Class {k}"): float(v) for k, v in aucs.items()}


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
        all_truths = []
        all_scores = []
        for dataset_path in config['datasets']:
            data_path = Path(dataset_path)
            cubes = glob.glob(str(data_path/ "*" / "*.cu3s"))
            cube_names = [Path(image).name for image in cubes]
            # Load only the PNG masks associated with the labels
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
                # Labels are the binary anomalous or not
                labels.extend(p["label"])
                if p["label"] is not np.nan: all_labels.extend(p["label"])
                # anomaly map is the numerical value of reconstruction errors generated by the network
                score = torch.max(p["anomaly_map"])
                scores.append(score)
                all_scores.append(p["anomaly_map"].squeeze(0).squeeze(0).detach().cpu().numpy())
                # Append the corresponding image label
                all_truths.append(p["mask"].squeeze(0).numpy())
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
        if self.create_roc:
            fpr, tpr, roc_auc = self.plot_pixel_level_roc(np.array(all_scores), np.array(all_truths))
            class_aucs = self.plot_per_class_pixel_roc(np.array(all_scores).astype(np.float32), np.array(all_truths).astype(np.float32), class_map=self.annotations)
            metrics[dataset_name] = {
                'overall_auc': float(roc_auc),
                'per_class_auc': class_aucs
            }

        with open(self.reporting_run_folder / "metrics.yaml", "w") as f:
            yaml.dump(metrics, f)

    def create_inference_png(self, batch, pred, label, score, image_name):
        """
        Creates one image
        :param batch: input batch of the dataset class
        :param pred: prediction outcome of the model
        :param label: label of the image
        :param score: calculated anomaly score
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
