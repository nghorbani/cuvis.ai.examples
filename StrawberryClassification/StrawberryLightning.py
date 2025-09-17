import itertools

import lightning as L
import torchvision.transforms

from UNet import FreshTwin2DUNet
import torch
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import AveragePrecision, Accuracy, ROC, AUROC
from torchvision.transforms.functional import equalize
import torch.nn.functional as F
import cv2 as cv
from gpu_pca import IncrementalPCAonGPU as IncPca
from tqdm import tqdm
import matplotlib.pyplot as plt


class GeneralizedDiceLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-6, generalize=False, mode=None):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.generalize = generalize
        self.mode = mode

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] - predicted probabilities (after softmax or sigmoid)
            targets: [B, H, W] or [B, C, H, W] - ground truth class indices or one-hot

        Returns:
            loss: scalar
        """

        if inputs.dim() != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {inputs.shape}")

        B, C, H, W = inputs.shape

        # Convert targets to one-hot if needed
        if targets.shape != inputs.shape:
            targets = F.one_hot(targets.long(), num_classes=C)  # [B, H, W, C]
            targets = targets.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        inputs = inputs.float()
        targets = targets.float()

        # Flatten spatial dimensions: [B, C, H*W]
        inputs_flat = inputs.view(B, C, -1)
        targets_flat = targets.view(B, C, -1)

        # Compute class weights: 1 / (sum of ground truth per class)^2
        if self.mode == "advanced":
            gt_sum = targets_flat.sum(-1)  # [B, C]
            class_weights = 1.0 / (gt_sum ** 2 + self.epsilon)  # [B, C]
        else:
            class_weights = torch.tensor([0, 1, 1], device="cuda")  # [B, C]

        # Numerator and denominator
        intersection = (inputs_flat * targets_flat).sum(-1)  # [B, C]
        union = (inputs_flat + targets_flat).sum(-1)  # [B, C]

        if self.generalize:
            denominator = (class_weights * union).sum(1)
            numerator = (class_weights * intersection).sum(1)
        else:
            denominator = union.sum(1)
            numerator = intersection.sum(1)

        dice_score = 2 * numerator / (denominator + self.epsilon)
        loss = 1 - dice_score

        return loss.mean()


class StrawberryLightning(L.LightningModule):
    def __init__(self, config, data_loader):
        super().__init__()
        self.pca_channels = config["pca_channels"]
        self.cube_channels = config["cube_channels"]
        self.pca = IncPca(n_components=self.pca_channels)
        self.learning_rate = config["learning_rate"] if "learning_rate" in config else 1e-3
        self.weight_decay = config["weight_decay"] if "weight_decay" in config else 1e-5
        self.num_classes = config["num_classes"]
        self.model = FreshTwin2DUNet(in_channels=self.pca_channels, num_classes=self.num_classes, pca=self.pca)

        self.roc = ROC(task='multiclass', num_classes=config["num_classes"])
        self.auroc = AUROC(task='multiclass', num_classes=config["num_classes"])
        self.ap = AveragePrecision(task='multiclass', num_classes=config["num_classes"])
        self.acc1 = Accuracy(task='multiclass', num_classes=config["num_classes"], threshold=0.1)
        self.acc2 = Accuracy(task='multiclass', num_classes=config["num_classes"], threshold=0.2)
        self.acc3 = Accuracy(task='multiclass', num_classes=config["num_classes"], threshold=0.3)
        self.acc4 = Accuracy(task='multiclass', num_classes=config["num_classes"], threshold=0.4)
        self.acc5 = Accuracy(task='multiclass', num_classes=config["num_classes"], threshold=0.5)
        self.dice = DiceScore(num_classes=4, include_background=False, input_format='one-hot')
        self.dice_loss = DiceScore(num_classes=4, include_background=False, input_format='one-hot')
        self.dice_bg = DiceScore(num_classes=4, include_background=True)
        self.dice1 = DiceScore(num_classes=4, include_background=False, input_format='one-hot')
        self.dice2 = DiceScore(num_classes=4, include_background=False, input_format='one-hot')
        self.dice3 = DiceScore(num_classes=4, include_background=False, input_format='one-hot')
        self.dice4 = DiceScore(num_classes=4, include_background=False, input_format='one-hot')
        self.dice5 = DiceScore(num_classes=4, include_background=False, input_format='one-hot')
        self.iou_bg = MeanIoU(num_classes=4, include_background=True)
        self.iou1 = MeanIoU(num_classes=4, include_background=False)
        self.iou2 = MeanIoU(num_classes=4, include_background=False)
        self.iou3 = MeanIoU(num_classes=4, include_background=False)
        self.iou4 = MeanIoU(num_classes=4, include_background=False)
        self.iou5 = MeanIoU(num_classes=4, include_background=False)
        self.to_PIL = torchvision.transforms.ToPILImage()
        self.val_loss = []
        self.train_loss = []
        self.save_imgs = False
        self.data_loader = data_loader
        self.image_height, self.image_width = config["cube_size"]

    def calc_pca(self, cube):
        pca_image = cube.squeeze(0).permute(1, 2, 0).reshape(-1, self.cube_channels)
        pca_image = self.pca.transform(pca_image)
        pca_image = pca_image.reshape(1, self.image_height, self.image_width, self.pca_channels).permute(0, 3, 1, 2).type(torch.float16)
        return pca_image

    def setup(self, stage):
        if self.pca.n_samples_seen_ == 0:
            for batch in tqdm(self.data_loader, desc="PCA fit"):
                pca_image = batch["image"].squeeze(0).permute(1, 2, 0).reshape(-1, self.cube_channels)
                self.pca.partial_fit(pca_image)

    def training_step(self, batch, batch_idx):
        res = self.model.forward(batch["image"])
        pred = torch.softmax(res, dim=0)

        one_hot_gt = torch.nn.functional.one_hot(batch["mask"].type(torch.long), num_classes=self.num_classes).movedim(-1, 1)

        if batch["number"][0] + "_" + batch["side"][0] + "_" + batch["day"][0] == "008_3_22" and False or self.save_imgs:

            gt_mask = batch["mask"]
            rgb_mask = torch.zeros((self.image_height, self.image_width, 3))
            rgb_mask[:, :, 0] = (gt_mask == 1).float()  # Red channel
            rgb_mask[:, :, 2] = (gt_mask == 2).float()  # Blue channel
            rgb_mask = cv.cvtColor((rgb_mask * 255).type(torch.uint8).detach().cpu().numpy(), cv.COLOR_BGR2RGB)

            prediction = torch.argmax(res, dim=0, keepdim=True)
            rgb_pred = torch.zeros((self.image_height, self.image_width, 3)).type(torch.uint8)
            rgb_pred[:, :, 0] = (prediction == 1) * 255  # Red channel
            rgb_pred[:, :, 1] = (prediction == 3) * 255  # Green channel
            rgb_pred[:, :, 2] = (prediction == 2) * 255  # Blue channel
            rgb_pred = rgb_pred.detach().cpu().numpy()
            if self.save_imgs:
                cv.imwrite(self.trainer.log_dir + "/images/" + batch["number"][0] + "_" + batch["side"][0] + "_" + batch["day"][
                    0] + "_E" + str(self.trainer.current_epoch) + ".png", rgb_pred)
                if self.trainer.current_epoch == 0:
                    cv.imwrite(self.trainer.log_dir + "/images/" + batch["number"][0] + "_" + batch["side"][0] + "_" + batch["day"][0] + ".png",
                               cv.cvtColor(batch["rgb_image"].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255, cv.COLOR_BGR2RGB))
                    cv.imwrite(self.trainer.log_dir + "/images/" + batch["number"][0] + "_" + batch["side"][0] + "_" + batch["day"][0] + "_mask.png",
                               rgb_mask)
            else:
                cv.imshow("008_3_22", cv.resize(rgb_pred, None, None, 4, 4))
                cv.imshow("008_3_22_mask", cv.resize(cv.normalize(rgb_mask, None, 0, 255, cv.NORM_MINMAX), None, None, 4, 4))
                cv.imshow("008_3_22_rgb",
                          cv.resize(cv.normalize(batch["rgb_image"].detach().cpu().squeeze().numpy(), None, 0, 255, cv.NORM_MINMAX),
                                    None,
                                    None,
                                    4,
                                    4))
                cv.waitKey()

        loss = GeneralizedDiceLoss(generalize=True)(pred.unsqueeze(0), one_hot_gt)

        self.train_loss.append(loss.item())
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        res = self.model.forward(batch["image"])
        gt_mask = batch['mask']
        pred = torch.softmax(res, dim=0).unsqueeze(0)
        one_hot_gt = torch.nn.functional.one_hot(gt_mask.type(torch.long), num_classes=self.num_classes).movedim(-1, 1)
        loss = GeneralizedDiceLoss(generalize=True)(pred, one_hot_gt).item()
        self.val_loss.append(loss)

        prediction = torch.argmax(res, dim=0, keepdim=True)

        one_hot_pred = torch.nn.functional.one_hot(prediction.type(torch.long), num_classes=self.num_classes).movedim(-1, 1)

        self.dice.update(one_hot_pred, one_hot_gt)
        self.dice_bg.update(one_hot_pred > 0.5, one_hot_gt)
        self.dice1.update(one_hot_pred > 0.1, one_hot_gt)
        self.dice2.update(one_hot_pred > 0.2, one_hot_gt)
        self.dice3.update(one_hot_pred > 0.3, one_hot_gt)
        self.dice4.update(one_hot_pred > 0.4, one_hot_gt)
        self.dice5.update(one_hot_pred > 0.5, one_hot_gt)

        self.iou_bg.update(prediction > 0.5, gt_mask)
        self.iou1.update(prediction > 0.1, gt_mask)
        self.iou2.update(prediction > 0.2, gt_mask)
        self.iou3.update(prediction > 0.3, gt_mask)
        self.iou4.update(prediction > 0.4, gt_mask)
        self.iou5.update(prediction > 0.5, gt_mask)

        self.ap.update(res.unsqueeze(0), gt_mask)

        self.acc1.update(res.unsqueeze(0), gt_mask)
        self.acc2.update(res.unsqueeze(0), gt_mask)
        self.acc3.update(res.unsqueeze(0), gt_mask)
        self.acc4.update(res.unsqueeze(0), gt_mask)
        self.acc5.update(res.unsqueeze(0), gt_mask)

        self.roc.update(res.unsqueeze(0), gt_mask)
        self.auroc.update(res.unsqueeze(0), gt_mask)

        if batch_idx % 10 == 0:
            input_img = batch["image"][0, 0].unsqueeze(0)
            rgb_img = batch["rgb_image"].squeeze(0)
            rgb_pred = torch.zeros(3, self.image_height, self.image_width)
            rgb_pred[0] = (prediction == 1).float()  # Red channel
            rgb_pred[2] = (prediction == 2).float()  # Blue channel
            eq_pred = equalize((255.0 * rgb_pred).to(torch.uint8))
            rgb_mask = torch.zeros(3, self.image_height, self.image_width)
            rgb_mask[0] = (gt_mask == 1).float()  # Red channel
            rgb_mask[2] = (gt_mask == 2).float()  # Blue channel
            eq_pca = equalize((255.0 * (input_img - input_img.min()) / (input_img.max() - input_img.min())).to(torch.uint8))
            name = batch["number"][0] + "_" + batch["side"][0] + "_" + batch["day"][0]
            self.logger.experiment.add_image(f'{name}/3_pred_equalized', eq_pred.squeeze(0), global_step=self.global_step)
            self.logger.experiment.add_image(f'{name}/2_gt_mask', rgb_mask, global_step=self.global_step)
            self.logger.experiment.add_image(f'{name}/1_pca', eq_pca, global_step=self.global_step)
            self.logger.experiment.add_image(f'{name}/0_rgb', rgb_img, global_step=self.global_step)
            self.logger.experiment.add_scalar(f'{name}/loss', loss, global_step=self.global_step)

        if self.save_imgs:
            rgb_pred = torch.zeros((self.image_height, self.image_width, 3)).type(torch.uint8)
            rgb_pred[:, :, 0] = (prediction == 1) * 255  # Red channel
            rgb_pred[:, :, 2] = (prediction == 2) * 255  # Blue channel
            rgb_pred = rgb_pred.detach().cpu().numpy()
            rgb_mask = torch.zeros(3, self.image_height, self.image_width)
            rgb_mask[0] = (gt_mask == 1).float()  # Red channel
            rgb_mask[2] = (gt_mask == 2).float()  # Blue channel
            rgb_mask = (rgb_mask * 255).type(torch.uint8)
            if self.save_imgs:
                cv.imwrite(self.trainer.log_dir + "/images/" + batch["number"][0] + "_" + batch["side"][0] + "_" + batch["day"][
                    0] + "_E" + str(self.trainer.current_epoch) + ".png", rgb_pred)
                if self.trainer.current_epoch == 0:
                    cv.imwrite(self.trainer.log_dir + "/images/" + batch["number"][0] + "_" + batch["side"][0] + "_" + batch["day"][0] + ".png",
                               cv.cvtColor(batch["rgb_image"].squeeze(0).permute(1, 2, 0).cpu().numpy(), cv.COLOR_BGR2RGB))
                    cv.imwrite(self.trainer.log_dir + "/images/" + batch["number"][0] + "_" + batch["side"][0] + "_" + batch["day"][0] + "_mask.png",
                               cv.cvtColor(rgb_mask.permute(1, 2, 0).cpu().numpy(), cv.COLOR_BGR2RGB))

        return prediction  # res

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(itertools.chain(self.parameters()),
                                     lr=self.learning_rate, weight_decay=self.weight_decay)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2),
                "monitor": "train/epoch_loss",
                "frequency": 1,
                "interval": "epoch"
            },
        }

    def on_validation_start(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        """Called after every validation epoch. Logs all accumulated data and clears buffers."""
        self.log('val_im/AP', self.ap, on_epoch=True)
        self.log('val_im/Acc_0.1', self.acc1, on_epoch=True)
        self.log('val_im/Acc_0.2', self.acc2, on_epoch=True)
        self.log('val_im/Acc_0.3', self.acc3, on_epoch=True)
        self.log('val_im/Acc_0.4', self.acc4, on_epoch=True)
        self.log('val_im/Acc_0.5', self.acc5, on_epoch=True)

        self.log('val_IoU/t=0.5_BG', self.iou_bg.compute(), on_epoch=True)
        self.log('val_IoU/t=0.1', self.iou1, on_epoch=True)
        self.log('val_IoU/t=0.2', self.iou2, on_epoch=True)
        self.log('val_IoU/t=0.3', self.iou3, on_epoch=True)
        self.log('val_IoU/t=0.4', self.iou4, on_epoch=True)
        self.log('val_IoU/t=0.5', self.iou5, on_epoch=True)
        self.log('val_F1/t=0.5_BG', self.dice_bg, on_epoch=True)
        self.log('val_F1/dice', self.dice, on_epoch=True)
        self.log('val_F1/t=0.1', self.dice1, on_epoch=True)
        self.log('val_F1/t=0.2', self.dice2, on_epoch=True)
        self.log('val_F1/t=0.3', self.dice3, on_epoch=True)
        self.log('val_F1/t=0.4', self.dice4, on_epoch=True)
        self.log('val_F1/t=0.5', self.dice5, on_epoch=True)

        self.log('val_ROC/AUROC', self.auroc, on_epoch=True)
        roc_fig, _ = self.roc.plot(score=True)
        self.logger.experiment.add_figure('curve/ROC', roc_fig, global_step=self.global_step)
        plt.close(roc_fig)
        self.roc.reset()

        val_loss = sum(self.val_loss) / len(self.val_loss)
        self.val_loss = None
        self.val_loss = []
        self.log('val/epoch_loss', val_loss, on_epoch=True)

    def on_train_start(self) -> None:
        pass

    def on_train_epoch_end(self) -> None:

        train_loss = sum(self.train_loss) / len(self.train_loss)
        self.train_loss = None
        self.train_loss = []
        self.log('train/epoch_loss', train_loss, on_epoch=True)

    def on_train_end(self) -> None:
        pass

    def forward(self, image):
        res = self.model.forward(image["image"])
        return res

    def on_save_checkpoint(self, checkpoint):
        checkpoint['pca'] = self.pca

    def on_load_checkpoint(self, checkpoint):
        self.pca = checkpoint['pca']
        self.model.pca = checkpoint['pca']
