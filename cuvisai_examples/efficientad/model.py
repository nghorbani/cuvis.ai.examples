"""Torch model for student, teacher and autoencoder model in EfficientAd."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import math

from loguru import logger
import torch
from torch import dtype, nn
from torch.nn import functional as F  # noqa: N812


def reshape_if_nessecary(x: torch.Tensor) -> torch.Tensor:
    """
    unsqueezes the tensor in case it only has a length of 3
    Parameters
    ----------
    x

    Returns
    -------

    """
    if x.shape.__len__() == 3:
        x = x.unsqueeze(0)
    return x.float()


def reduce_tensor_elems(tensor: torch.Tensor, m: int = 2**24) -> torch.Tensor:
    """Reduce tensor elements.

    This function flatten n-dimensional tensors,  selects m elements from it
    and returns the selected elements as tensor. It is used to select
    at most 2**24 for torch.quantile operation, as it is the maximum
    supported number of elements.
    https://github.com/pytorch/pytorch/blob/b9f81a483a7879cd3709fd26bcec5f1ee33577e6/aten/src/ATen/native/Sorting.cpp#L291.

    Args:
        tensor (torch.Tensor): input tensor from which elements are selected
        m (int): number of maximum tensor elements.
            Defaults to ``2**24``

    Returns:
            Tensor: reduced tensor
    """
    tensor = torch.flatten(tensor)
    if len(tensor) > m:
        # select a random subset with m elements.
        perm = torch.randperm(len(tensor), device=tensor.device)
        idx = perm[:m]
        tensor = tensor[idx]
    return tensor


class EfficientAdModelSize(str, Enum):
    """Supported EfficientAd model sizes."""

    M = "medium"
    S = "small"


class SmallPatchDescriptionNetwork(nn.Module):
    """Patch Description Network small.

    Args:
        out_channels (int): number of convolution output channels
        padding (bool): use padding in convoluional layers
            Defaults to ``False``.
        in_channels(int): number of input channels
    """

    def __init__(self, out_channels: int, padding: bool = False, in_channels: int = 6) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Output from the network.
        """
        if x.shape.__len__() == 3:
            x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        return self.conv4(x)


class MediumPatchDescriptionNetwork(nn.Module):
    """Patch Description Network medium.

    Args:
        out_channels (int): number of convolution output channels
        padding (bool): use padding in convoluional layers
            Defaults to ``False``.
        in_channels(int): number of input channels
    """

    def __init__(self, out_channels: int, padding: bool = False, in_channels: int = 6) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Output from the network.
        """

        if x.shape.__len__() == 3:
            x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.conv6(x)


class Encoder(nn.Module):
    """Autoencoder Encoder model.

    Args:
          in_channels(int): number of input channels
    """

    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass through the network.

        Args:
            x (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Output from the network.
        """
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        return self.enconv6(x)


class Decoder(nn.Module):
    """Autoencoder Decoder model.

    Args:
        out_channels (int): number of convolution output channels
        padding (int): use padding in convoluional layers
        in_channels(int): number of input channels
    """

    def __init__(self, out_channels: int, padding: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.padding = padding
        # use ceil to match output shape of PDN
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, image_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input batch.
            image_size (tuple): size of input images.

        Returns:
            torch.Tensor: Output from the network.
        """
        last_upsample = (
            math.ceil(image_size[0] / 4) if self.padding else math.ceil(image_size[0] / 4) - 8,
            math.ceil(image_size[1] / 4) if self.padding else math.ceil(image_size[1] / 4) - 8,
        )
        x = F.interpolate(x, size=(image_size[0] // 64 - 1, image_size[1] // 64 - 1), mode="bilinear")
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=(image_size[0] // 32, image_size[1] // 32), mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=(image_size[0] // 16 - 1, image_size[1] // 16 - 1), mode="bilinear")
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=(image_size[0] // 8, image_size[1] // 8), mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=(image_size[0] // 4 - 1, image_size[1] // 4 - 1), mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=(image_size[0] // 2 - 1, image_size[1] // 2 - 1), mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        return self.deconv8(x)


class AutoEncoder(nn.Module):
    """EfficientAd Autoencoder.

    Args:
       out_channels (int): number of convolution output channels
       padding (int): use padding in convoluional layers
       in_channels(int): number of input channels
    """

    def __init__(self, out_channels: int, padding: int, in_channels: int = 6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels, padding)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor, image_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        """Perform the forward pass through the network.

        Args:
            x (torch.Tensor): Input batch.
            image_size (tuple): size of input images.

        Returns:
            torch.Tensor: Output from the network.
        """
        if x.shape.__len__() == 3:
            x = x.unsqueeze(0)
        x = self.encoder(x)
        return self.decoder(x, image_size)
        # return self.decoder(x)


class EfficientAdModel(nn.Module):
    """EfficientAd model.

    Args:
        teacher_out_channels (int): number of convolution output channels of the pre-trained teacher model
        model_size (str): size of student and teacher model
        padding (bool): use padding in convoluional layers
            Defaults to ``False``.
        pad_maps (bool): relevant if padding is set to False. In this case, pad_maps = True pads the
            output anomaly maps so that their size matches the size in the padding = True case.
            Defaults to ``True``.
        in_channels(int): number of input channels
        use_imgnet_penalty(bool): weather to use the imgNet penalty in training
    """

    def __init__(
        self,
        teacher_out_channels: int,
        model_size: EfficientAdModelSize = EfficientAdModelSize.S,
        padding: bool = False,
        pad_maps: bool = True,
        in_channels: int = 6,
        use_imgnet_penalty: bool = False,
    ) -> None:
        super().__init__()
        self.pad_maps = pad_maps
        self.teacher: MediumPatchDescriptionNetwork | SmallPatchDescriptionNetwork
        self.student: MediumPatchDescriptionNetwork | SmallPatchDescriptionNetwork

        if model_size == EfficientAdModelSize.M:
            self.teacher = MediumPatchDescriptionNetwork(
                out_channels=teacher_out_channels,
                padding=padding,
                in_channels=in_channels,
            ).eval()
            self.student = MediumPatchDescriptionNetwork(
                out_channels=teacher_out_channels * 2,
                padding=padding,
                in_channels=in_channels,
            )

        elif model_size == EfficientAdModelSize.S:
            self.teacher = SmallPatchDescriptionNetwork(
                out_channels=teacher_out_channels,
                padding=padding,
                in_channels=in_channels,
            ).eval()
            self.student = SmallPatchDescriptionNetwork(
                out_channels=teacher_out_channels * 2,
                padding=padding,
                in_channels=in_channels,
            )

        else:
            msg = f"Unknown model size {model_size}"
            raise ValueError(msg)

        self.autoencoder: AutoEncoder = AutoEncoder(
            out_channels=teacher_out_channels, padding=padding, in_channels=in_channels
        )
        self.teacher_out_channels: int = teacher_out_channels

        # self.mean_std: nn.ParameterDict = nn.ParameterDict(
        #     {
        #         "mean": torch.zeros((1, self.teacher_out_channels, 1, 1)),
        #         "std": torch.zeros((1, self.teacher_out_channels, 1, 1)),
        #     },
        # )

        # self.quantiles: nn.ParameterDict = nn.ParameterDict(
        #     {
        #         "qa_st": torch.tensor(0.0),
        #         "qb_st": torch.tensor(0.0),
        #         "qa_ae": torch.tensor(0.0),
        #         "qb_ae": torch.tensor(0.0),
        #     },
        # )

        # Teacher normalization stats
        self.register_buffer(
            "teacher_mean", torch.zeros(1, self.teacher_out_channels, 1, 1, dtype=torch.float32), persistent=True
        )
        # std should start at 1 (not 0) to avoid division-by-zero before calibration
        self.register_buffer(
            "teacher_std", torch.ones(1, self.teacher_out_channels, 1, 1, dtype=torch.float32), persistent=True
        )

        # Quantile scalars for map normalization
        # Persist them if you want them saved/restored with checkpoints
        self.register_buffer("qa_st", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("qb_st", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("qa_ae", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("qb_ae", torch.tensor(0.0, dtype=torch.float32), persistent=False)

        self.use_imgnet_penalty = use_imgnet_penalty

    @torch.no_grad()
    def set_teacher_stats(self, mean, std):
        self.teacher_mean.copy_(mean.to(self.teacher_mean.dtype).to(self.teacher_mean.device))
        self.teacher_std.copy_(std.to(self.teacher_std.dtype).to(self.teacher_std.device).clamp_min(1e-6))

    @torch.no_grad()
    def set_quantiles(self, qa_st, qb_st, qa_ae, qb_ae):
        self.qa_st.fill_(float(qa_st))
        self.qb_st.fill_(float(qb_st))
        self.qa_ae.fill_(float(qa_ae))
        self.qb_ae.fill_(float(qb_ae))

    @staticmethod
    def is_set(p_value) -> bool:
        """Check if any of the parameters in the parameter dictionary is set.

        Args:
            p_dic (nn.ParameterDict): Parameter dictionary.

        Returns:
            bool: Boolean indicating whether any of the parameters in the parameter dictionary is set.
        """
        return p_value.sum() != 0

    def forward(
        self,
        batch: torch.Tensor,
        batch_imagenet: torch.Tensor | None = None,
        normalize: bool = True,
        return_all_maps: bool = False,
    ) -> torch.Tensor | dict:
        """Perform the forward-pass of the EfficientAd models.

        Args:
            batch (torch.Tensor): Input images.
            batch_imagenet (torch.Tensor): ImageNet batch. Defaults to None.
            normalize (bool): Normalize anomaly maps or not
            return_all_maps(bool): weather to return all three maps or only the combined anomaly map.

        Returns:
            Tensor: Predictions
        """
        image_size = batch.shape[-2:]
        with torch.no_grad():
            mean_std_set = self.is_set(self.teacher_mean) and self.is_set(self.teacher_std)
            quantiles_set = (
                self.is_set(self.qa_st)
                and self.is_set(self.qb_st)
                and self.is_set(self.qa_ae)
                and self.is_set(self.qb_ae)
            )

        with torch.no_grad():
            self.teacher.eval()
            teacher_output = self.teacher(batch)
            if mean_std_set:
                teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std

        student_output = self.student(batch)
        distance_st = torch.pow(teacher_output - student_output[:, : self.teacher_out_channels, :, :], 2)

        if self.training:
            # Student loss
            distance_st = reduce_tensor_elems(distance_st)
            d_hard = torch.quantile(distance_st, 0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
            if self.use_imgnet_penalty:
                student_output_penalty = self.student(batch_imagenet)[:, : self.teacher_out_channels, :, :]
                loss_penalty = torch.mean(student_output_penalty**2)
            else:
                loss_penalty = 0
            loss_st = loss_hard + loss_penalty

            # Autoencoder and Student AE Loss
            aug_img = batch
            ae_output_aug = self.autoencoder(aug_img, image_size)

            with torch.no_grad():
                teacher_output_aug = self.teacher(aug_img)
                if mean_std_set:
                    teacher_output_aug = (teacher_output_aug - self.teacher_mean) / self.teacher_std

            student_output_ae_aug = self.student(aug_img)[:, self.teacher_out_channels :, :, :]

            distance_ae = torch.pow(teacher_output_aug - ae_output_aug, 2)
            distance_stae = torch.pow(ae_output_aug - student_output_ae_aug, 2)

            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            return loss_st, loss_ae, loss_stae

        # Eval mode.
        with torch.no_grad():
            ae_output = self.autoencoder(batch, image_size)

            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            map_stae = torch.mean(
                (ae_output - student_output[:, self.teacher_out_channels :]) ** 2,
                dim=1,
                keepdim=True,
            )

        if self.pad_maps:
            map_st = F.pad(map_st, (4, 4, 4, 4))
            map_stae = F.pad(map_stae, (4, 4, 4, 4))
        map_st = F.interpolate(map_st, size=image_size, mode="bilinear")
        map_stae = F.interpolate(map_stae, size=image_size, mode="bilinear")

        if quantiles_set and normalize:
            denom_st = (self.qb_st - self.qa_st).abs().clamp_min(1e-12)
            denom_ae = (self.qb_ae - self.qa_ae).abs().clamp_min(1e-12)

            map_st = 0.1 * (map_st - self.qa_st) / denom_st
            map_stae = 0.1 * (map_stae - self.qa_ae) / denom_ae

        map_combined = map_st  # 0.5 * map_st + 0.5 * map_stae
        if return_all_maps:
            return {"anomaly_map": map_combined, "map_st": map_st, "map_ae": map_stae}
        else:
            return map_combined
