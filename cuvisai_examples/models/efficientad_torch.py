from cuvisai_examples.registry import MODELS
import torch
from torch import nn
from torch.nn import functional as F
import math


class SmallPatchDescriptionNetwork(nn.Module):
    def __init__(
        self, out_channels: int, padding: bool = False, in_channels: int = 6
    ) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(
            in_channels, 128, kernel_size=4, stride=1, padding=3 * pad_mult
        )
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(
            256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        return self.conv4(x)


class MediumPatchDescriptionNetwork(nn.Module):
    def __init__(
        self, out_channels: int, padding: bool = False, in_channels: int = 6
    ) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(
            in_channels, 256, kernel_size=4, stride=1, padding=3 * pad_mult
        )
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(
            512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult
        )
        self.conv6 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0 * pad_mult
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
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
    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        return self.enconv6(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int, padding: int) -> None:
        super().__init__()
        self.padding = padding
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

    def forward(
        self, x: torch.Tensor, image_size: tuple[int, int] | torch.Size
    ) -> torch.Tensor:
        last_upsample = (
            math.ceil(image_size[0] / 4)
            if self.padding
            else math.ceil(image_size[0] / 4) - 8,
            math.ceil(image_size[1] / 4)
            if self.padding
            else math.ceil(image_size[1] / 4) - 8,
        )
        x = F.interpolate(
            x, size=(image_size[0] // 64 - 1, image_size[1] // 64 - 1), mode="bilinear"
        )
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(
            x, size=(image_size[0] // 32, image_size[1] // 32), mode="bilinear"
        )
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(
            x, size=(image_size[0] // 16 - 1, image_size[1] // 16 - 1), mode="bilinear"
        )
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(
            x, size=(image_size[0] // 8, image_size[1] // 8), mode="bilinear"
        )
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(
            x, size=(image_size[0] // 4 - 1, image_size[1] // 4 - 1), mode="bilinear"
        )
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(
            x, size=(image_size[0] // 2 - 1, image_size[1] // 2 - 1), mode="bilinear"
        )
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        return self.deconv8(x)


class AutoEncoder(nn.Module):
    def __init__(self, out_channels: int, padding: int, in_channels: int = 6) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels, padding)

    def forward(
        self, x: torch.Tensor, image_size: tuple[int, int] | torch.Size
    ) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.encoder(x)
        return self.decoder(x, image_size)


@MODELS.register("efficientad.MediumBackbones")
class MediumBackbones(nn.Module):
    def __init__(
        self,
        teacher_out_channels: int = 384,
        padding: bool = False,
        in_channels: int = 6,
    ):
        super().__init__()
        self.teacher = MediumPatchDescriptionNetwork(
            out_channels=teacher_out_channels, padding=padding, in_channels=in_channels
        )
        self.student = MediumPatchDescriptionNetwork(
            out_channels=teacher_out_channels * 2,
            padding=padding,
            in_channels=in_channels,
        )
        self.ae = AutoEncoder(
            out_channels=teacher_out_channels, padding=padding, in_channels=in_channels
        )


@MODELS.register("efficientad.SmallBackbones")
class SmallBackbones(nn.Module):
    def __init__(
        self,
        teacher_out_channels: int = 384,
        padding: bool = False,
        in_channels: int = 6,
    ):
        super().__init__()
        self.teacher = SmallPatchDescriptionNetwork(
            out_channels=teacher_out_channels, padding=padding, in_channels=in_channels
        )
        self.student = SmallPatchDescriptionNetwork(
            out_channels=teacher_out_channels * 2,
            padding=padding,
            in_channels=in_channels,
        )
        self.ae = AutoEncoder(
            out_channels=teacher_out_channels, padding=padding, in_channels=in_channels
        )
