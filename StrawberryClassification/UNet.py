import torch
import torch.nn as nn
import torch.nn.functional as F


class FreshTwin2DUNet(nn.Module):
    def __init__(self, in_channels, num_classes,pca = None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.pca = pca
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up8 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up9 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up10 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv10 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up11 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv11 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.output_2 = nn.Conv2d(32, num_classes, kernel_size=1)
        self.in_channels = in_channels

    def forward(self, x):
        # Encoder
        pca_image = x.squeeze(0).permute(1, 2, 0).reshape(-1, x.shape[1])
        pca_image = self.pca.transform(pca_image)
        pca_image = pca_image.reshape(1, x.shape[2], x.shape[3], self.in_channels).permute(0, 3, 1, 2).type(torch.float32)

        c1 = self.conv1(pca_image)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)

        u9 = self.up9(c4)
        u9 = torch.cat([u9, c3], dim=1)
        c9 = self.conv9(u9)

        u10 = self.up10(c9)
        u10 = torch.cat([u10, c2], dim=1)
        c10 = self.conv10(u10)

        u11 = self.up11(c10)
        u11 = torch.cat([u11, c1], dim=1)
        c11 = self.conv11(u11)


        b, ch, h, w = c11.shape
        c11 = c11.reshape(b* ch,  h, w)

        out = self.output_2(c11)
       
        return out