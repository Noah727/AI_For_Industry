from __future__ import annotations

import torch
from torch import nn


class TinyKeypointUNet(nn.Module):
    """Small fully convolutional model for port/plug heatmaps."""

    def __init__(self, out_channels: int = 2, output_stride: int = 4):
        super().__init__()
        if output_stride not in (1, 4):
            raise ValueError("TinyKeypointUNet supports output_stride 1 or 4")
        self.output_stride = output_stride
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.low_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )
        self.up2 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.up1 = nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1)
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(image)
        skip2 = self.enc2(self.down1(skip1))
        hidden = self.bottleneck(self.down2(skip2))
        if self.output_stride == 4:
            return self.low_head(hidden)

        hidden = nn.functional.interpolate(
            hidden,
            size=skip2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hidden = self.act(self.up2(torch.cat([hidden, skip2], dim=1)))
        hidden = nn.functional.interpolate(
            hidden,
            size=skip1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hidden = self.act(self.up1(torch.cat([hidden, skip1], dim=1)))
        return self.head(hidden)
