from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class ResNetPortPoseRegressor(nn.Module):
    """Small ResNet-18 regressor for target port XYZ in base_link."""

    def __init__(self, state_dim: int = 26, task_dim: int = 4):
        super().__init__()
        backbone = resnet18(weights=None)
        image_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim + task_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(image_dim + 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        task: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.backbone(image)
        proprio = self.state_encoder(torch.cat([state, task], dim=-1))
        return self.head(torch.cat([visual, proprio], dim=-1))
