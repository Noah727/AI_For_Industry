from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class MultiCameraRelativePortPoseRegressor(nn.Module):
    """Shared ResNet-18 over multiple cameras with task-specific XYZ heads."""

    def __init__(
        self,
        state_dim: int = 26,
        task_dim: int = 4,
        num_cameras: int = 3,
        num_tasks: int = 2,
        output_dim: int = 3,
    ):
        super().__init__()
        backbone = resnet18(weights=None)
        image_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.num_cameras = num_cameras

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim + task_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        fused_dim = image_dim * num_cameras + 128
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, output_dim),
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        task: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_cameras, channels, height, width = images.shape
        if num_cameras != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} cameras, got {num_cameras}")

        visual = self.backbone(
            images.reshape(batch_size * num_cameras, channels, height, width)
        )
        visual = visual.reshape(batch_size, num_cameras, -1).flatten(start_dim=1)
        proprio = self.state_encoder(torch.cat([state, task], dim=-1))
        features = self.trunk(torch.cat([visual, proprio], dim=-1))

        outputs = torch.stack([head(features) for head in self.heads], dim=1)
        batch_indices = torch.arange(batch_size, device=images.device)
        return outputs[batch_indices, task_id]
