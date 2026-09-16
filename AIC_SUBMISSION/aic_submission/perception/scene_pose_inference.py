from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from aic_submission.perception.plug_aware_inference import (
    PlugAwarePoseInference,
    PlugAwarePrediction,
)
from aic_submission.perception.relative_dataset import (
    DEFAULT_CAMERA_KEYS,
    task_id_from_vector,
)
from aic_submission.perception.relative_model import MultiCameraRelativePortPoseRegressor


class ScenePoseInference(PlugAwarePoseInference):
    """Predict absolute port pose and current plug pose from camera observations."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
        image_stride: int = 4,
    ):
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = checkpoint.get("config", {})
        self.camera_keys = tuple(self.config.get("camera_keys", DEFAULT_CAMERA_KEYS))
        self.image_stride = max(1, image_stride)
        self.zero_state = bool(self.config.get("zero_state", True))
        self.target_mean = torch.tensor(
            checkpoint["target_stats"]["mean"], dtype=torch.float32, device=self.device
        )
        self.target_std = torch.tensor(
            checkpoint["target_stats"]["std"], dtype=torch.float32, device=self.device
        )

        self.model = MultiCameraRelativePortPoseRegressor(
            num_cameras=len(self.camera_keys),
            output_dim=14,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.checkpoint_path = checkpoint_path
        self.checkpoint_epoch = checkpoint.get("epoch")
        self.checkpoint_val_loss = checkpoint.get("val_loss")

    def predict(self, obs, task) -> PlugAwarePrediction:
        image_by_key = {
            "left_image": obs.left_image,
            "center_image": obs.center_image,
            "right_image": obs.right_image,
        }
        images = torch.stack(
            [
                self.image_array_to_tensor(self.image_msg_to_array(image_by_key[key]))
                for key in self.camera_keys
            ],
            dim=0,
        ).unsqueeze(0)

        state_np = self.state_from_observation(obs)
        if self.zero_state:
            state_np = np.zeros_like(state_np)
        state = torch.from_numpy(state_np).unsqueeze(0)
        task_vector = self.task_vector(task)
        task_tensor = torch.tensor([task_vector], dtype=torch.float32)
        task_id = torch.tensor([task_id_from_vector(task_vector)], dtype=torch.long)

        with torch.no_grad():
            pred_norm = self.model(
                images.to(self.device, non_blocking=True),
                state.to(self.device, non_blocking=True),
                task_tensor.to(self.device, non_blocking=True),
                task_id.to(self.device, non_blocking=True),
            )
            pred = pred_norm * self.target_std + self.target_mean

        pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
        port_xyz = pred_np[:3]
        plug_relative = pred_np[3:6]
        port_quat = self.normalize_quat_xyzw(pred_np[6:10])
        plug_quat = self.normalize_quat_xyzw(pred_np[10:14])
        tcp_xyz = self.tcp_xyz_from_observation(obs)
        return PlugAwarePrediction(
            port_relative_xyz=port_xyz - tcp_xyz,
            plug_relative_xyz=plug_relative,
            port_xyz=port_xyz,
            plug_xyz=tcp_xyz + plug_relative,
            port_quat_xyzw=port_quat,
            plug_quat_xyzw=plug_quat,
        )
