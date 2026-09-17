from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from aic_submission.perception.relative_dataset import (
    DEFAULT_CAMERA_KEYS,
    task_id_from_vector,
)
from aic_submission.perception.dataset import task_vector_from_task
from aic_submission.perception.relative_model import (
    MultiCameraRelativePortPoseRegressor,
)


class RelativePortPoseInference:
    """Runtime wrapper for the relative multi-camera port-pose model."""

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
        self.task_vector_mode = str(self.config.get("task_vector_mode", "basic"))
        self.task_dim = int(self.config.get("task_dim", 4))
        self.image_stride = max(1, image_stride)
        self.target_mean = torch.tensor(
            checkpoint["target_stats"]["mean"], dtype=torch.float32, device=self.device
        )
        self.target_std = torch.tensor(
            checkpoint["target_stats"]["std"], dtype=torch.float32, device=self.device
        )

        self.model = MultiCameraRelativePortPoseRegressor(
            num_cameras=len(self.camera_keys),
            task_dim=self.task_dim,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.checkpoint_path = checkpoint_path
        self.checkpoint_epoch = checkpoint.get("epoch")
        self.checkpoint_val_loss = checkpoint.get("val_loss")

    def image_msg_to_array(self, image_msg) -> np.ndarray:
        data = np.frombuffer(image_msg.data, dtype=np.uint8)
        pixels = image_msg.height * image_msg.width
        if pixels <= 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        channels = max(1, len(data) // pixels)
        expected = image_msg.height * image_msg.width * channels
        image = data[:expected].reshape(image_msg.height, image_msg.width, channels)
        if channels == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels > 3:
            image = image[:, :, :3]
        stride = (
            self.image_stride if image.shape[0] > 300 or image.shape[1] > 400 else 1
        )
        return image[::stride, ::stride].copy()

    @staticmethod
    def image_array_to_tensor(image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = (image - 0.5) / 0.5
        return torch.from_numpy(image)

    @staticmethod
    def state_from_observation(obs) -> np.ndarray:
        tcp_pose = obs.controller_state.tcp_pose
        tcp_vel = obs.controller_state.tcp_velocity
        return np.array(
            [
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                *obs.controller_state.tcp_error,
                *obs.joint_states.position[:7],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def task_vector(task) -> tuple[float, float, float, float]:
        return task_vector_from_task(task, mode="basic")

    def _task_vector_for_checkpoint(self, task) -> tuple[float, ...]:
        return task_vector_from_task(task, mode=self.task_vector_mode)

    @staticmethod
    def tcp_xyz_from_observation(obs) -> np.ndarray:
        tcp_pose = obs.controller_state.tcp_pose
        return np.array(
            [tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z],
            dtype=np.float32,
        )

    def predict_relative_xyz(self, obs, task) -> np.ndarray:
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

        state = torch.from_numpy(self.state_from_observation(obs)).unsqueeze(0)
        task_vector = self._task_vector_for_checkpoint(task)
        task_tensor = torch.tensor([task_vector], dtype=torch.float32)
        task_id = torch.tensor([task_id_from_vector(task_vector)], dtype=torch.long)

        with torch.no_grad():
            pred_norm = self.model(
                images.to(self.device, non_blocking=True),
                state.to(self.device, non_blocking=True),
                task_tensor.to(self.device, non_blocking=True),
                task_id.to(self.device, non_blocking=True),
            )
            pred_relative = pred_norm * self.target_std + self.target_mean

        return pred_relative.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def predict_port_xyz(self, obs, task) -> np.ndarray:
        return self.tcp_xyz_from_observation(obs) + self.predict_relative_xyz(obs, task)
