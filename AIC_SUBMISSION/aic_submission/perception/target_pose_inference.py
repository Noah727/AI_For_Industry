from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from geometry_msgs.msg import Point, Pose, Quaternion

from aic_submission.perception.dataset import task_vector_from_task
from aic_submission.perception.relative_dataset import (
    DEFAULT_CAMERA_KEYS,
    task_id_from_vector,
)
from aic_submission.perception.inference import RelativePortPoseInference
from aic_submission.perception.target_pose_dataset import canonicalize_quat_xyzw
from aic_submission.perception.target_pose_model import (
    MultiCameraTargetTcpPoseRegressor,
)


class TargetTcpPoseInference:
    """Runtime wrapper for teacher TCP target pose imitation."""

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

        self.model = MultiCameraTargetTcpPoseRegressor(
            num_cameras=len(self.camera_keys),
            task_dim=self.task_dim,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.checkpoint_path = checkpoint_path
        self.checkpoint_epoch = checkpoint.get("epoch")
        self.checkpoint_val_loss = checkpoint.get("val_loss")

    def _predict_target_vector(self, obs, task) -> np.ndarray:
        image_by_key = {
            "left_image": obs.left_image,
            "center_image": obs.center_image,
            "right_image": obs.right_image,
        }
        images = torch.stack(
            [
                RelativePortPoseInference.image_array_to_tensor(
                    RelativePortPoseInference.image_msg_to_array(self, image_by_key[key])
                )
                for key in self.camera_keys
            ],
            dim=0,
        ).unsqueeze(0)

        state = torch.from_numpy(
            RelativePortPoseInference.state_from_observation(obs)
        ).unsqueeze(0)
        task_vector = task_vector_from_task(task, mode=self.task_vector_mode)
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

        target = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
        target[3:7] = canonicalize_quat_xyzw(target[3:7])
        return target

    def predict_target_pose(self, obs, task) -> Pose:
        target = self._predict_target_vector(obs, task)
        tcp_xyz = RelativePortPoseInference.tcp_xyz_from_observation(obs)
        target_xyz = tcp_xyz + target[:3]
        quat = target[3:7]

        return Pose(
            position=Point(
                x=float(target_xyz[0]),
                y=float(target_xyz[1]),
                z=float(target_xyz[2]),
            ),
            orientation=Quaternion(
                x=float(quat[0]),
                y=float(quat[1]),
                z=float(quat[2]),
                w=float(quat[3]),
            ),
        )
