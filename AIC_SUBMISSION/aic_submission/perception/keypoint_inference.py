from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aic_submission.perception.keypoint_dataset import (
    CAMERA_FULL_HEIGHT,
    CAMERA_FULL_WIDTH,
    CAMERA_HORIZONTAL_FOV,
    CAMERA_PREFIX_BY_IMAGE,
    TCP_TO_CAMERA_OPTICAL,
    _matrix_to_pose_array,
    _pose_array_to_matrix,
    _quat_xyzw_to_matrix,
)
from aic_submission.perception.keypoint_model import TinyKeypointUNet


@dataclass(frozen=True)
class KeypointPrediction:
    image_key: str
    image_stride: int
    heatmap_stride: int
    port_uv_input: np.ndarray
    plug_uv_input: np.ndarray
    port_uv_full: np.ndarray
    plug_uv_full: np.ndarray
    port_confidence: float
    plug_confidence: float

    @property
    def delta_uv_full(self) -> np.ndarray:
        return self.port_uv_full - self.plug_uv_full

    @property
    def min_confidence(self) -> float:
        return min(self.port_confidence, self.plug_confidence)


class KeypointHeatmapInference:
    """Runtime wrapper for the port/plug heatmap model."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
        image_key: str = "center_image",
        image_stride: int = 4,
    ):
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if image_key not in CAMERA_PREFIX_BY_IMAGE:
            raise ValueError(f"Unsupported image key: {image_key}")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.image_key = checkpoint.get("image_key", image_key)
        if self.image_key != image_key:
            self.image_key = image_key
        self.image_stride = max(1, int(image_stride))
        self.heatmap_stride = int(checkpoint.get("heatmap_stride", 4))
        self.model = TinyKeypointUNet(
            out_channels=2,
            output_stride=self.heatmap_stride,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.checkpoint_path = checkpoint_path
        self.checkpoint_epoch = checkpoint.get("epoch")
        self.checkpoint_val_pixel_error = checkpoint.get("val_pixel_error")

    @staticmethod
    def image_array_to_tensor(image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = (image - 0.5) / 0.5
        return torch.from_numpy(image)

    def image_msg_to_array(self, image_msg) -> tuple[np.ndarray, int]:
        data = np.frombuffer(image_msg.data, dtype=np.uint8)
        pixels = image_msg.height * image_msg.width
        if pixels <= 0:
            return np.zeros((0, 0, 3), dtype=np.uint8), 1

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
        return image[::stride, ::stride].copy(), stride

    @staticmethod
    def _decode_heatmap(heatmap: torch.Tensor) -> tuple[np.ndarray, float]:
        flat_index = int(torch.argmax(heatmap).item())
        width = heatmap.shape[-1]
        y = flat_index // width
        x = flat_index % width
        confidence = float(torch.max(heatmap).item())
        return np.asarray([x, y], dtype=np.float32), confidence

    def predict(self, obs, task=None) -> KeypointPrediction:
        image_by_key = {
            "left_image": obs.left_image,
            "center_image": obs.center_image,
            "right_image": obs.right_image,
        }
        image, applied_stride = self.image_msg_to_array(image_by_key[self.image_key])
        image_tensor = self.image_array_to_tensor(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(image_tensor.to(self.device, non_blocking=True))
            heatmaps = torch.sigmoid(logits).squeeze(0).detach().cpu()

        port_hm_uv, port_conf = self._decode_heatmap(heatmaps[0])
        plug_hm_uv, plug_conf = self._decode_heatmap(heatmaps[1])
        port_uv_input = (port_hm_uv + 0.5) * float(self.heatmap_stride)
        plug_uv_input = (plug_hm_uv + 0.5) * float(self.heatmap_stride)
        port_uv_full = port_uv_input * float(applied_stride)
        plug_uv_full = plug_uv_input * float(applied_stride)
        return KeypointPrediction(
            image_key=self.image_key,
            image_stride=applied_stride,
            heatmap_stride=self.heatmap_stride,
            port_uv_input=port_uv_input.astype(np.float32),
            plug_uv_input=plug_uv_input.astype(np.float32),
            port_uv_full=port_uv_full.astype(np.float32),
            plug_uv_full=plug_uv_full.astype(np.float32),
            port_confidence=port_conf,
            plug_confidence=plug_conf,
        )

    @staticmethod
    def camera_k_from_info(camera_info) -> np.ndarray:
        k = np.asarray(camera_info.k, dtype=np.float32)
        if k.size == 9 and abs(float(k[0])) > 1.0e-5:
            return k.reshape(3, 3)
        focal = CAMERA_FULL_WIDTH / (2.0 * np.tan(CAMERA_HORIZONTAL_FOV / 2.0))
        return np.asarray(
            [
                [focal, 0.0, CAMERA_FULL_WIDTH / 2.0],
                [0.0, focal, CAMERA_FULL_HEIGHT / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def tcp_pose_array_from_observation(obs) -> np.ndarray:
        pose = obs.controller_state.tcp_pose
        return np.asarray(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float32,
        )

    @classmethod
    def camera_pose_base_from_observation(
        cls,
        obs,
        image_key: str,
    ) -> np.ndarray:
        prefix = CAMERA_PREFIX_BY_IMAGE[image_key]
        base_to_tcp = _pose_array_to_matrix(cls.tcp_pose_array_from_observation(obs))
        base_to_camera = base_to_tcp @ TCP_TO_CAMERA_OPTICAL[prefix]
        return _matrix_to_pose_array(base_to_camera)

    @staticmethod
    def point_base_to_camera(
        point_base: np.ndarray, camera_pose_base: np.ndarray
    ) -> np.ndarray:
        rotation = _quat_xyzw_to_matrix(camera_pose_base[3:7])
        translation = camera_pose_base[:3].astype(np.float32)
        return rotation.T @ (point_base[:3].astype(np.float32) - translation)

    @classmethod
    def pixel_delta_to_base_vector(
        cls,
        obs,
        image_key: str,
        delta_uv_full: np.ndarray,
        reference_point_base: np.ndarray,
    ) -> np.ndarray | None:
        camera_info = {
            "left_image": obs.left_camera_info,
            "center_image": obs.center_camera_info,
            "right_image": obs.right_camera_info,
        }[image_key]
        camera_k = cls.camera_k_from_info(camera_info)
        camera_pose_base = cls.camera_pose_base_from_observation(obs, image_key)
        reference_cam = cls.point_base_to_camera(reference_point_base, camera_pose_base)
        depth = float(reference_cam[2])
        fx, fy = float(camera_k[0, 0]), float(camera_k[1, 1])
        if depth <= 1.0e-4 or abs(fx) < 1.0e-5 or abs(fy) < 1.0e-5:
            return None

        delta_cam = np.asarray(
            [
                float(delta_uv_full[0]) * depth / fx,
                float(delta_uv_full[1]) * depth / fy,
                0.0,
            ],
            dtype=np.float32,
        )
        rotation = _quat_xyzw_to_matrix(camera_pose_base[3:7])
        return (rotation @ delta_cam).astype(np.float32)
