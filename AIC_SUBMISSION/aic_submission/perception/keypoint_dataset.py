from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from aic_submission.perception.dataset import (
    SampleRef,
    discover_teacher_samples,
    split_by_episode,
)


CAMERA_PREFIX_BY_IMAGE = {
    "left_image": "left",
    "center_image": "center",
    "right_image": "right",
}

CAMERA_FULL_WIDTH = 1152
CAMERA_FULL_HEIGHT = 1024
CAMERA_HORIZONTAL_FOV = 0.8718
LEGACY_IMAGE_STRIDE = 4.0


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return (rz @ ry @ rx).astype(np.float32)


def _matrix_from_xyz_rpy(
    xyz: tuple[float, float, float], rpy: tuple[float, float, float]
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = _rpy_to_matrix(*rpy)
    matrix[:3, 3] = np.asarray(xyz, dtype=np.float32)
    return matrix


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = matrix[:3, :3].astype(np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=np.float32)
    quat /= max(float(np.linalg.norm(quat)), 1.0e-8)
    return quat


def _pose_array_to_matrix(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = _quat_xyzw_to_matrix(pose[3:7])
    matrix[:3, 3] = pose[:3].astype(np.float32)
    return matrix


def _matrix_to_pose_array(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [matrix[:3, 3].astype(np.float32), _matrix_to_quat_xyzw(matrix)]
    ).astype(np.float32)


def _tool0_to_camera_optical(camera_prefix: str) -> np.ndarray:
    tool0_to_cam_mount = _matrix_from_xyz_rpy((0.0, 0.0, -0.0265), (0.0, 0.0, 0.0))
    camera_mount_to_link = {
        "center": _matrix_from_xyz_rpy(
            (0.0, -0.1077, -0.00719),
            (0.0, -1.30899630, 1.57079623),
        ),
        "left": _matrix_from_xyz_rpy(
            (-0.09326, -0.053843, -0.007188),
            (0.0, -1.30899630, 0.523599027),
        ),
        "right": _matrix_from_xyz_rpy(
            (0.09326, -0.053843, -0.007188),
            (0.0, -1.30899630, 2.61799343),
        ),
    }[camera_prefix]
    link_to_sensor = _matrix_from_xyz_rpy((0.02174, 0.0, 0.0145), (0.0, 0.0, 0.0))
    sensor_to_optical = _matrix_from_xyz_rpy(
        (0.0, 0.0, 0.0),
        (-1.5708, 0.0, -1.5708),
    )
    return (
        tool0_to_cam_mount @ camera_mount_to_link @ link_to_sensor @ sensor_to_optical
    )


def _tcp_to_camera_optical(camera_prefix: str) -> np.ndarray:
    tool0_to_tcp = _matrix_from_xyz_rpy((0.0, 0.0, 0.1965), (0.0, 0.0, 0.0))
    return np.linalg.inv(tool0_to_tcp) @ _tool0_to_camera_optical(camera_prefix)


TCP_TO_CAMERA_OPTICAL = {
    prefix: _tcp_to_camera_optical(prefix) for prefix in ("left", "center", "right")
}


@dataclass(frozen=True)
class KeypointSampleRef:
    sample: SampleRef
    image_key: str
    camera_prefix: str


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat.astype(np.float64)
    norm = np.linalg.norm([x, y, z, w])
    if norm < 1.0e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def project_point_base_to_image(
    point_base: np.ndarray,
    camera_pose_base: np.ndarray,
    camera_k: np.ndarray,
    image_shape: tuple[int, int],
    image_stride: float,
) -> tuple[np.ndarray, bool]:
    """Project a base_link XYZ point into a downsampled camera image."""

    height, width = image_shape
    if not np.all(np.isfinite(camera_pose_base)):
        return np.zeros(2, dtype=np.float32), False

    camera_t = camera_pose_base[:3].astype(np.float32)
    camera_r = _quat_xyzw_to_matrix(camera_pose_base[3:7])
    point_cam = camera_r.T @ (point_base[:3].astype(np.float32) - camera_t)
    if point_cam[2] <= 1.0e-5:
        return np.zeros(2, dtype=np.float32), False

    fx, fy = float(camera_k[0, 0]), float(camera_k[1, 1])
    cx, cy = float(camera_k[0, 2]), float(camera_k[1, 2])
    if abs(fx) < 1.0e-5 or abs(fy) < 1.0e-5:
        return np.zeros(2, dtype=np.float32), False

    u = (fx * float(point_cam[0]) / float(point_cam[2]) + cx) / image_stride
    v = (fy * float(point_cam[1]) / float(point_cam[2]) + cy) / image_stride
    valid = 0.0 <= u < width and 0.0 <= v < height
    return np.asarray([u, v], dtype=np.float32), bool(valid)


def gaussian_heatmap(
    image_shape: tuple[int, int],
    center_uv: np.ndarray,
    sigma: float,
    valid: bool,
) -> np.ndarray:
    height, width = image_shape
    if not valid:
        return np.zeros((height, width), dtype=np.float32)

    u, v = float(center_uv[0]), float(center_uv[1])
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    heatmap = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2.0 * sigma * sigma))
    return heatmap.astype(np.float32)


def discover_keypoint_samples(
    root: Path,
    image_key: str = "center_image",
) -> list[KeypointSampleRef]:
    if image_key not in CAMERA_PREFIX_BY_IMAGE:
        raise ValueError(f"Unsupported image key: {image_key}")

    camera_prefix = CAMERA_PREFIX_BY_IMAGE[image_key]
    required_base = {image_key, "port_pose_base", "plug_pose_base"}
    required_camera = {f"{camera_prefix}_camera_k", f"{camera_prefix}_camera_pose_base"}

    refs: list[KeypointSampleRef] = []
    for sample in discover_teacher_samples(root):
        try:
            with np.load(sample.path) as data:
                keys = set(data.files)
                has_enriched_camera = required_camera.issubset(keys)
                has_legacy_camera = "tcp_pose_base" in keys
                if required_base.issubset(keys) and (
                    has_enriched_camera or has_legacy_camera
                ):
                    refs.append(
                        KeypointSampleRef(
                            sample=sample,
                            image_key=image_key,
                            camera_prefix=camera_prefix,
                        )
                    )
        except Exception:
            continue
    return refs


def split_keypoint_samples(
    samples: list[KeypointSampleRef],
    validation_fraction: float = 0.25,
) -> tuple[list[KeypointSampleRef], list[KeypointSampleRef]]:
    train_base, val_base = split_by_episode(
        [sample.sample for sample in samples],
        validation_fraction=validation_fraction,
    )
    train_paths = {sample.path for sample in train_base}
    val_paths = {sample.path for sample in val_base}
    return (
        [sample for sample in samples if sample.sample.path in train_paths],
        [sample for sample in samples if sample.sample.path in val_paths],
    )


class TeacherKeypointDataset(Dataset):
    """Project teacher TF labels into camera-space port/plug heatmaps."""

    def __init__(
        self,
        samples: list[KeypointSampleRef],
        sigma: float = 1.5,
        heatmap_stride: int = 4,
    ):
        self.samples = samples
        self.sigma = sigma
        self.heatmap_stride = max(1, int(heatmap_stride))

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _legacy_camera_k() -> np.ndarray:
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
    def _legacy_camera_pose_base(data, camera_prefix: str) -> np.ndarray:
        tcp_pose_base = data["tcp_pose_base"].astype(np.float32)
        base_to_tcp = _pose_array_to_matrix(tcp_pose_base)
        base_to_camera = base_to_tcp @ TCP_TO_CAMERA_OPTICAL[camera_prefix]
        return _matrix_to_pose_array(base_to_camera)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.samples[index]
        data = np.load(ref.sample.path)
        image = data[ref.image_key].astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = (image - 0.5) / 0.5

        height, width = image.shape[1:]
        prefix = ref.camera_prefix
        if f"{prefix}_camera_k" in data.files:
            camera_k = data[f"{prefix}_camera_k"].astype(np.float32)
            camera_pose_base = data[f"{prefix}_camera_pose_base"].astype(np.float32)
            if "image_stride" in data.files:
                image_stride = float(np.asarray(data["image_stride"]).item())
            else:
                camera_width = float(np.asarray(data[f"{prefix}_camera_width"]).item())
                image_stride = max(1.0, camera_width / max(1, width))
        else:
            camera_k = self._legacy_camera_k()
            camera_pose_base = self._legacy_camera_pose_base(data, prefix)
            image_stride = LEGACY_IMAGE_STRIDE

        coords = []
        coords_full = []
        valids = []
        heatmaps = []
        heatmap_shape = (
            max(1, height // self.heatmap_stride),
            max(1, width // self.heatmap_stride),
        )
        for target_key in ("port_pose_base", "plug_pose_base"):
            coord, valid = project_point_base_to_image(
                data[target_key][:3],
                camera_pose_base,
                camera_k,
                (height, width),
                image_stride,
            )
            heatmap_coord = coord / float(self.heatmap_stride)
            heatmap_valid = (
                valid
                and 0.0 <= heatmap_coord[0] < heatmap_shape[1]
                and 0.0 <= heatmap_coord[1] < heatmap_shape[0]
            )
            coords.append(heatmap_coord.astype(np.float32))
            coords_full.append(coord.astype(np.float32))
            valids.append(float(valid))
            heatmaps.append(
                gaussian_heatmap(
                    heatmap_shape,
                    heatmap_coord,
                    self.sigma,
                    heatmap_valid,
                )
            )

        return {
            "image": torch.from_numpy(image),
            "heatmap": torch.from_numpy(np.stack(heatmaps, axis=0)),
            "valid": torch.tensor(valids, dtype=torch.float32),
            "coords_uv": torch.from_numpy(np.stack(coords, axis=0)),
            "coords_uv_full": torch.from_numpy(np.stack(coords_full, axis=0)),
            "task": torch.tensor(ref.sample.task_vector, dtype=torch.float32),
        }
