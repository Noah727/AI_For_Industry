from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from aic_submission.perception.dataset import SampleRef
from aic_submission.perception.relative_dataset import (
    DEFAULT_CAMERA_KEYS,
    task_id_from_vector,
)


def canonicalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    """Use a stable quaternion sign so component regression is well behaved."""

    quat = quat.astype(np.float32, copy=True)
    norm = np.linalg.norm(quat)
    if norm > 1.0e-8:
        quat /= norm
    if quat[3] < 0.0:
        quat *= -1.0
    return quat


def _target_vector(sample_path: Path) -> np.ndarray:
    data = np.load(sample_path)
    tcp_xyz = data["tcp_pose_base"][:3].astype(np.float32)
    target_pose = data["teacher_tcp_target_base"].astype(np.float32)
    target_delta_xyz = target_pose[:3] - tcp_xyz
    target_quat_xyzw = canonicalize_quat_xyzw(target_pose[3:7])
    return np.concatenate([target_delta_xyz, target_quat_xyzw]).astype(np.float32)


def compute_target_pose_stats(samples: list[SampleRef]) -> dict[str, object]:
    """Compute train-set normalization stats for teacher TCP target pose."""

    if not samples:
        raise ValueError("Cannot compute target pose stats from an empty sample list")

    count = 0
    total = np.zeros(7, dtype=np.float64)
    total_sq = np.zeros(7, dtype=np.float64)
    for sample in samples:
        target = _target_vector(sample.path).astype(np.float64)
        count += 1
        total += target
        total_sq += target * target

    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 1.0e-12)
    std = np.sqrt(variance)
    return {
        "count": count,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


class TargetTcpPoseDataset(Dataset):
    """Teacher dataset for next TCP target pose imitation."""

    def __init__(
        self,
        samples: list[SampleRef],
        target_mean: list[float] | tuple[float, ...] | np.ndarray,
        target_std: list[float] | tuple[float, ...] | np.ndarray,
        camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS,
        include_metadata: bool = False,
    ):
        self.samples = samples
        self.camera_keys = camera_keys
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        self.include_metadata = include_metadata

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
        image = data[key].astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return (image - 0.5) / 0.5

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_ref = self.samples[index]
        data = np.load(sample_ref.path)

        images = np.stack(
            [self._load_image(data, key) for key in self.camera_keys], axis=0
        )
        state = data["state"].astype(np.float32)
        task = np.asarray(sample_ref.task_vector, dtype=np.float32)
        task_id = task_id_from_vector(sample_ref.task_vector)

        tcp_xyz = data["tcp_pose_base"][:3].astype(np.float32)
        teacher_pose = data["teacher_tcp_target_base"].astype(np.float32)
        target_delta_xyz = teacher_pose[:3] - tcp_xyz
        target_quat_xyzw = canonicalize_quat_xyzw(teacher_pose[3:7])
        target_vector = np.concatenate([target_delta_xyz, target_quat_xyzw]).astype(
            np.float32
        )
        target = (target_vector - self.target_mean) / self.target_std

        item: dict[str, torch.Tensor | str] = {
            "images": torch.from_numpy(images),
            "state": torch.from_numpy(state),
            "task": torch.from_numpy(task),
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "target": torch.from_numpy(target.astype(np.float32)),
            "target_vector": torch.from_numpy(target_vector),
            "target_delta_xyz": torch.from_numpy(target_delta_xyz),
            "target_quat_xyzw": torch.from_numpy(target_quat_xyzw),
            "tcp_xyz": torch.from_numpy(tcp_xyz),
            "teacher_target_xyz": torch.from_numpy(teacher_pose[:3]),
        }
        if self.include_metadata:
            item["stage"] = str(data["stage"])
            item["sample_path"] = str(sample_ref.path)
            item["episode_name"] = sample_ref.episode_name
        return item
