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


def _canonical_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float64, copy=True)
    norm = np.linalg.norm(quat)
    if norm > 1.0e-8:
        quat /= norm
    if quat[3] < 0.0:
        quat *= -1.0
    return quat


def _plug_aware_target(sample_path: Path) -> np.ndarray:
    data = np.load(sample_path)
    tcp_xyz = data["tcp_pose_base"][:3].astype(np.float64)
    port_relative = data["port_pose_base"][:3].astype(np.float64) - tcp_xyz
    plug_relative = data["plug_pose_base"][:3].astype(np.float64) - tcp_xyz
    port_quat = _canonical_quat_xyzw(data["port_pose_base"][3:7])
    plug_quat = _canonical_quat_xyzw(data["plug_pose_base"][3:7])
    return np.concatenate(
        [port_relative, plug_relative, port_quat, plug_quat],
        axis=0,
    )


def compute_plug_aware_target_stats(samples: list[SampleRef]) -> dict[str, object]:
    """Compute train-set normalization stats for port/tip relative targets."""

    if not samples:
        raise ValueError("Cannot compute target stats from an empty sample list")

    count = 0
    total = np.zeros(14, dtype=np.float64)
    total_sq = np.zeros(14, dtype=np.float64)
    for sample in samples:
        target = _plug_aware_target(sample.path)
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
        "layout": "port_relative_xyz plug_relative_xyz port_quat_xyzw plug_quat_xyzw",
    }


class PlugAwarePoseDataset(Dataset):
    """Teacher dataset for joint port and plug-position regression."""

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

        port_xyz = data["port_pose_base"][:3].astype(np.float32)
        plug_xyz = data["plug_pose_base"][:3].astype(np.float32)
        tcp_xyz = data["tcp_pose_base"][:3].astype(np.float32)
        port_relative = port_xyz - tcp_xyz
        plug_relative = plug_xyz - tcp_xyz
        port_quat = _canonical_quat_xyzw(data["port_pose_base"][3:7]).astype(np.float32)
        plug_quat = _canonical_quat_xyzw(data["plug_pose_base"][3:7]).astype(np.float32)
        target_raw = np.concatenate(
            [port_relative, plug_relative, port_quat, plug_quat],
            axis=0,
        )
        target = (target_raw - self.target_mean) / self.target_std

        item: dict[str, torch.Tensor | str] = {
            "images": torch.from_numpy(images),
            "state": torch.from_numpy(state),
            "task": torch.from_numpy(task),
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "target": torch.from_numpy(target.astype(np.float32)),
            "target_port_relative_xyz": torch.from_numpy(port_relative),
            "target_plug_relative_xyz": torch.from_numpy(plug_relative),
            "target_port_quat_xyzw": torch.from_numpy(port_quat),
            "target_plug_quat_xyzw": torch.from_numpy(plug_quat),
            "port_xyz": torch.from_numpy(port_xyz),
            "plug_xyz": torch.from_numpy(plug_xyz),
            "tcp_xyz": torch.from_numpy(tcp_xyz),
        }
        if self.include_metadata:
            item["stage"] = str(data["stage"])
            item["sample_path"] = str(sample_ref.path)
            item["episode_name"] = sample_ref.episode_name
        return item
