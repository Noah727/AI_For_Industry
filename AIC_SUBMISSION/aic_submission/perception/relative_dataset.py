from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from aic_submission.perception.dataset import SampleRef


DEFAULT_CAMERA_KEYS = ("left_image", "center_image", "right_image")


def task_id_from_vector(task_vector: tuple[float, float, float, float]) -> int:
    """Map the currently used task types to model heads."""

    return 0 if task_vector[0] > task_vector[1] else 1


def task_name_from_id(task_id: int) -> str:
    return "sfp->sfp" if task_id == 0 else "sc->sc"


def _relative_target(sample_path: Path) -> np.ndarray:
    data = np.load(sample_path)
    port_xyz = data["port_pose_base"][:3].astype(np.float64)
    tcp_xyz = data["tcp_pose_base"][:3].astype(np.float64)
    return port_xyz - tcp_xyz


def compute_relative_target_stats(samples: list[SampleRef]) -> dict[str, object]:
    """Compute train-set normalization stats for port_xyz - tcp_xyz."""

    if not samples:
        raise ValueError("Cannot compute target stats from an empty sample list")

    count = 0
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    for sample in samples:
        target = _relative_target(sample.path)
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


class RelativePortPoseDataset(Dataset):
    """Teacher dataset for normalized relative port-pose regression."""

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
        tcp_xyz = data["tcp_pose_base"][:3].astype(np.float32)
        relative_xyz = port_xyz - tcp_xyz
        target = (relative_xyz - self.target_mean) / self.target_std

        item: dict[str, torch.Tensor | str] = {
            "images": torch.from_numpy(images),
            "state": torch.from_numpy(state),
            "task": torch.from_numpy(task),
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "target": torch.from_numpy(target.astype(np.float32)),
            "target_relative_xyz": torch.from_numpy(relative_xyz),
            "port_xyz": torch.from_numpy(port_xyz),
            "tcp_xyz": torch.from_numpy(tcp_xyz),
        }
        if self.include_metadata:
            item["stage"] = str(data["stage"])
            item["sample_path"] = str(sample_ref.path)
            item["episode_name"] = sample_ref.episode_name
        return item
