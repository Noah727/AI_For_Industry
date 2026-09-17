from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRef:
    path: Path
    task_vector: tuple[float, ...]
    episode_name: str


def _load_metadata(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _task_vector_from_fields(task: dict, mode: str = "basic") -> tuple[float, ...]:
    mode = mode.strip().lower()
    plug_type = str(task.get("plug_type", ""))
    port_type = str(task.get("port_type", ""))
    target_module = str(task.get("target_module_name", ""))
    port_name = str(task.get("port_name", ""))

    basic = [
        1.0 if plug_type == "sfp" else 0.0,
        1.0 if plug_type == "sc" else 0.0,
        1.0 if port_type == "sfp" else 0.0,
        1.0 if port_type == "sc" else 0.0,
    ]
    if mode == "basic":
        return tuple(basic)
    if mode != "rich":
        raise ValueError(f"Unsupported task vector mode: {mode}")

    def one_hot_index(pattern: str, size: int) -> list[float]:
        match = re.search(pattern, target_module)
        values = [0.0] * size
        if match is not None:
            index = int(match.group(1))
            if 0 <= index < size:
                values[index] = 1.0
        return values

    nic_rail = one_hot_index(r"nic_card_mount_(\d+)$", 5)
    sc_rail = one_hot_index(r"sc_port_(\d+)$", 2)
    sfp_port = [
        1.0 if port_name == "sfp_port_0" else 0.0,
        1.0 if port_name == "sfp_port_1" else 0.0,
    ]
    sc_port_base = [1.0 if port_name == "sc_port_base" else 0.0]
    return tuple(basic + nic_rail + sc_rail + sfp_port + sc_port_base)


def _task_vector(metadata: dict, mode: str = "basic") -> tuple[float, ...]:
    task = metadata["task"]
    return _task_vector_from_fields(task, mode=mode)


def task_vector_from_task(task, mode: str = "basic") -> tuple[float, ...]:
    """Build the model task vector from a runtime Task-like object."""

    return _task_vector_from_fields(
        {
            "plug_type": getattr(task, "plug_type", ""),
            "port_type": getattr(task, "port_type", ""),
            "target_module_name": getattr(task, "target_module_name", ""),
            "port_name": getattr(task, "port_name", ""),
        },
        mode=mode,
    )


def task_vector_dim(mode: str = "basic") -> int:
    empty_task = {
        "plug_type": "",
        "port_type": "",
        "target_module_name": "",
        "port_name": "",
    }
    return len(_task_vector_from_fields(empty_task, mode=mode))


def discover_teacher_samples(
    root: Path, task_vector_mode: str = "basic"
) -> list[SampleRef]:
    samples: list[SampleRef] = []
    for episode_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metadata_path = episode_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        task_vector = _task_vector(_load_metadata(metadata_path), mode=task_vector_mode)
        for sample_path in sorted(episode_dir.glob("sample_*.npz")):
            samples.append(
                SampleRef(
                    path=sample_path,
                    task_vector=task_vector,
                    episode_name=episode_dir.name,
                )
            )
    return samples


def split_by_episode(
    samples: list[SampleRef], validation_fraction: float = 0.25
) -> tuple[list[SampleRef], list[SampleRef]]:
    episode_tasks: dict[str, tuple[float, float, float, float]] = {}
    for sample in samples:
        episode_tasks.setdefault(sample.episode_name, sample.task_vector)

    if len(episode_tasks) <= 1:
        return samples, samples

    episodes_by_task: dict[tuple[float, float, float, float], list[str]] = {}
    for episode_name, task_vector in episode_tasks.items():
        episodes_by_task.setdefault(task_vector, []).append(episode_name)

    val_episodes: set[str] = set()
    for episodes in episodes_by_task.values():
        episodes = sorted(episodes)
        if len(episodes) <= 1:
            continue
        val_count = max(1, round(len(episodes) * validation_fraction))
        val_episodes.update(episodes[-val_count:])

    if not val_episodes:
        val_count = max(1, round(len(episode_tasks) * validation_fraction))
        val_episodes = set(sorted(episode_tasks)[-val_count:])

    train_samples = [
        sample for sample in samples if sample.episode_name not in val_episodes
    ]
    val_samples = [sample for sample in samples if sample.episode_name in val_episodes]
    return train_samples, val_samples


class TeacherPortPoseDataset(Dataset):
    """Load TeacherRecorderPolicy samples for port-position regression."""

    def __init__(
        self,
        samples: list[SampleRef],
        image_key: str = "center_image",
        include_metadata: bool = False,
    ):
        self.samples = samples
        self.image_key = image_key
        self.include_metadata = include_metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample_ref = self.samples[index]
        data = np.load(sample_ref.path)

        image = data[self.image_key].astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = (image - 0.5) / 0.5

        state = data["state"].astype(np.float32)
        task = np.asarray(sample_ref.task_vector, dtype=np.float32)
        target_xyz = data["port_pose_base"][:3].astype(np.float32)

        item = {
            "image": torch.from_numpy(image),
            "state": torch.from_numpy(state),
            "task": torch.from_numpy(task),
            "target_xyz": torch.from_numpy(target_xyz),
        }
        if self.include_metadata:
            item["stage"] = str(data["stage"])
            item["sample_path"] = str(sample_ref.path)
            item["episode_name"] = sample_ref.episode_name
        return item
