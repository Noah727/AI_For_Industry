#!/usr/bin/env python3
"""Inspect and sanity-check TeacherRecorderPolicy datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "left_image",
    "center_image",
    "right_image",
    "state",
    "wrench",
    "joint_names",
    "port_pose_base",
    "plug_pose_base",
    "tcp_pose_base",
    "teacher_tcp_target_base",
    "stage",
    "z_offset",
    "tick_index",
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_sample(path: Path) -> tuple[list[str], dict]:
    errors: list[str] = []
    info: dict = {}

    try:
        sample = np.load(path)
    except Exception as ex:
        return [f"cannot load npz: {ex}"], info

    keys = set(sample.files)
    missing = REQUIRED_KEYS - keys
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")

    for image_key in ["left_image", "center_image", "right_image"]:
        if image_key not in keys:
            continue
        image = sample[image_key]
        if image.ndim != 3 or image.shape[2] < 3:
            errors.append(f"{image_key} has bad shape {image.shape}")
        if image.dtype != np.uint8:
            errors.append(f"{image_key} has dtype {image.dtype}, expected uint8")
        if image.size and image.max() == image.min():
            errors.append(f"{image_key} is constant value {image.min()}")
        info[f"{image_key}_shape"] = tuple(image.shape)

    expected_vectors = {
        "state": 26,
        "wrench": 6,
        "port_pose_base": 7,
        "plug_pose_base": 7,
        "tcp_pose_base": 7,
        "teacher_tcp_target_base": 7,
    }
    for key, size in expected_vectors.items():
        if key not in keys:
            continue
        value = sample[key]
        if value.shape != (size,):
            errors.append(f"{key} has shape {value.shape}, expected ({size},)")
        if not np.isfinite(value).all():
            errors.append(f"{key} contains non-finite values")

    if "stage" in keys:
        info["stage"] = str(sample["stage"])
    if "z_offset" in keys:
        info["z_offset"] = float(sample["z_offset"])
    if "port_pose_base" in keys:
        info["port_xyz"] = sample["port_pose_base"][:3].astype(float)
    if "teacher_tcp_target_base" in keys:
        info["teacher_xyz"] = sample["teacher_tcp_target_base"][:3].astype(float)

    return errors, info


def inspect_dataset(root: Path, max_errors: int) -> int:
    episode_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not episode_dirs:
        print(f"No episode directories found under {root}")
        return 1

    total_samples = 0
    total_errors = 0
    task_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    image_shapes: Counter[tuple[int, ...]] = Counter()
    samples_per_episode: dict[str, int] = {}
    z_offsets: list[float] = []
    port_xyz_by_task: dict[str, list[np.ndarray]] = defaultdict(list)

    for episode_dir in episode_dirs:
        metadata_path = episode_dir / "metadata.json"
        if not metadata_path.exists():
            print(f"ERROR {episode_dir}: missing metadata.json")
            total_errors += 1
            continue

        metadata = load_json(metadata_path)
        task = metadata.get("task", {})
        task_name = f"{task.get('plug_type', '?')}->{task.get('port_type', '?')}"
        task_counts[task_name] += 1

        sample_paths = sorted(episode_dir.glob("sample_*.npz"))
        samples_per_episode[episode_dir.name] = len(sample_paths)
        total_samples += len(sample_paths)

        summary_path = episode_dir / "summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
            expected = int(summary.get("num_samples", -1))
            if expected != len(sample_paths):
                print(
                    f"ERROR {episode_dir}: summary says {expected} samples, "
                    f"found {len(sample_paths)}"
                )
                total_errors += 1

        for sample_path in sample_paths:
            errors, info = validate_sample(sample_path)
            for error in errors:
                total_errors += 1
                if total_errors <= max_errors:
                    print(f"ERROR {sample_path}: {error}")

            if "stage" in info:
                stage_counts[info["stage"]] += 1
            if "center_image_shape" in info:
                image_shapes[info["center_image_shape"]] += 1
            if "z_offset" in info:
                z_offsets.append(info["z_offset"])
            if "port_xyz" in info:
                port_xyz_by_task[task_name].append(info["port_xyz"])

    print(f"Dataset root: {root}")
    print(f"Episodes: {len(episode_dirs)}")
    print(f"Samples: {total_samples}")
    print(f"Task episodes: {dict(task_counts)}")
    print(f"Samples per episode: {samples_per_episode}")
    print(f"Stage counts: {dict(stage_counts)}")
    print(f"Center image shapes: {dict(image_shapes)}")

    if z_offsets:
        print(
            "z_offset range: "
            f"{min(z_offsets):.5f} to {max(z_offsets):.5f}"
        )

    for task_name, points in port_xyz_by_task.items():
        xyz = np.stack(points, axis=0)
        lo = xyz.min(axis=0)
        hi = xyz.max(axis=0)
        print(
            f"{task_name} port xyz range: "
            f"x[{lo[0]:.4f}, {hi[0]:.4f}] "
            f"y[{lo[1]:.4f}, {hi[1]:.4f}] "
            f"z[{lo[2]:.4f}, {hi[2]:.4f}]"
        )

    if total_errors:
        print(f"Validation finished with {total_errors} error(s).")
        return 1

    print("Validation passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/tmp/aic_teacher_dataset"),
        help="Teacher dataset root directory.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum number of detailed validation errors to print.",
    )
    args = parser.parse_args()
    return inspect_dataset(args.root, args.max_errors)


if __name__ == "__main__":
    raise SystemExit(main())
