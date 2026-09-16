#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aic_submission.perception.dataset import discover_teacher_samples, split_by_episode
from aic_submission.perception.relative_dataset import (
    DEFAULT_CAMERA_KEYS,
    task_name_from_id,
)
from aic_submission.perception.target_pose_dataset import TargetTcpPoseDataset
from aic_submission.perception.target_pose_model import (
    MultiCameraTargetTcpPoseRegressor,
)


def _sample_task(sample) -> dict:
    metadata_path = sample.path.parent / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)["task"]


def _matches_task_filter(sample, task_filter: str) -> bool:
    if task_filter == "all":
        return True

    task = _sample_task(sample)
    plug_type = str(task.get("plug_type", ""))
    port_type = str(task.get("port_type", ""))
    port_name = str(task.get("port_name", ""))

    if task_filter == "sfp":
        return plug_type == "sfp" and port_type == "sfp"
    if task_filter == "sc":
        return plug_type == "sc" and port_type == "sc"
    if task_filter == "sfp_port_0":
        return plug_type == "sfp" and port_type == "sfp" and port_name == "sfp_port_0"
    if task_filter == "sfp_port_1":
        return plug_type == "sfp" and port_type == "sfp" and port_name == "sfp_port_1"

    raise ValueError(f"Unsupported task filter: {task_filter}")


def _filter_samples(samples: list, task_filter: str) -> list:
    return [sample for sample in samples if _matches_task_filter(sample, task_filter)]


def _quat_angle_error_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1.0e-8)
    target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1.0e-8)
    dots = np.abs(np.sum(pred * target, axis=1))
    dots = np.clip(dots, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def _summarize(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("AIC_SUBMISSION/runs/target_pose_100ep/best_model.pt"),
    )
    parser.add_argument("--root", type=Path, default=Path("/tmp/aic_teacher_dataset_100"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split", choices=("val", "all"), default="val")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    target_stats = checkpoint["target_stats"]
    camera_keys = tuple(config.get("camera_keys", DEFAULT_CAMERA_KEYS))
    task_vector_mode = str(config.get("task_vector_mode", "basic"))
    task_dim = int(config.get("task_dim", 4))
    task_filter = str(config.get("task_filter", "all"))

    samples = discover_teacher_samples(args.root, task_vector_mode=task_vector_mode)
    samples = _filter_samples(samples, task_filter)
    train_samples, val_samples = split_by_episode(samples)
    eval_samples = val_samples if args.split == "val" else samples
    dataset = TargetTcpPoseDataset(
        eval_samples,
        target_mean=target_stats["mean"],
        target_std=target_stats["std"],
        camera_keys=camera_keys,
        include_metadata=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiCameraTargetTcpPoseRegressor(
        num_cameras=len(camera_keys),
        task_dim=task_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_mean = torch.tensor(target_stats["mean"], dtype=torch.float32, device=device)
    target_std = torch.tensor(target_stats["std"], dtype=torch.float32, device=device)

    pos_errors_mm: list[float] = []
    angle_errors_deg: list[float] = []
    by_stage_pos = defaultdict(list)
    by_stage_ang = defaultdict(list)
    by_task_pos = defaultdict(list)
    by_task_ang = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            task = batch["task"].to(device, non_blocking=True)
            task_id = batch["task_id"].to(device, non_blocking=True)
            target_vector = batch["target_vector"].numpy()

            pred_norm = model(images, state, task, task_id)
            pred = (pred_norm * target_std + target_mean).cpu().numpy()

            pos_error = np.linalg.norm(pred[:, :3] - target_vector[:, :3], axis=1)
            angle_error = _quat_angle_error_deg(pred[:, 3:7], target_vector[:, 3:7])

            pos_errors_mm.extend((pos_error * 1000.0).tolist())
            angle_errors_deg.extend(angle_error.tolist())

            stages = batch["stage"]
            task_ids = batch["task_id"].numpy().tolist()
            for i, stage in enumerate(stages):
                by_stage_pos[stage].append(float(pos_error[i] * 1000.0))
                by_stage_ang[stage].append(float(angle_error[i]))
                task_name = task_name_from_id(int(task_ids[i]))
                by_task_pos[task_name].append(float(pos_error[i] * 1000.0))
                by_task_ang[task_name].append(float(angle_error[i]))

    summary = {
        "checkpoint": str(args.checkpoint),
        "root": str(args.root),
        "split": args.split,
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_eval": len(eval_samples),
        "device": str(device),
        "task_vector_mode": task_vector_mode,
        "task_dim": task_dim,
        "task_filter": task_filter,
        "position_error_mm": _summarize(pos_errors_mm),
        "angle_error_deg": _summarize(angle_errors_deg),
        "by_stage_position_error_mm": {
            key: _summarize(values) for key, values in sorted(by_stage_pos.items())
        },
        "by_stage_angle_error_deg": {
            key: _summarize(values) for key, values in sorted(by_stage_ang.items())
        },
        "by_task_position_error_mm": {
            key: _summarize(values) for key, values in sorted(by_task_pos.items())
        },
        "by_task_angle_error_deg": {
            key: _summarize(values) for key, values in sorted(by_task_ang.items())
        },
    }

    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
