#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aic_submission.perception.dataset import discover_teacher_samples, split_by_episode
from aic_submission.perception.relative_dataset import (
    DEFAULT_CAMERA_KEYS,
    RelativePortPoseDataset,
    task_name_from_id,
)
from aic_submission.perception.relative_model import (
    MultiCameraRelativePortPoseRegressor,
)


def _cap_samples(
    train_samples: list,
    val_samples: list,
    max_samples: int,
    rng: random.Random,
) -> tuple[list, list]:
    if max_samples <= 0:
        return train_samples, val_samples

    train_fraction = len(train_samples) / max(1, len(train_samples) + len(val_samples))
    train_count = min(len(train_samples), max(1, round(max_samples * train_fraction)))
    val_count = min(len(val_samples), max(1, max_samples - train_count))

    train_samples = list(train_samples)
    val_samples = list(val_samples)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples[:train_count], val_samples[:val_count]


def _summarize(errors_m: list[float]) -> dict[str, float]:
    arr_mm = np.asarray(errors_m, dtype=np.float64) * 1000.0
    return {
        "count": int(arr_mm.size),
        "mean_mm": float(np.mean(arr_mm)),
        "median_mm": float(np.median(arr_mm)),
        "p90_mm": float(np.percentile(arr_mm, 90)),
        "max_mm": float(np.max(arr_mm)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    target_stats = checkpoint["target_stats"]

    root = args.root or Path(config.get("root", "/tmp/aic_teacher_dataset_100"))
    camera_keys = tuple(config.get("camera_keys", DEFAULT_CAMERA_KEYS))
    task_vector_mode = str(config.get("task_vector_mode", "basic"))
    task_dim = int(config.get("task_dim", 4))
    max_samples = (
        args.max_samples
        if args.max_samples is not None
        else int(config.get("max_samples", 0))
    )
    seed = args.seed if args.seed is not None else int(config.get("seed", 13))

    samples = discover_teacher_samples(root, task_vector_mode=task_vector_mode)
    if not samples:
        raise RuntimeError(f"No teacher samples found under {root}")

    rng = random.Random(seed)
    train_samples, val_samples = split_by_episode(samples)
    train_samples, val_samples = _cap_samples(
        train_samples, val_samples, max_samples, rng
    )
    if args.split == "train":
        eval_samples = train_samples
    elif args.split == "all":
        eval_samples = train_samples + val_samples
    else:
        eval_samples = val_samples

    dataset = RelativePortPoseDataset(
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
    model = MultiCameraRelativePortPoseRegressor(
        num_cameras=len(camera_keys),
        task_dim=task_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_mean = torch.tensor(target_stats["mean"], dtype=torch.float32, device=device)
    target_std = torch.tensor(target_stats["std"], dtype=torch.float32, device=device)

    all_errors: list[float] = []
    all_relative_errors: list[float] = []
    all_abs_axis_errors: list[np.ndarray] = []
    by_task: dict[str, list[float]] = defaultdict(list)
    by_stage: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            task = batch["task"].to(device, non_blocking=True)
            task_id = batch["task_id"].to(device, non_blocking=True)
            tcp_xyz = batch["tcp_xyz"].to(device, non_blocking=True)
            port_xyz = batch["port_xyz"].to(device, non_blocking=True)
            target_relative = batch["target_relative_xyz"].to(device, non_blocking=True)

            pred_norm = model(images, state, task, task_id)
            pred_relative = pred_norm * target_std + target_mean
            pred_port = tcp_xyz + pred_relative

            delta = pred_port - port_xyz
            relative_delta = pred_relative - target_relative
            errors = torch.linalg.vector_norm(delta, dim=-1).cpu().numpy()
            relative_errors = (
                torch.linalg.vector_norm(relative_delta, dim=-1).cpu().numpy()
            )
            axis_errors = torch.abs(delta).cpu().numpy()
            task_ids = batch["task_id"].cpu().numpy()

            all_errors.extend(float(error) for error in errors)
            all_relative_errors.extend(float(error) for error in relative_errors)
            all_abs_axis_errors.extend(axis_errors)
            for task_index, error in zip(task_ids, errors):
                by_task[task_name_from_id(int(task_index))].append(float(error))
            for stage, error in zip(batch["stage"], errors):
                by_stage[str(stage)].append(float(error))

    axis_mm = np.asarray(all_abs_axis_errors, dtype=np.float64) * 1000.0
    report = {
        "checkpoint": str(args.checkpoint),
        "root": str(root),
        "split": args.split,
        "camera_keys": list(camera_keys),
        "task_vector_mode": task_vector_mode,
        "task_dim": task_dim,
        "max_samples": max_samples,
        "seed": seed,
        "device": str(device),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "target_stats": target_stats,
        "summary": _summarize(all_errors),
        "relative_summary": _summarize(all_relative_errors),
        "axis_abs_error_mean_mm": {
            "x": float(np.mean(axis_mm[:, 0])),
            "y": float(np.mean(axis_mm[:, 1])),
            "z": float(np.mean(axis_mm[:, 2])),
        },
        "by_task": {
            name: _summarize(errors) for name, errors in sorted(by_task.items())
        },
        "by_stage": {
            name: _summarize(errors) for name, errors in sorted(by_stage.items())
        },
    }

    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
