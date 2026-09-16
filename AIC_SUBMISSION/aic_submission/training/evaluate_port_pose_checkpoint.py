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

from aic_submission.perception.dataset import (
    TeacherPortPoseDataset,
    discover_teacher_samples,
    split_by_episode,
)
from aic_submission.perception.model import ResNetPortPoseRegressor


def _task_name(task_vector: np.ndarray) -> str:
    plug = "sfp" if task_vector[0] > task_vector[1] else "sc"
    port = "sfp" if task_vector[2] > task_vector[3] else "sc"
    return f"{plug}->{port}"


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
    parser.add_argument("--image-key", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # This checkpoint is generated locally by train_port_pose_smoke.py and includes
    # argparse Path objects in its metadata, so PyTorch's weights-only loader cannot
    # decode it.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})

    root = args.root or Path(checkpoint_args.get("root", "/tmp/aic_teacher_dataset"))
    image_key = args.image_key or checkpoint_args.get("image_key", "center_image")
    max_samples = (
        args.max_samples
        if args.max_samples is not None
        else int(checkpoint_args.get("max_samples", 0))
    )
    seed = args.seed if args.seed is not None else int(checkpoint_args.get("seed", 13))

    samples = discover_teacher_samples(root)
    if not samples:
        raise RuntimeError(f"No teacher samples found under {root}")

    rng = random.Random(seed)
    rng.shuffle(samples)
    if max_samples and max_samples > 0:
        samples = samples[:max_samples]

    train_samples, val_samples = split_by_episode(samples)
    if args.split == "train":
        eval_samples = train_samples
    elif args.split == "all":
        eval_samples = samples
    else:
        eval_samples = val_samples

    dataset = TeacherPortPoseDataset(
        eval_samples, image_key=image_key, include_metadata=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetPortPoseRegressor().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_errors: list[float] = []
    all_abs_axis_errors: list[np.ndarray] = []
    by_task: dict[str, list[float]] = defaultdict(list)
    by_stage: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            task = batch["task"].to(device, non_blocking=True)
            target = batch["target_xyz"].to(device, non_blocking=True)

            pred = model(image, state, task)
            delta = pred - target
            errors = torch.linalg.vector_norm(delta, dim=-1).cpu().numpy()
            axis_errors = torch.abs(delta).cpu().numpy()
            tasks = batch["task"].cpu().numpy()

            all_errors.extend(float(error) for error in errors)
            all_abs_axis_errors.extend(axis_errors)
            for task_vector, error in zip(tasks, errors):
                by_task[_task_name(task_vector)].append(float(error))
            for stage, error in zip(batch["stage"], errors):
                by_stage[str(stage)].append(float(error))

    axis_mm = np.asarray(all_abs_axis_errors, dtype=np.float64) * 1000.0
    report = {
        "checkpoint": str(args.checkpoint),
        "root": str(root),
        "split": args.split,
        "image_key": image_key,
        "max_samples": max_samples,
        "seed": seed,
        "device": str(device),
        "summary": _summarize(all_errors),
        "axis_abs_error_mean_mm": {
            "x": float(np.mean(axis_mm[:, 0])),
            "y": float(np.mean(axis_mm[:, 1])),
            "z": float(np.mean(axis_mm[:, 2])),
        },
        "by_task": {name: _summarize(errors) for name, errors in sorted(by_task.items())},
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
