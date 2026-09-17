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
from aic_submission.perception.plug_aware_dataset import PlugAwarePoseDataset
from aic_submission.perception.relative_dataset import (
    DEFAULT_CAMERA_KEYS,
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
    max_samples = (
        args.max_samples
        if args.max_samples is not None
        else int(config.get("max_samples", 0))
    )
    seed = args.seed if args.seed is not None else int(config.get("seed", 17))

    samples = discover_teacher_samples(root)
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

    dataset = PlugAwarePoseDataset(
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
        output_dim=14,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_mean = torch.tensor(target_stats["mean"], dtype=torch.float32, device=device)
    target_std = torch.tensor(target_stats["std"], dtype=torch.float32, device=device)

    port_errors: list[float] = []
    plug_errors: list[float] = []
    gap_errors: list[float] = []
    by_task_gap: dict[str, list[float]] = defaultdict(list)
    by_stage_gap: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            task = batch["task"].to(device, non_blocking=True)
            task_id = batch["task_id"].to(device, non_blocking=True)
            tcp_xyz = batch["tcp_xyz"].to(device, non_blocking=True)
            port_xyz = batch["port_xyz"].to(device, non_blocking=True)
            plug_xyz = batch["plug_xyz"].to(device, non_blocking=True)

            pred_norm = model(images, state, task, task_id)
            pred_rel = pred_norm * target_std + target_mean
            pred_port = tcp_xyz + pred_rel[:, :3]
            pred_plug = tcp_xyz + pred_rel[:, 3:6]

            port_delta = pred_port - port_xyz
            plug_delta = pred_plug - plug_xyz
            gap_delta = (pred_port - pred_plug) - (port_xyz - plug_xyz)
            port_batch = torch.linalg.vector_norm(port_delta, dim=-1).cpu().numpy()
            plug_batch = torch.linalg.vector_norm(plug_delta, dim=-1).cpu().numpy()
            gap_batch = torch.linalg.vector_norm(gap_delta, dim=-1).cpu().numpy()
            task_ids = batch["task_id"].cpu().numpy()

            port_errors.extend(float(error) for error in port_batch)
            plug_errors.extend(float(error) for error in plug_batch)
            gap_errors.extend(float(error) for error in gap_batch)
            for task_index, error in zip(task_ids, gap_batch):
                by_task_gap[task_name_from_id(int(task_index))].append(float(error))
            for stage, error in zip(batch["stage"], gap_batch):
                by_stage_gap[str(stage)].append(float(error))

    report = {
        "checkpoint": str(args.checkpoint),
        "root": str(root),
        "split": args.split,
        "camera_keys": list(camera_keys),
        "max_samples": max_samples,
        "seed": seed,
        "device": str(device),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "target_stats": target_stats,
        "port_summary": _summarize(port_errors),
        "plug_summary": _summarize(plug_errors),
        "gap_summary": _summarize(gap_errors),
        "gap_by_task": {
            name: _summarize(errors) for name, errors in sorted(by_task_gap.items())
        },
        "gap_by_stage": {
            name: _summarize(errors) for name, errors in sorted(by_stage_gap.items())
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
