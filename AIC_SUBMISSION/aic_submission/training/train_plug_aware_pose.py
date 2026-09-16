#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from aic_submission.perception.dataset import discover_teacher_samples, split_by_episode
from aic_submission.perception.plug_aware_dataset import (
    PlugAwarePoseDataset,
    compute_plug_aware_target_stats,
)
from aic_submission.perception.relative_dataset import DEFAULT_CAMERA_KEYS
from aic_submission.perception.relative_model import MultiCameraRelativePortPoseRegressor


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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            task = batch["task"].to(device, non_blocking=True)
            task_id = batch["task_id"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            pred = model(images, state, task, task_id)
            loss = criterion(pred, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_count += batch_size

    return total_loss / max(1, total_count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/tmp/aic_teacher_dataset_100"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("AIC_SUBMISSION/runs/plug_aware_pose_100ep"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    samples = discover_teacher_samples(args.root)
    if not samples:
        raise RuntimeError(f"No teacher samples found under {args.root}")

    train_samples, val_samples = split_by_episode(samples)
    train_samples, val_samples = _cap_samples(
        train_samples, val_samples, args.max_samples, rng
    )
    target_stats = compute_plug_aware_target_stats(train_samples)

    camera_keys = DEFAULT_CAMERA_KEYS
    train_dataset = PlugAwarePoseDataset(
        train_samples,
        target_mean=target_stats["mean"],
        target_std=target_stats["std"],
        camera_keys=camera_keys,
    )
    val_dataset = PlugAwarePoseDataset(
        val_samples,
        target_mean=target_stats["mean"],
        target_std=target_stats["std"],
        camera_keys=camera_keys,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
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
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "root": str(args.root),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_samples": args.max_samples,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "camera_keys": list(camera_keys),
        "output_dim": 14,
        "target_layout": (
            "port_relative_xyz plug_relative_xyz "
            "port_quat_xyzw plug_quat_xyzw"
        ),
    }
    history = {
        **config,
        "num_samples": len(train_samples) + len(val_samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "device": str(device),
        "target_stats": target_stats,
        "epochs": [],
    }

    print(
        f"Training plug-aware model on {len(train_samples)} train / "
        f"{len(val_samples)} val samples using {device}."
    )
    print(f"Target mean: {target_stats['mean']}")
    print(f"Target std:  {target_stats['std']}")

    best_val = float("inf")
    best_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        print(
            f"epoch {epoch:03d} train_smooth_l1={train_loss:.6f} "
            f"val_smooth_l1={val_loss:.6f}"
        )
        history["epochs"].append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": config,
                    "target_stats": target_stats,
                },
                best_path,
            )

    with open(args.output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved best checkpoint to {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
