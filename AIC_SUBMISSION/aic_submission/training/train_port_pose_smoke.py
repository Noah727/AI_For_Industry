#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from aic_submission.perception.dataset import (
    TeacherPortPoseDataset,
    discover_teacher_samples,
    split_by_episode,
)
from aic_submission.perception.model import ResNetPortPoseRegressor


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
            image = batch["image"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            task = batch["task"].to(device, non_blocking=True)
            target = batch["target_xyz"].to(device, non_blocking=True)

            pred = model(image, state, task)
            loss = criterion(pred, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = image.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_count += batch_size

    return total_loss / max(1, total_count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/tmp/aic_teacher_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("AIC_SUBMISSION/runs/port_pose_smoke"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--image-key", type=str, default="center_image")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = discover_teacher_samples(args.root)
    if not samples:
        raise RuntimeError(f"No teacher samples found under {args.root}")

    random.shuffle(samples)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    train_samples, val_samples = split_by_episode(samples)
    train_dataset = TeacherPortPoseDataset(train_samples, image_key=args.image_key)
    val_dataset = TeacherPortPoseDataset(val_samples, image_key=args.image_key)

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
    model = ResNetPortPoseRegressor().to(device)
    criterion = nn.SmoothL1Loss(beta=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = {
        "root": str(args.root),
        "num_samples": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "device": str(device),
        "epochs": [],
    }

    print(
        f"Training port pose smoke model on {len(train_samples)} train / "
        f"{len(val_samples)} val samples using {device}."
    )

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
                    "args": vars(args),
                },
                best_path,
            )

    with open(args.output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved best checkpoint to {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
