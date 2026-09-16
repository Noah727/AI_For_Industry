#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from aic_submission.perception.keypoint_dataset import (
    TeacherKeypointDataset,
    discover_keypoint_samples,
    split_keypoint_samples,
)
from aic_submission.perception.keypoint_model import TinyKeypointUNet


def masked_heatmap_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    mask = valid[:, :, None, None]
    loss = (pred - target).pow(2) * mask
    denom = mask.sum() * target.shape[-1] * target.shape[-2]
    return loss.sum() / denom.clamp_min(1.0)


def _argmax_uv(heatmap: torch.Tensor) -> torch.Tensor:
    flat_index = heatmap.flatten(start_dim=-2).argmax(dim=-1)
    width = heatmap.shape[-1]
    u = flat_index % width
    v = torch.div(flat_index, width, rounding_mode="floor")
    return torch.stack([u, v], dim=-1).to(torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    heatmap_stride: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_error = 0.0
    total_count = 0
    total_keypoints = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["heatmap"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            coords = batch["coords_uv"].to(device, non_blocking=True)

            logits = model(image)
            loss = masked_heatmap_loss(logits, target, valid)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                pred_uv = _argmax_uv(torch.sigmoid(logits))
                error = torch.linalg.norm(pred_uv - coords, dim=-1) * valid
                keypoints = int(valid.sum().item())
                if keypoints:
                    total_error += float(error.sum().detach().cpu()) * heatmap_stride
                    total_keypoints += keypoints

            batch_size = image.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_count += batch_size

    mean_loss = total_loss / max(1, total_count)
    mean_error = total_error / max(1, total_keypoints)
    return mean_loss, mean_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/tmp/aic_teacher_dataset"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("AIC_SUBMISSION/runs/keypoint_smoke"),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--image-key", type=str, default="center_image")
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--heatmap-stride", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = discover_keypoint_samples(args.root, image_key=args.image_key)
    if not samples:
        raise RuntimeError(
            "No camera-calibrated keypoint samples found. Collect a short new "
            "TeacherRecorderPolicy dataset after the recorder patch, then rerun "
            f"with --root {args.root}."
        )

    random.shuffle(samples)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    train_samples, val_samples = split_keypoint_samples(samples)
    train_dataset = TeacherKeypointDataset(
        train_samples,
        sigma=args.sigma,
        heatmap_stride=args.heatmap_stride,
    )
    val_dataset = TeacherKeypointDataset(
        val_samples,
        sigma=args.sigma,
        heatmap_stride=args.heatmap_stride,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyKeypointUNet(out_channels=2, output_stride=args.heatmap_stride).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = {
        "root": str(args.root),
        "image_key": args.image_key,
        "num_samples": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "device": str(device),
        "epochs": [],
    }

    print(
        f"Training keypoint model on {len(train_samples)} train / "
        f"{len(val_samples)} val samples using {device}."
    )

    best_val = float("inf")
    best_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_px = run_epoch(
            model,
            train_loader,
            device,
            args.heatmap_stride,
            optimizer,
        )
        val_loss, val_px = run_epoch(
            model,
            val_loader,
            device,
            args.heatmap_stride,
        )
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.6f} train_px={train_px:.2f} "
            f"val_loss={val_loss:.6f} val_px={val_px:.2f}"
        )
        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_pixel_error": train_px,
                "val_loss": val_loss,
                "val_pixel_error": val_px,
            }
        )
        if val_px < best_val:
            best_val = val_px
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_pixel_error": val_px,
                    "image_key": args.image_key,
                    "sigma": args.sigma,
                    "heatmap_stride": args.heatmap_stride,
                    "args": {key: str(value) for key, value in vars(args).items()},
                },
                best_path,
            )

    with open(args.output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved best checkpoint to {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
