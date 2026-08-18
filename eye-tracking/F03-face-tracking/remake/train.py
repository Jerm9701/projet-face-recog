from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from model import ArcMarginProduct, iresnet50


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    epochs: int = 24
    batch_size: int = 128
    workers: int = 8
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    embedding_dim: int = 512
    dropout: float = 0.4
    arcface_scale: float = 64.0
    arcface_margin: float = 0.5
    val_fraction: float = 0.02
    seed: int = 42
    amp: bool = True
    grad_clip: float = 5.0
    log_every: int = 100
    save_every: int = 1
    resume: str = ""
    max_steps: int = 200_000
    lr_milestones: tuple[int, ...] = (100_000, 140_000, 160_000)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train an ArcFace-style IR-ResNet on CASIA-WebFace."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="runs/casia_arcface")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--arcface-scale", type=float, default=64.0)
    parser.add_argument("--arcface-margin", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument(
        "--lr-milestones",
        type=int,
        nargs="+",
        default=[100_000, 140_000, 160_000],
    )
    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
        val_fraction=args.val_fraction,
        seed=args.seed,
        amp=not args.no_amp,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        max_steps=args.max_steps,
        lr_milestones=tuple(args.lr_milestones),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader, int]:
    # The paper normalizes RGB values with (x - 127.5) / 128.
    # ToTensor maps pixels to [0, 1], so the equivalent is:
    #   mean = 127.5 / 255 = 0.5
    #   std  = 128 / 255 ~= 0.5019608
    paper_std = 128.0 / 255.0

    train_transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(paper_std, paper_std, paper_std),
            ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(paper_std, paper_std, paper_std),
            ),
        ]
    )

    # Expected layout:
    # data_dir/
    #   person_0001/*.jpg
    #   person_0002/*.jpg
    base_dataset = datasets.ImageFolder(config.data_dir)
    num_classes = len(base_dataset.classes)

    val_size = max(1, int(len(base_dataset) * config.val_fraction))
    train_size = len(base_dataset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset = random_split(
        base_dataset, [train_size, val_size], generator=generator
    )

    # Give each split its own transform while preserving ImageFolder indices.
    train_dataset = DatasetWithTransform(train_subset, train_transform)
    val_dataset = DatasetWithTransform(val_subset, val_transform)

    common = dict(
        batch_size=config.batch_size,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.workers > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, num_classes


class DatasetWithTransform(torch.utils.data.Dataset):
    def __init__(self, subset: torch.utils.data.Subset, transform: Any) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        original_transform = self.subset.dataset.transform
        try:
            self.subset.dataset.transform = self.transform
            return self.subset[index]
        finally:
            self.subset.dataset.transform = original_transform


def set_iteration_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    global_step: int,
    milestones: tuple[int, ...],
) -> float:
    decay_count = sum(global_step >= milestone for milestone in milestones)
    lr = base_lr * (0.1 ** decay_count)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


@torch.no_grad()
def validate(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    backbone.eval()
    head.eval()

    loss_sum = 0.0
    correct = 0
    samples = 0

    for images, labels in tqdm(loader, desc="validate", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            embeddings = backbone(images)
            logits = head(embeddings, labels)
            loss = criterion(logits, labels)

        batch_size = images.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        samples += batch_size

    return loss_sum / max(samples, 1), correct / max(samples, 1)


def save_checkpoint(
    path: Path,
    backbone: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    config: TrainConfig,
    class_to_idx: dict[str, int],
) -> None:
    state = {
        "backbone": backbone.state_dict(),
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": asdict(config),
        "class_to_idx": class_to_idx,
    }
    torch.save(state, path)


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = config.amp and device.type == "cuda"

    train_loader, val_loader, num_classes = build_dataloaders(config)
    class_to_idx = train_loader.dataset.subset.dataset.class_to_idx

    backbone = iresnet50(
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)
    head = ArcMarginProduct(
        embedding_dim=config.embedding_dim,
        num_classes=num_classes,
        scale=config.arcface_scale,
        margin=config.arcface_margin,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        list(backbone.parameters()) + list(head.parameters()),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler(enabled=amp_enabled)

    start_epoch = 0
    global_step = 0
    best_val_loss = math.inf

    if config.resume:
        # Allow --resume to be used in multiple ways:
        #  - path to a checkpoint file
        #  - the literal strings 'True', 'true', '1', or 'auto' to mean "load output_dir/last.pt"
        resume_arg = config.resume
        if isinstance(resume_arg, str) and resume_arg.lower() in ("true", "1", "auto"):
            resume_path = output_dir / "last.pt"
        else:
            resume_path = Path(resume_arg)

        if not resume_path.exists():
            print(f"Resume checkpoint not found: {resume_path} — starting from scratch.")
        else:
            checkpoint = torch.load(resume_path, map_location="cpu")
            backbone.load_state_dict(checkpoint["backbone"]) 
            head.load_state_dict(checkpoint["head"]) 
            optimizer.load_state_dict(checkpoint["optimizer"]) 
            scaler.load_state_dict(checkpoint.get("scaler", {}))
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]
            best_val_loss = checkpoint.get("best_val_loss", math.inf)

    print(
        f"device={device} classes={num_classes} "
        f"train_images={len(train_loader.dataset)} "
        f"val_images={len(val_loader.dataset)}"
    )

    stop_training = False

    for epoch in range(start_epoch, config.epochs):
        backbone.train()
        head.train()

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{config.epochs}")
        for images, labels in progress:
            if global_step >= config.max_steps:
                stop_training = True
                break

            lr = set_iteration_lr(
                optimizer,
                config.lr,
                global_step,
                config.lr_milestones,
            )

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                embeddings = backbone(images)
                logits = head(embeddings, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()),
                    config.grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_samples += batch_size
            global_step += 1

            if global_step % config.log_every == 0:
                progress.set_postfix(
                    loss=f"{running_loss / running_samples:.4f}",
                    acc=f"{running_correct / running_samples:.3%}",
                    lr=f"{lr:.2e}",
                    step=global_step,
                )

        val_loss, val_acc = validate(
            backbone, head, val_loader, criterion, device, amp_enabled
        )
        train_loss = running_loss / max(running_samples, 1)
        train_acc = running_correct / max(running_samples, 1)

        print(
            f"epoch={epoch + 1} step={global_step} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3%} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3%}"
        )

        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                output_dir / "last.pt",
                backbone,
                head,
                optimizer,
                scaler,
                epoch,
                global_step,
                best_val_loss,
                config,
                class_to_idx,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                output_dir / "best.pt",
                backbone,
                head,
                optimizer,
                scaler,
                epoch,
                global_step,
                best_val_loss,
                config,
                class_to_idx,
            )

        if stop_training:
            break

    # Save a lightweight inference checkpoint containing only the backbone.
    torch.save(
        {
            "backbone": backbone.state_dict(),
            "embedding_dim": config.embedding_dim,
        },
        output_dir / "backbone_final.pt",
    )


if __name__ == "__main__":
    main()
