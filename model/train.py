"""
Two-stage training pipeline for the ElephantVoices rumble-isolation U-Net.

Stage 1 (mask-only):
    Train a depth-4 U-Net with `embedding_dim=0` using BCEWithLogitsLoss on
    ElephantSpectrogramDataset with SNR-jitter augmentation. Saves the final
    weights to `backend/weights/unet_baseline.pth`.

Stage 2 (instance segmentation):
    Instantiate the U-Net with `embedding_dim=20`, load the Stage-1 weights
    (the mask head transfers exactly; the new embedding head stays at its
    Kaiming init), and continue training on InstanceMixDataset with
    InstanceSegLoss (= BCEWithLogitsLoss + DeepClusteringLoss). Saves to
    `backend/weights/unet_instance.pth`.

Both stages use an 80/20 train/validation split **at the file level** (no
file appears in both splits) so validation is a true generalisation probe.
Each stage runs its own CosineAnnealingLR schedule.

Override epoch counts at runtime with env vars for quick smoke tests:
    STAGE1_EPOCHS=1 STAGE2_EPOCHS=1 python model/train.py
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data_engine import CallAnnotation, NoiseBank, load_raven_annotations
from dataset import (
    CHUNK_SAMPLES,
    ElephantSpectrogramDataset,
    InstanceMixDataset,
    _discover_sheet,
    _synthesize_dummy_annotations,
)
from losses import InstanceSegLoss
from unet import UNet


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

STAGE1_EPOCHS: int = int(os.environ.get("STAGE1_EPOCHS", "30"))
STAGE2_EPOCHS: int = int(os.environ.get("STAGE2_EPOCHS", "30"))
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "4"))
LEARNING_RATE: float = float(os.environ.get("LEARNING_RATE", "1e-3"))
WEIGHT_DECAY: float = 1e-4
EMBEDDING_DIM: int = 20
DC_WEIGHT: float = 1.0
MASK_WEIGHT: float = 1.0
TRAIN_FRAC: float = float(os.environ.get("TRAIN_FRAC", "0.8"))
SEED: int = int(os.environ.get("SEED", "42"))
NUM_WORKERS: int = 0  # In-memory audio cache - workers add overhead here.


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_annotations_by_file(
    annotations: Sequence[CallAnnotation],
    train_frac: float,
    seed: int,
) -> Tuple[List[CallAnnotation], List[CallAnnotation]]:
    """80/20 split so that every file is fully in one partition.

    Splitting by annotations would let the same file leak into both train
    and val; splitting by files eliminates that source of optimism.
    """
    files = sorted({ann.file_name for ann in annotations})
    if not files:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(files)
    split_at = max(1, int(round(len(files) * train_frac)))
    split_at = min(split_at, len(files) - 1) if len(files) >= 2 else split_at
    train_files = set(files[:split_at])
    val_files = set(files[split_at:])
    if not val_files and files:
        val_files = {files[-1]}
        train_files = set(files[:-1])

    train = [a for a in annotations if a.file_name in train_files]
    val = [a for a in annotations if a.file_name in val_files]
    return train, val


# ---------------------------------------------------------------------------
# Epoch runners
# ---------------------------------------------------------------------------

def run_mask_epoch(
    model: UNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_training: bool,
) -> Dict[str, float]:
    model.train(is_training)
    total = 0.0
    n = 0
    with torch.set_grad_enabled(is_training):
        for spec, mask in loader:
            spec = spec.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(spec)
            loss = criterion(logits, mask)
            if is_training:
                loss.backward()
                optimizer.step()
            total += float(loss.item())
            n += 1
    return {"loss": total / max(n, 1)}


def run_instance_epoch(
    model: UNet,
    loader: DataLoader,
    criterion: InstanceSegLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_training: bool,
) -> Dict[str, float]:
    model.train(is_training)
    total = 0.0
    mask_total = 0.0
    dc_total = 0.0
    n = 0
    with torch.set_grad_enabled(is_training):
        for spec, mask, inst_y in loader:
            spec = spec.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            inst_y = inst_y.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            out = model(spec)
            if not isinstance(out, tuple):
                raise RuntimeError(
                    "Stage 2 requires a U-Net built with embedding_dim > 0"
                )
            mask_logits, embeddings = out
            losses = criterion(mask_logits, embeddings, mask, inst_y)
            loss = losses["total"]
            if is_training:
                loss.backward()
                optimizer.step()
            total += float(loss.item())
            mask_total += float(losses["mask"].item())
            dc_total += float(losses["dc"].item())
            n += 1
    n = max(n, 1)
    return {"loss": total / n, "mask": mask_total / n, "dc": dc_total / n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    seed_everything(SEED)

    project_root = Path(__file__).resolve().parent.parent
    audio_dir = project_root / "Audio Files (04-10-2026)"
    weights_dir = project_root / "backend" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load annotations via auto-discovery --------------------------------
    sheet = _discover_sheet(project_root)
    if sheet is not None:
        print(f"[train] Annotations: {sheet}")
        annotations = load_raven_annotations(sheet)
    else:
        print("[train] No spreadsheet found - using synthesized dummy annotations.")
        annotations = _synthesize_dummy_annotations(audio_dir)
    print(f"[train] total annotations: {len(annotations)}")
    unique_files = sorted({a.file_name for a in annotations})
    print(f"[train] unique source files: {len(unique_files)}")

    # ---- 80/20 file-level split --------------------------------------------
    train_anns, val_anns = split_annotations_by_file(annotations, TRAIN_FRAC, SEED)
    print(f"[train] split -> train: {len(train_anns)} anns | val: {len(val_anns)} anns")

    # ---- Noise bank for SNR jitter -----------------------------------------
    print("[train] building noise bank from negative space...")
    noise_bank = NoiseBank(audio_dir, annotations, chunk_samples=CHUNK_SAMPLES)
    status = "empty" if noise_bank.is_empty() else f"{len(noise_bank._pool)} intervals"
    print(f"[train] noise bank: {status}")

    device = pick_device()
    print(f"[train] device: {device}")

    # =========================================================================
    # STAGE 1: baseline mask-only U-Net
    # =========================================================================
    print("\n" + "=" * 64)
    print("STAGE 1: baseline mask-only U-Net (BCEWithLogitsLoss)")
    print("=" * 64)

    train_ds = ElephantSpectrogramDataset(
        train_anns, audio_dir, train=True, noise_bank=noise_bank
    )
    val_ds = ElephantSpectrogramDataset(
        val_anns, audio_dir, train=False, noise_bank=None
    )
    print(f"[stage1] train ds: {len(train_ds)} | val ds: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = UNet(in_channels=1, out_channels=1, embedding_dim=0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage1] parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(STAGE1_EPOCHS, 1), eta_min=LEARNING_RATE * 0.01
    )

    best_val = float("inf")
    for epoch in range(1, STAGE1_EPOCHS + 1):
        train_stats = run_mask_epoch(model, train_loader, criterion, optimizer, device, True)
        val_stats = run_mask_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        best_val = min(best_val, val_stats["loss"])
        print(
            f"[stage1] epoch {epoch:02d}/{STAGE1_EPOCHS} | "
            f"train {train_stats['loss']:.4f} | "
            f"val {val_stats['loss']:.4f} | "
            f"lr {lr_now:.2e}"
        )

    baseline_path = weights_dir / "unet_baseline.pth"
    torch.save(model.state_dict(), baseline_path)
    print(f"[stage1] saved {baseline_path} (best val {best_val:.4f})")

    # =========================================================================
    # STAGE 2: instance segmentation with Deep Clustering embedding head
    # =========================================================================
    print("\n" + "=" * 64)
    print("STAGE 2: instance seg (BCE + DeepClusteringLoss)")
    print("=" * 64)

    train_file_count = len({a.file_name for a in train_anns})
    val_file_count = len({a.file_name for a in val_anns})
    if train_file_count < 2 or val_file_count < 2:
        print(
            f"[stage2] skipped: need >=2 files per split "
            f"(train={train_file_count}, val={val_file_count}). "
            "The baseline weights are still saved; drop the real CSV and rerun."
        )
        return

    inst_model = UNet(
        in_channels=1, out_channels=1, embedding_dim=EMBEDDING_DIM
    ).to(device)
    # strict=False because the new model has out_embed.* keys that the Stage-1
    # state_dict doesn't. Every key it DOES have must match, so any mismatch
    # in the mask head would still raise.
    baseline_state = torch.load(baseline_path, map_location=device, weights_only=True)
    missing, unexpected = inst_model.load_state_dict(baseline_state, strict=False)
    print(
        f"[stage2] loaded baseline | missing {len(missing)} | unexpected {len(unexpected)}"
    )

    inst_train_ds = InstanceMixDataset(
        train_anns, audio_dir, num_sources=2, length=len(train_anns)
    )
    inst_val_ds = InstanceMixDataset(
        val_anns, audio_dir, num_sources=2, length=max(16, len(val_anns))
    )
    print(f"[stage2] train mixtures: {len(inst_train_ds)} | val mixtures: {len(inst_val_ds)}")

    inst_train_loader = DataLoader(
        inst_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    inst_val_loader = DataLoader(
        inst_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    inst_criterion = InstanceSegLoss(dc_weight=DC_WEIGHT, mask_weight=MASK_WEIGHT)
    inst_optimizer = AdamW(
        inst_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    inst_scheduler = CosineAnnealingLR(
        inst_optimizer, T_max=max(STAGE2_EPOCHS, 1), eta_min=LEARNING_RATE * 0.01
    )

    best_val_inst = float("inf")
    for epoch in range(1, STAGE2_EPOCHS + 1):
        train_stats = run_instance_epoch(
            inst_model, inst_train_loader, inst_criterion, inst_optimizer, device, True
        )
        val_stats = run_instance_epoch(
            inst_model, inst_val_loader, inst_criterion, inst_optimizer, device, False
        )
        inst_scheduler.step()
        lr_now = inst_optimizer.param_groups[0]["lr"]
        best_val_inst = min(best_val_inst, val_stats["loss"])
        print(
            f"[stage2] epoch {epoch:02d}/{STAGE2_EPOCHS} | "
            f"train total {train_stats['loss']:.4f} "
            f"(mask {train_stats['mask']:.4f} dc {train_stats['dc']:.4f}) | "
            f"val total {val_stats['loss']:.4f} "
            f"(mask {val_stats['mask']:.4f} dc {val_stats['dc']:.4f}) | "
            f"lr {lr_now:.2e}"
        )

    inst_path = weights_dir / "unet_instance.pth"
    torch.save(inst_model.state_dict(), inst_path)
    print(f"[stage2] saved {inst_path} (best val {best_val_inst:.4f})")
    print("[train] done.")


if __name__ == "__main__":
    main()
