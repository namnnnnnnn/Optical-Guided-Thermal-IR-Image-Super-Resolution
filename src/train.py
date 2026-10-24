"""
train.py
Role: Full training loop for AttentionThermalSR model.
Loss: MSE + 0.1 * Gradient Loss
Optimizer: Adam lr=1e-3
Scheduler: CosineAnnealingLR T_max=EPOCHS (25)
Training subset: first 1,000 scenes (700 train / 150 val / 150 test)
Saves best model checkpoint based on validation loss.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import ThermalOpticalDataset, get_dataloaders
from model import AttentionThermalSR
from config import DATA_ROOT, CHECKPOINT_PATH


# ─── Config ───────────────────────────────────────────────────────────────────

ROOT_DIR       = DATA_ROOT
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
MAX_SAMPLES    = 1000   # training subset size (compute constraint)

EPOCHS        = 25
BATCH_SIZE    = 8
LEARNING_RATE = 1e-3
GRAD_LAMBDA   = 0.1    # weight for gradient loss


# ─── Loss Functions ───────────────────────────────────────────────────────────

def mse_loss(output, target):
    """Pixel-level accuracy loss."""
    return F.mse_loss(output, target)


def gradient_loss(output, target):
    """
    Edge sharpness loss.
    Compares spatial gradients of prediction vs target in X and Y directions.
    Forces model to reconstruct edges, not just smooth blobs.
    """
    # X direction (horizontal differences)
    output_dx = output[:, :, :, 1:] - output[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    # Y direction (vertical differences)
    output_dy = output[:, :, 1:, :] - output[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    return F.l1_loss(output_dx, target_dx) + F.l1_loss(output_dy, target_dy)


def total_loss(output, target):
    """Combined loss: MSE + 0.1 * Gradient Loss."""
    return mse_loss(output, target) + GRAD_LAMBDA * gradient_loss(output, target)


# ─── Training Loop ─────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load data
    print("[Train] Loading dataset...")
    train_loader, val_loader, _ = get_dataloaders(ROOT_DIR, batch_size=BATCH_SIZE, max_samples=MAX_SAMPLES)

    # Initialize model
    model = AttentionThermalSR().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')

    print(f"\n[Train] Starting training for {EPOCHS} epochs...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10} | {'LR':>10} | {'Status'}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):

        # ── Training phase ──
        model.train()
        train_loss_sum = 0.0

        for optical, thermal_lr, thermal_hr in train_loader:
            optical    = optical.to(device)
            thermal_lr = thermal_lr.to(device)
            thermal_hr = thermal_hr.to(device)

            optimizer.zero_grad()
            output = model(thermal_lr, optical)
            loss   = total_loss(output, thermal_hr)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        train_loss_avg = train_loss_sum / len(train_loader)

        # ── Validation phase ──
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for optical, thermal_lr, thermal_hr in val_loader:
                optical    = optical.to(device)
                thermal_lr = thermal_lr.to(device)
                thermal_hr = thermal_hr.to(device)

                output   = model(thermal_lr, optical)
                val_loss = total_loss(output, thermal_hr)
                val_loss_sum += val_loss.item()

        val_loss_avg = val_loss_sum / len(val_loader)

        # ── Scheduler step ──
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Save best checkpoint ──
        status = ""
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            status = "✓ Saved"

        # ── Print epoch summary ──
        print(f"{epoch:>6} | {train_loss_avg:>12.6f} | {val_loss_avg:>10.6f} | {current_lr:>10.6f} | {status}")

    print("\n" + "=" * 65)
    print(f"[Train] Training complete.")
    print(f"[Train] Best val loss : {best_val_loss:.6f}")
    print(f"[Train] Checkpoint    : {CHECKPOINT_PATH}")
    print("=" * 65)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()