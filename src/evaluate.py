"""
evaluate.py
Role: Evaluates AttentionThermalSR on the test set: PSNR/SSIM/RMSE for the SR
      output and the LR-input baseline, plus unseen-only metrics (excluding
      scenes from the 1,000-scene training subset).
Output: outputs/{predictions, zoomed, diff_maps, attention_maps, metrics}/
"""

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from torch.utils.data import random_split

from dataset import get_dataloaders
from model import AttentionThermalSR
from config import DATA_ROOT, CHECKPOINT_PATH as CKPT, OUTPUT_DIR as OUT


# ─── Config ───────────────────────────────────────────────────────────────────

ROOT_DIR        = DATA_ROOT
CHECKPOINT_PATH = CKPT
OUTPUT_DIR      = OUT
TRAIN_SUBSET    = 1000    # must match MAX_SAMPLES in train.py

PREDICTIONS_DIR  = os.path.join(OUTPUT_DIR, "predictions")
ATTN_DIR         = os.path.join(OUTPUT_DIR, "attention_maps")
METRICS_DIR      = os.path.join(OUTPUT_DIR, "metrics")
ZOOM_DIR         = os.path.join(OUTPUT_DIR, "zoomed")
DIFF_DIR         = os.path.join(OUTPUT_DIR, "diff_maps")

BATCH_SIZE       = 8
MAX_SAVE_SAMPLES = 10
ZOOM_SIZE        = 80     # center crop size in pixels


# ─── Attention Map Hook ────────────────────────────────────────────────────────

spatial_attn_map = []

def hook_fn(module, input, output):
    spatial_attn_map.append(output.detach().cpu())


# ─── Metric Functions ──────────────────────────────────────────────────────────

def compute_rmse(hr, sr):
    return float(np.sqrt(np.mean((hr - sr) ** 2)))

def compute_psnr(hr, sr):
    return float(psnr(hr, sr, data_range=1.0))

def compute_ssim(hr, sr):
    return float(ssim(hr, sr, data_range=1.0))


def get_training_scene_indices(total_scenes, subset=TRAIN_SUBSET):
    """
    Re-creates the exact train split used by train.py (first `subset` scenes,
    70/15/15, seed 42) and returns the dataset indices used for training.
    Used to flag test scenes the model has already seen.
    """
    n = min(subset, total_scenes)
    tr = int(0.70 * n); va = int(0.15 * n); te = n - tr - va
    torch.manual_seed(42)
    train_part, _, _ = random_split(range(n), [tr, va, te])
    return set(train_part.indices)


# ─── Save Full Comparison Plot ─────────────────────────────────────────────────

def save_comparison(thermal_lr, thermal_sr, thermal_hr, psnr_val, ssim_val, idx):
    """Saves side-by-side LR | SR | HR comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(thermal_lr, cmap='hot')
    axes[0].set_title("LR Thermal (Input)\n[Blurry — model input]", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(thermal_sr, cmap='hot')
    axes[1].set_title(
        f"SR Thermal (Predicted)\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}",
        fontsize=12
    )
    axes[1].axis('off')

    axes[2].imshow(thermal_hr, cmap='hot')
    axes[2].set_title("HR Thermal (Ground Truth)\n[Sharp — target]", fontsize=12)
    axes[2].axis('off')

    plt.suptitle("Thermal Super-Resolution Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTIONS_DIR, f"sample_{idx:03d}.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ─── Save Zoomed Crop Comparison ──────────────────────────────────────────────

def save_zoomed(thermal_lr, thermal_sr, thermal_hr, idx):
    """
    Zooms into center patch to clearly show sharpness difference.
    Uses interpolation='nearest' so pixels are crisp, not smoothed.
    """
    H, W = thermal_lr.shape
    cx, cy = H // 2, W // 2
    z = ZOOM_SIZE // 2

    lr_crop = thermal_lr[cx-z:cx+z, cy-z:cy+z]
    sr_crop = thermal_sr[cx-z:cx+z, cy-z:cy+z]
    hr_crop = thermal_hr[cx-z:cx+z, cy-z:cy+z]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(lr_crop, cmap='hot', interpolation='nearest')
    axes[0].set_title("LR Thermal — Zoomed\n[Blurry input]", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(sr_crop, cmap='hot', interpolation='nearest')
    axes[1].set_title("SR Thermal — Zoomed\n[Model output — sharper edges]", fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(hr_crop, cmap='hot', interpolation='nearest')
    axes[2].set_title("HR Thermal — Zoomed\n[Ground truth]", fontsize=12)
    axes[2].axis('off')

    plt.suptitle(f"Zoomed Center Crop ({ZOOM_SIZE}×{ZOOM_SIZE} px) — Detail Comparison",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ZOOM_DIR, f"zoom_{idx:03d}.png"), dpi=200, bbox_inches='tight')
    plt.close()


# ─── Save Difference / Error Map ──────────────────────────────────────────────

def save_diff_map(thermal_lr, thermal_sr, thermal_hr, idx):
    """
    Shows error maps:
    - Left:  HR ground truth
    - Middle: LR error map  (bright = LR was wrong here)
    - Right:  SR error map  (bright = SR still wrong here)
    SR error should be much darker than LR error = model improved those regions.
    """
    diff_lr_hr = np.abs(thermal_hr - thermal_lr)
    diff_sr_hr = np.abs(thermal_hr - thermal_sr)

    # Shared colour scale fitted to the data (errors are ~0.01, a fixed 0-0.3
    # scale would make both maps look uniformly dark)
    vmax = max(np.percentile(diff_lr_hr, 99), np.percentile(diff_sr_hr, 99), 1e-3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(thermal_hr, cmap='hot')
    axes[0].set_title("HR Thermal (Ground Truth)", fontsize=12)
    axes[0].axis('off')

    im1 = axes[1].imshow(diff_lr_hr, cmap='RdYlGn_r', vmin=0, vmax=vmax)
    axes[1].set_title("LR Error Map\n[Bright = large error in LR]", fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff_sr_hr, cmap='RdYlGn_r', vmin=0, vmax=vmax)
    axes[2].set_title("SR Error Map\n[Bright = remaining error in SR]", fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.suptitle(f"Absolute error vs HR (shared scale 0-{vmax:.3f}) | "
                 f"mean LR error {diff_lr_hr.mean():.4f} | mean SR error {diff_sr_hr.mean():.4f}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DIFF_DIR, f"diff_{idx:03d}.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ─── Save Attention Map ────────────────────────────────────────────────────────

def save_attention_map(attn_tensor, optical_tensor, idx):
    attn = attn_tensor[0, 0].numpy()
    optical = optical_tensor[0].permute(1, 2, 0).numpy()
    optical = (optical - optical.min()) / (optical.max() - optical.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(optical)
    axes[0].set_title("Optical RGB Input")
    axes[0].axis('off')

    axes[1].imshow(attn, cmap='viridis')
    axes[1].set_title("Spatial Attention Map\n[Bright = model focused here]")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(ATTN_DIR, f"attn_{idx:03d}.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main Evaluation ───────────────────────────────────────────────────────────

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Using device: {device}")

    # Create all output directories
    for d in [PREDICTIONS_DIR, ATTN_DIR, METRICS_DIR, ZOOM_DIR, DIFF_DIR]:
        os.makedirs(d, exist_ok=True)

    # Load model
    model = AttentionThermalSR().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"[Evaluate] Loaded checkpoint: {CHECKPOINT_PATH}")

    # Register attention hook
    model.spatial_attention.sigmoid.register_forward_hook(hook_fn)

    # Load test set
    _, _, test_loader = get_dataloaders(ROOT_DIR, batch_size=BATCH_SIZE)
    test_indices = test_loader.dataset.indices          # positions in full dataset
    seen_in_training = get_training_scene_indices(len(test_loader.dataset.dataset))
    print(f"[Evaluate] Running on test set...\n")

    all_psnr, all_ssim, all_rmse = [], [], []
    lr_psnr, lr_ssim, lr_rmse = [], [], []
    seen_flags = []
    csv_rows = []
    sample_idx = 0

    with torch.no_grad():
        for batch_idx, (optical, thermal_lr, thermal_hr) in enumerate(test_loader):
            optical    = optical.to(device)
            thermal_lr = thermal_lr.to(device)
            thermal_hr = thermal_hr.to(device)

            spatial_attn_map.clear()
            output = model(thermal_lr, optical)

            for i in range(output.shape[0]):
                sr = output[i, 0].cpu().numpy()
                hr = thermal_hr[i, 0].cpu().numpy()
                lr = thermal_lr[i, 0].cpu().numpy()

                p = compute_psnr(hr, sr)
                s = compute_ssim(hr, sr)
                r = compute_rmse(hr, sr)

                # Baseline: LR input compared directly with HR ground truth
                bp = compute_psnr(hr, lr)
                bs = compute_ssim(hr, lr)
                br = compute_rmse(hr, lr)

                seen = int(test_indices[sample_idx] in seen_in_training)

                all_psnr.append(p)
                all_ssim.append(s)
                all_rmse.append(r)
                lr_psnr.append(bp)
                lr_ssim.append(bs)
                lr_rmse.append(br)
                seen_flags.append(seen)
                csv_rows.append([sample_idx, f"{p:.4f}", f"{s:.4f}", f"{r:.4f}",
                                 f"{bp:.4f}", f"{bs:.4f}", f"{br:.4f}", seen])

                if sample_idx < MAX_SAVE_SAMPLES:
                    save_comparison(lr, sr, hr, p, s, sample_idx)
                    save_zoomed(lr, sr, hr, sample_idx)
                    save_diff_map(lr, sr, hr, sample_idx)
                    if spatial_attn_map:
                        save_attention_map(
                            spatial_attn_map[0][i:i+1],
                            optical[i:i+1].cpu(),
                            sample_idx
                        )

                sample_idx += 1

    # Save CSV
    csv_path = os.path.join(METRICS_DIR, "results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "PSNR", "SSIM", "RMSE",
                         "LR_PSNR", "LR_SSIM", "LR_RMSE", "seen_in_training"])
        writer.writerows(csv_rows)

    all_psnr, all_ssim, all_rmse = map(np.array, (all_psnr, all_ssim, all_rmse))
    lr_psnr, lr_ssim, lr_rmse = map(np.array, (lr_psnr, lr_ssim, lr_rmse))
    unseen = np.array(seen_flags) == 0

    def row(name, a, b):
        return f"  {name:<6}: SR {a.mean():.4f} ± {a.std():.4f} | LR baseline {b.mean():.4f} | gain {a.mean() - b.mean():+.4f}"

    print("=" * 72)
    print("  Evaluation Results (Test Set)")
    print("=" * 72)
    print(row("PSNR", all_psnr, lr_psnr) + " dB")
    print(row("SSIM", all_ssim, lr_ssim))
    print(row("RMSE", all_rmse, lr_rmse))
    print("-" * 72)
    print(f"  Unseen-only ({unseen.sum()} of {len(unseen)} scenes, excludes training overlap):")
    print(row("PSNR", all_psnr[unseen], lr_psnr[unseen]) + " dB")
    print(row("SSIM", all_ssim[unseen], lr_ssim[unseen]))
    print(row("RMSE", all_rmse[unseen], lr_rmse[unseen]))
    print("=" * 72)
    print(f"  Samples evaluated : {sample_idx}")
    print(f"  Saved outputs:")
    print(f"    predictions/    → full LR | SR | HR comparison")
    print(f"    zoomed/         → center crop detail comparison")
    print(f"    diff_maps/      → error reduction visualization")
    print(f"    attention_maps/ → where model focused")
    print(f"  Metrics CSV       : {csv_path}")
    print("=" * 50)
    print("\n✓ Evaluation complete")


if __name__ == "__main__":
    evaluate()