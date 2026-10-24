"""
export_samples.py
Role: Exports test-set scenes as 8-bit PNGs into demo/samples/ so the Gradio
      demo can be tried without the GeoTIFF dataset. Writes per scene:
      scene_XX_thermal_lr.png, scene_XX_optical.png, scene_XX_thermal_hr.png
Usage: python export_samples.py [num_samples]
"""

import os
import sys
import numpy as np
from PIL import Image

from dataset import get_dataloaders
from config import DATA_ROOT, SAMPLES_DIR


def to_uint8(arr):
    """(C, H, W) float in [0,1] -> (H, W) or (H, W, 3) uint8."""
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).round().astype(np.uint8)
    return arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)


def export(num_samples=5):
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    # Same seeded split as evaluate.py, so these are genuine test scenes
    _, _, test_loader = get_dataloaders(DATA_ROOT, batch_size=1)
    test_set = test_loader.dataset

    for i in range(min(num_samples, len(test_set))):
        optical, thermal_lr, thermal_hr = test_set[i]
        stem = os.path.join(SAMPLES_DIR, f"scene_{i:02d}")

        Image.fromarray(to_uint8(thermal_lr.numpy()), mode="L").save(f"{stem}_thermal_lr.png")
        Image.fromarray(to_uint8(optical.numpy()), mode="RGB").save(f"{stem}_optical.png")
        Image.fromarray(to_uint8(thermal_hr.numpy()), mode="L").save(f"{stem}_thermal_hr.png")
        print(f"[Export] {stem}_*.png")

    print(f"\n✓ Exported {min(num_samples, len(test_set))} scenes to {SAMPLES_DIR}")


if __name__ == "__main__":
    export(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
