"""
dataset.py
Role: Loads optical + thermal LR/HR pairs from processed dataset.
      Returns tensors ready for model input.
Dataset root: set via config.DATA_ROOT (env var THERMAL_SR_DATA)
Layout: <root>/<region>/<scene>/{optical.tif, thermal_lr.tif, thermal_hr.tif}
Scenes: 25,000 | Image size: 264x264 | Split: 70/15/15 (seed 42)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio


class ThermalOpticalDataset(Dataset):
    """
    Loads (optical, thermal_lr, thermal_hr) triplets from processed dataset.

    Returns:
        optical     : float32 tensor (3, 264, 264) - RGB guidance image
        thermal_lr  : float32 tensor (1, 264, 264) - degraded thermal input
        thermal_hr  : float32 tensor (1, 264, 264) - ground truth thermal
    """

    def __init__(self, root_dir, max_samples=None):
        """
        Args:
            root_dir: path to ssl4eo_processed folder
            max_samples: maximum number of samples to include (None for all)
        """
        self.root_dir = root_dir
        self.scene_paths = []
        self.max_samples = max_samples

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(
                f"Dataset folder not found: {root_dir}\n"
                f"Set it with:  $env:THERMAL_SR_DATA = \"<path to ssl4eo_processed>\"  (PowerShell)")

        # Walk through all region/scene folders and collect valid scene paths
        for region in sorted(os.listdir(root_dir)):
            region_path = os.path.join(root_dir, region)
            if not os.path.isdir(region_path):
                continue
            for scene in sorted(os.listdir(region_path)):
                scene_path = os.path.join(region_path, scene)
                if not os.path.isdir(scene_path):
                    continue

                # Only include scenes that have all 3 required files
                optical_path    = os.path.join(scene_path, 'optical.tif')
                thermal_lr_path = os.path.join(scene_path, 'thermal_lr.tif')
                thermal_hr_path = os.path.join(scene_path, 'thermal_hr.tif')

                if (os.path.exists(optical_path) and
                    os.path.exists(thermal_lr_path) and
                    os.path.exists(thermal_hr_path)):
                    self.scene_paths.append(scene_path)

        print(f"[Dataset] Total valid scenes found: {len(self.scene_paths)}")

        if max_samples is not None:
            self.scene_paths = self.scene_paths[:max_samples]
            print(f"[Dataset] Using first {len(self.scene_paths)} samples")

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        scene_path = self.scene_paths[idx]

        optical    = self._load_tif(os.path.join(scene_path, 'optical.tif'))     # (3, H, W)
        thermal_lr = self._load_tif(os.path.join(scene_path, 'thermal_lr.tif'))  # (1, H, W)
        thermal_hr = self._load_tif(os.path.join(scene_path, 'thermal_hr.tif'))  # (1, H, W)

        # Normalize each image to [0, 1]
        optical    = self._normalize(optical)
        thermal_lr = self._normalize(thermal_lr)
        thermal_hr = self._normalize(thermal_hr)

        # Convert to float32 tensors
        optical    = torch.tensor(optical,    dtype=torch.float32)
        thermal_lr = torch.tensor(thermal_lr, dtype=torch.float32)
        thermal_hr = torch.tensor(thermal_hr, dtype=torch.float32)

        return optical, thermal_lr, thermal_hr

    def _load_tif(self, path):
        """Read a GeoTIFF and return numpy array of shape (C, H, W)."""
        with rasterio.open(path) as src:
            arr = src.read()  # rasterio reads as (bands, H, W) automatically
        return arr.astype(np.float32)

    def _normalize(self, arr):
        """Normalize array to [0, 1] range. Add 1e-8 to avoid divide-by-zero."""
        return arr / (arr.max() + 1e-8)


def get_dataloaders(root_dir, batch_size=8, num_workers=0, max_samples=None):
    """
    Creates train, validation, and test DataLoaders.

    Args:
        root_dir   : path to ssl4eo_processed folder
        batch_size : number of scenes per batch (default 8 for 264x264)
        num_workers: 0 for Windows (avoids multiprocessing issues)

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset = ThermalOpticalDataset(
        root_dir,
        max_samples=max_samples)

    total = len(dataset)
    train_size = int(0.70 * total)
    val_size   = int(0.15 * total)
    test_size  = total - train_size - val_size  # remainder goes to test

    print(f"[Split] Train: {train_size} | Val: {val_size} | Test: {test_size}")

    # Fixed seed for reproducibility
    torch.manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# ─── Quick verification ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import DATA_ROOT
    ROOT = DATA_ROOT

    train_loader, val_loader, test_loader = get_dataloaders(ROOT, batch_size=8)

    # Check one batch
    optical, thermal_lr, thermal_hr = next(iter(train_loader))

    print("\n[Verification]")
    print(f"  optical     shape : {optical.shape}")     # (B, 3, 264, 264)
    print(f"  thermal_lr  shape : {thermal_lr.shape}")  # (B, 1, 264, 264)
    print(f"  thermal_hr  shape : {thermal_hr.shape}")  # (B, 1, 264, 264)
    print(f"  optical     range : [{optical.min():.3f}, {optical.max():.3f}]")
    print(f"  thermal_lr  range : [{thermal_lr.min():.3f}, {thermal_lr.max():.3f}]")
    print(f"  thermal_hr  range : [{thermal_hr.min():.3f}, {thermal_hr.max():.3f}]")
    print(f"  NaN in optical    : {torch.isnan(optical).any()}")
    print(f"  NaN in thermal_lr : {torch.isnan(thermal_lr).any()}")
    print(f"  NaN in thermal_hr : {torch.isnan(thermal_hr).any()}")
    print("\n✓ Dataset ready for model training")