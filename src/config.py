"""
config.py
Role: Central place for all paths used by training, evaluation and the demo.
      Override via env vars: THERMAL_SR_DATA (dataset folder),
      THERMAL_SR_CHECKPOINT (.pth weights), THERMAL_SR_OUTPUTS (eval results).
"""

import os

# Repository root = parent of this src/ folder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _default_data_root():
    """Dataset inside the repo (data/) or next to it (../ssl4eo_processed)."""
    candidates = [os.path.join(REPO_ROOT, "data", "ssl4eo_processed"),
                  os.path.join(os.path.dirname(REPO_ROOT), "ssl4eo_processed")]
    return next((c for c in candidates if os.path.isdir(c)), candidates[0])


DATA_ROOT = os.environ.get("THERMAL_SR_DATA", _default_data_root())

CHECKPOINT_PATH = os.environ.get(
    "THERMAL_SR_CHECKPOINT",
    os.path.join(REPO_ROOT, "checkpoints", "best_model_1000.pth")
)

OUTPUT_DIR = os.environ.get(
    "THERMAL_SR_OUTPUTS",
    os.path.join(REPO_ROOT, "outputs")
)

SAMPLES_DIR = os.path.join(REPO_ROOT, "demo", "samples")
