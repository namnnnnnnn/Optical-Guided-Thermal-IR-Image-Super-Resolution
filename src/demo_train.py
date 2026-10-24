"""
demo_train.py
Role: Quick visual sanity check - runs the trained model on one scene and
      shows LR | SR | HR side by side.
"""

import torch
import matplotlib.pyplot as plt

from model import AttentionThermalSR
from dataset import ThermalOpticalDataset
from config import DATA_ROOT, CHECKPOINT_PATH

# Paths
ROOT_DIR = DATA_ROOT
CHECKPOINT = CHECKPOINT_PATH

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = AttentionThermalSR().to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

# Load one test sample
dataset = ThermalOpticalDataset(ROOT_DIR)
optical, thermal_lr, thermal_hr = dataset[0]

# Add batch dimension
optical = optical.unsqueeze(0).to(device)
thermal_lr = thermal_lr.unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    thermal_sr = model(thermal_lr, optical)

# Display
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(thermal_lr[0, 0].cpu(), cmap="inferno")
plt.title("LR Thermal")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(thermal_sr[0, 0].cpu(), cmap="inferno")
plt.title("SR Prediction")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(thermal_hr[0].cpu(), cmap="inferno")
plt.title("HR Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()