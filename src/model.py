"""
model.py
Role: Defines the full AttentionThermalSR model.
Architecture:
    LR Thermal (1ch) --> ThermalEncoder -----------------------> Fusion
    HR Optical  (3ch) --> OpticalEncoder --> CBAM Attention --> Fusion
                                                                   |
                                                        Decoder --> HR Thermal Output
"""

import torch
import torch.nn as nn


# ─── Component 1: Base Conv Block ─────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU. Keeps H,W same via padding=1."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ─── Component 2: Thermal Encoder ─────────────────────────────────────────────

class ThermalEncoder(nn.Module):
    """
    Encodes LR thermal image into feature space.
    Input:  (B, 1, H, W)
    Output: (B, 64, H, W)
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU(1, 32),
            ConvBNReLU(32, 64)
        )

    def forward(self, x):
        return self.encoder(x)


# ─── Component 3: Optical Encoder ─────────────────────────────────────────────

class OpticalEncoder(nn.Module):
    """
    Encodes optical RGB image into feature space.
    Input:  (B, 3, H, W)
    Output: (B, 64, H, W)
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU(3, 32),
            ConvBNReLU(32, 64)
        )

    def forward(self, x):
        return self.encoder(x)


# ─── Component 4: Channel Attention (CBAM part 1) ─────────────────────────────

class ChannelAttention(nn.Module):
    """
    Asks: WHICH feature channels are most important?
    Squeezes spatial dims -> learns per-channel weights -> rescales input.
    Input:  (B, 64, H, W)
    Output: (B, 64, H, W)
    """
    def __init__(self, channels=64, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)   # (B, 64, 1, 1)
        self.fc = nn.Sequential(
            nn.Flatten(),                      # (B, 64)
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()                       # weights in [0, 1]
        )
        self.channels = channels

    def forward(self, x):
        weights = self.fc(self.pool(x))                    # (B, 64)
        weights = weights.view(-1, self.channels, 1, 1)    # (B, 64, 1, 1)
        return x * weights                                 # rescale channels


# ─── Component 5: Spatial Attention (CBAM part 2) ─────────────────────────────

class SpatialAttention(nn.Module):
    """
    Asks: WHERE in the image should we focus?
    Looks at avg + max across channels -> generates spatial attention map.
    Input:  (B, 64, H, W)
    Output: (B, 64, H, W)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)          # (B, 1, H, W)
        mx, _ = x.max(dim=1, keepdim=True)         # (B, 1, H, W)
        combined = torch.cat([avg, mx], dim=1)      # (B, 2, H, W)
        attn_map = self.sigmoid(self.conv(combined))# (B, 1, H, W)
        return x * attn_map                         # rescale spatial regions


# ─── Component 6: Decoder ─────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Reconstructs HR thermal image from fused features.
    Input:  (B, 64, H, W)
    Output: (B, 1, H, W)  values in [0, 1]
    """
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            ConvBNReLU(64, 32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()                              # output in [0, 1]
        )

    def forward(self, x):
        return self.decoder(x)


# ─── Component 7: Full Model ───────────────────────────────────────────────────

class AttentionThermalSR(nn.Module):
    """
    Full Optical-Guided Thermal Super-Resolution model with CBAM attention.

    Inputs:
        thermal_lr : (B, 1, H, W)  low-resolution thermal
        optical    : (B, 3, H, W)  high-resolution optical RGB

    Output:
        thermal_sr : (B, 1, H, W)  super-resolved thermal
    """
    def __init__(self):
        super().__init__()
        self.thermal_encoder  = ThermalEncoder()
        self.optical_encoder  = OpticalEncoder()
        self.channel_attention = ChannelAttention(channels=64, reduction=8)
        self.spatial_attention = SpatialAttention()
        self.decoder          = Decoder()

    def forward(self, thermal_lr, optical):
        # Encode both inputs into feature space
        f_thermal = self.thermal_encoder(thermal_lr)    # (B, 64, H, W)
        f_optical = self.optical_encoder(optical)       # (B, 64, H, W)

        # Apply CBAM attention to optical features
        f_optical = self.channel_attention(f_optical)   # (B, 64, H, W)
        f_optical = self.spatial_attention(f_optical)   # (B, 64, H, W)

        # Additive fusion: thermal features + attended optical features
        f_fused = f_thermal + f_optical                 # (B, 64, H, W)

        # Decode fused features to HR thermal image
        thermal_sr = self.decoder(f_fused)              # (B, 1, H, W)

        return thermal_sr


# ─── Quick Verification ────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model] Using device: {device}")

    model = AttentionThermalSR().to(device)

    # Dummy tensors matching your dataset shape
    thermal_lr = torch.randn(8, 1, 264, 264).to(device)
    optical    = torch.randn(8, 3, 264, 264).to(device)

    with torch.no_grad():
        output = model(thermal_lr, optical)

    print(f"\n[Verification]")
    print(f"  thermal_lr input shape : {thermal_lr.shape}")
    print(f"  optical    input shape : {optical.shape}")
    print(f"  output     shape       : {output.shape}")    # must be (8, 1, 264, 264)
    print(f"  output     range       : [{output.min():.3f}, {output.max():.3f}]")  # must be [0,1]

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  total parameters       : {total_params:,}")
    print(f"\n✓ Model ready for training")