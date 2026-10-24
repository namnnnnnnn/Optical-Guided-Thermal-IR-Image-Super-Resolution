"""
app.py
Role: Gradio web UI for AttentionThermalSR inference.
      Upload LR thermal + optical image → get SR thermal output.
Deploy: Hugging Face Spaces (free) or run locally.
"""

import os
import sys
import numpy as np
import torch
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import AttentionThermalSR
from config import CHECKPOINT_PATH as CKPT, SAMPLES_DIR


# ─── Config ───────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = CKPT
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Load Model Once at Startup ────────────────────────────────────────────────

if not os.path.exists(CHECKPOINT_PATH):
    sys.exit(f"[Demo] Checkpoint not found: {CHECKPOINT_PATH}\n"
             f"       Place best_model_1000.pth in checkpoints/ or set THERMAL_SR_CHECKPOINT.")

print(f"[Demo] Loading model on {DEVICE}...")
model = AttentionThermalSR().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()
print("[Demo] Model loaded. Ready for inference.")


# ─── Helper Functions ──────────────────────────────────────────────────────────

def preprocess_thermal(pil_image):
    """
    Convert grayscale thermal PIL image to model input tensor.
    Output: (1, 1, H, W) float32 tensor normalized to [0,1]
    """
    arr = np.array(pil_image.convert('L')).astype(np.float32)  # (H, W)
    arr = arr / (arr.max() + 1e-8)                              # normalize [0,1]
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)        # (1,1,H,W)
    return tensor


def preprocess_optical(pil_image):
    """
    Convert RGB optical PIL image to model input tensor.
    Output: (1, 3, H, W) float32 tensor normalized to [0,1]
    """
    arr = np.array(pil_image.convert('RGB')).astype(np.float32)  # (H,W,3)
    arr = arr / (arr.max() + 1e-8)                                # normalize [0,1]
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)     # (1,3,H,W)
    return tensor


def postprocess(output_tensor):
    """
    Convert model output tensor to colored heatmap PIL image.
    """
    arr = output_tensor.squeeze().cpu().detach().numpy()  # (H, W)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)  # normalize

    # Apply 'hot' colormap → converts grayscale to orange/red heatmap
    colormap = plt.get_cmap('hot')
    colored = colormap(arr)                          # (H, W, 4) RGBA
    colored = (colored[:, :, :3] * 255).astype(np.uint8)  # (H, W, 3) RGB
    return Image.fromarray(colored, mode='RGB')


def resize_to_match(thermal_pil, optical_pil):
    """Resize optical to match thermal size if different."""
    if thermal_pil.size != optical_pil.size:
        optical_pil = optical_pil.resize(thermal_pil.size, Image.BICUBIC)
    return optical_pil


# ─── Inference Function ────────────────────────────────────────────────────────

def run_super_resolution(thermal_lr_image, optical_image):
    """
    Main inference function called by Gradio.
    Inputs: PIL images from Gradio upload
    Output: SR thermal PIL image + comparison figure
    """
    if thermal_lr_image is None or optical_image is None:
        return None, None

    # Resize optical to match thermal if needed
    optical_image = resize_to_match(thermal_lr_image, optical_image)

    # Preprocess
    thermal_tensor = preprocess_thermal(thermal_lr_image).to(DEVICE)
    optical_tensor = preprocess_optical(optical_image).to(DEVICE)

    # Inference
    with torch.no_grad():
        sr_tensor = model(thermal_tensor, optical_tensor)

    # Postprocess
    sr_image = postprocess(sr_tensor)

    # Generate comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(np.array(thermal_lr_image.convert('L')), cmap='hot')
    axes[0].set_title("LR Thermal (Input)")
    axes[0].axis('off')

    axes[1].imshow(np.array(optical_image.convert('RGB')))
    axes[1].set_title("Optical RGB (Guide)")
    axes[1].axis('off')

    axes[2].imshow(np.array(sr_image), cmap='hot')
    axes[2].set_title("SR Thermal (Output)")
    axes[2].axis('off')

    plt.tight_layout()

    return sr_image, fig


# ─── Gradio Interface ──────────────────────────────────────────────────────────

with gr.Blocks(title="Thermal IR Super-Resolution") as demo:

    gr.Markdown("""
    # 🌡️ Optical-Guided Thermal IR Super-Resolution
    **Upload a low-resolution thermal image and its paired optical (RGB) image
    to get a super-resolved thermal output.**

    - Model: AttentionThermalSR (CBAM Attention)
    - Dataset: Landsat-8 Satellite Imagery (SSL4EO)
    - Test-set metrics (3,750 scenes): PSNR 43.56 dB | SSIM 0.9911 | RMSE 0.0074
    """)

    with gr.Row():
        with gr.Column():
            thermal_input = gr.Image(
                type="pil",
                label="Low-Resolution Thermal Image (grayscale)"
            )
            optical_input = gr.Image(
                type="pil",
                label="Optical RGB Image (guide)"
            )
            run_btn = gr.Button("🚀 Run Super-Resolution", variant="primary")

        with gr.Column():
            sr_output = gr.Image(
                type="pil",
                label="Super-Resolved Thermal Output"
            )
            comparison_plot = gr.Plot(
                label="LR | Optical | SR Comparison"
            )

    # Example pairs exported with src/export_samples.py (shown only if present)
    if os.path.isdir(SAMPLES_DIR):
        examples = []
        for name in sorted(os.listdir(SAMPLES_DIR)):
            if name.endswith("_thermal_lr.png"):
                stem = name[:-len("_thermal_lr.png")]
                opt = os.path.join(SAMPLES_DIR, f"{stem}_optical.png")
                if os.path.exists(opt):
                    examples.append([os.path.join(SAMPLES_DIR, name), opt])
        if examples:
            gr.Examples(examples=examples, inputs=[thermal_input, optical_input],
                        label="Sample scenes (click to load)")

    run_btn.click(
        fn=run_super_resolution,
        inputs=[thermal_input, optical_input],
        outputs=[sr_output, comparison_plot]
    )

    gr.Markdown("""
    ---
    **Project:** Optical-Guided Thermal IR Super-Resolution | Final Year Major Project  
    **Architecture:** Dual CNN Encoder + CBAM Attention + Additive Fusion + Decoder  
    **Training:** 25 epochs on a 1,000-scene subset (700 train / 150 val) | ~58K parameters
    """)


# ─── Launch ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Set THERMAL_SR_SHARE=1 to get a temporary public gradio.live URL
    demo.launch(share=os.environ.get("THERMAL_SR_SHARE") == "1")