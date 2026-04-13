from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from dl_pipeline.model import UNet
from dl_pipeline.train import train_unet
from preprocessing.loader import load_and_preprocess_image


class UNetGradCAM:
    """Grad-CAM helper for a selected U-Net encoder block."""

    def __init__(self, model: UNet, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(
            _module: nn.Module,
            _grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Generate a Grad-CAM heatmap from a single image tensor."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(image_tensor)
        probabilities = torch.sigmoid(logits)
        target_score = probabilities.max()
        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy().astype(np.float32)
        cam -= cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam /= max_value

        return cam


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on top of the grayscale image."""
    base_image = np.clip(image, 0.0, 1.0)
    base_image_uint8 = (base_image * 255).astype(np.uint8)
    base_image_bgr = cv2.cvtColor(base_image_uint8, cv2.COLOR_GRAY2BGR)

    heatmap_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(base_image_bgr, 1.0 - alpha, colored_heatmap, alpha, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay


def generate_gradcam_for_image(
    model: UNet,
    image_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an image, generate a heatmap, and create an overlay visualization."""
    image = load_and_preprocess_image(image_path)
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)

    gradcam = UNetGradCAM(model, model.encoder3)
    heatmap = gradcam.generate(image_tensor)
    overlay = overlay_heatmap_on_image(image, heatmap)

    return image, heatmap, overlay


def visualize_gradcam(image: np.ndarray, heatmap: np.ndarray, overlay: np.ndarray) -> None:
    """Show original image, Grad-CAM heatmap, and overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Image")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")

    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def run_smoke_test(image_path: str | Path) -> None:
    """Small test snippet using a briefly trained U-Net."""
    results = train_unet(
        epochs=1,
        batch_size=2,
        max_samples=12,
        model_output_path=None,
        device="cpu",
    )
    model: UNet = results["model"]

    image, heatmap, overlay = generate_gradcam_for_image(model, image_path, device="cpu")

    print(f"Image shape: {image.shape}")
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: ({heatmap.min():.3f}, {heatmap.max():.3f})")
    print(f"Overlay shape: {overlay.shape}")

    visualize_gradcam(image, heatmap, overlay)


if __name__ == "__main__":
    sample_path = Path("data/raw/busi/benign/benign (1).png")
    run_smoke_test(sample_path)
