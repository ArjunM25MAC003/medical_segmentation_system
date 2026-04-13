from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.enhancement import apply_clahe
from preprocessing.loader import load_and_preprocess_image


def create_binary_mask(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a normalized image into a binary mask."""
    mask = (image >= threshold).astype(np.uint8)
    return mask


def get_kernel(kernel_size: int = 3) -> np.ndarray:
    """Create a square structuring element for morphology."""
    return np.ones((kernel_size, kernel_size), dtype=np.uint8)


def apply_erosion(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Shrink bright regions and remove tiny noise."""
    kernel = get_kernel(kernel_size)
    return cv2.erode(mask, kernel, iterations=iterations)


def apply_dilation(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Expand bright regions and fill small gaps."""
    kernel = get_kernel(kernel_size)
    return cv2.dilate(mask, kernel, iterations=iterations)


def apply_opening(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Remove small foreground speckles."""
    kernel = get_kernel(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def apply_closing(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Close small holes inside foreground regions."""
    kernel = get_kernel(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def visualize_morphology_operations(mask: np.ndarray) -> None:
    """Show the input mask and each morphology result side by side."""
    eroded = apply_erosion(mask)
    dilated = apply_dilation(mask)
    opened = apply_opening(mask)
    closed = apply_closing(mask)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    images = [mask, eroded, dilated, opened, closed]
    titles = ["Original Mask", "Erosion", "Dilation", "Opening", "Closing"]

    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def run_smoke_test(image_path: str | Path) -> None:
    """Simple module-level test snippet for quick verification."""
    image = load_and_preprocess_image(image_path)
    enhanced_image = apply_clahe(image)
    binary_mask = create_binary_mask(enhanced_image, threshold=0.55)

    eroded = apply_erosion(binary_mask)
    dilated = apply_dilation(binary_mask)
    opened = apply_opening(binary_mask)
    closed = apply_closing(binary_mask)

    print(f"Mask foreground pixels: {int(binary_mask.sum())}")
    print(f"Erosion foreground pixels: {int(eroded.sum())}")
    print(f"Dilation foreground pixels: {int(dilated.sum())}")
    print(f"Opening foreground pixels: {int(opened.sum())}")
    print(f"Closing foreground pixels: {int(closed.sum())}")

    visualize_morphology_operations(binary_mask)


if __name__ == "__main__":
    sample_path = Path("data/raw/busi/benign/benign (1).png")
    run_smoke_test(sample_path)
