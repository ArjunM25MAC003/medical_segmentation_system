from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb

from classical_pipeline.morphology import apply_closing, apply_opening
from preprocessing.enhancement import apply_clahe
from preprocessing.loader import load_and_preprocess_image


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a normalized image into uint8 for OpenCV routines."""
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def otsu_threshold_segmentation(image: np.ndarray) -> np.ndarray:
    """Generate a binary mask using Otsu thresholding."""
    image_uint8 = to_uint8(image)
    _, binary_mask = cv2.threshold(
        image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return (binary_mask > 0).astype(np.uint8)


def watershed_segmentation(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segment connected regions using watershed markers."""
    image_uint8 = to_uint8(image)
    otsu_mask = otsu_threshold_segmentation(image)
    cleaned_mask = apply_closing(apply_opening(otsu_mask, kernel_size=3), kernel_size=3)
    mask_uint8 = (cleaned_mask * 255).astype(np.uint8)

    distance_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    _, sure_foreground = cv2.threshold(
        distance_transform, 0.4 * distance_transform.max(), 255, 0
    )
    sure_foreground = sure_foreground.astype(np.uint8)

    sure_background = cv2.dilate(mask_uint8, np.ones((3, 3), dtype=np.uint8), iterations=2)
    unknown = cv2.subtract(sure_background, sure_foreground)

    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0

    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    watershed_markers = cv2.watershed(image_bgr, markers.copy())

    separated_regions = np.zeros_like(watershed_markers, dtype=np.int32)
    separated_regions[watershed_markers > 1] = watershed_markers[watershed_markers > 1] - 1

    return cleaned_mask.astype(np.uint8), separated_regions


def visualize_segmentation_results(
    image: np.ndarray,
    binary_mask: np.ndarray,
    separated_regions: np.ndarray,
) -> None:
    """Show original image, binary mask, and separated watershed regions."""
    overlay = label2rgb(
        separated_regions, image=np.clip(image, 0.0, 1.0), bg_label=0, alpha=0.4
    )
    overlay = np.clip(overlay, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Image")
    axes[1].imshow(binary_mask, cmap="gray")
    axes[1].set_title("Otsu Binary Mask")
    axes[2].imshow(overlay)
    axes[2].set_title("Watershed Regions")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def run_smoke_test(image_path: str | Path) -> None:
    """Simple module-level test snippet for quick verification."""
    image = load_and_preprocess_image(image_path)
    enhanced_image = apply_clahe(image)

    binary_mask = otsu_threshold_segmentation(enhanced_image)
    refined_mask, separated_regions = watershed_segmentation(enhanced_image)

    unique_regions = np.unique(separated_regions)
    region_count = int(np.sum(unique_regions > 0))

    print(f"Otsu foreground pixels: {int(binary_mask.sum())}")
    print(f"Watershed mask foreground pixels: {int(refined_mask.sum())}")
    print(f"Separated regions: {region_count}")

    visualize_segmentation_results(enhanced_image, refined_mask, separated_regions)


if __name__ == "__main__":
    sample_path = Path("data/raw/busi/benign/benign (1).png")
    run_smoke_test(sample_path)
