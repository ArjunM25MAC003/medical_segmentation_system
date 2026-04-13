from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.loader import load_and_preprocess_image, normalize_intensity


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a normalized float image to uint8 for OpenCV transforms."""
    image = normalize_intensity(image)
    return (image * 255).astype(np.uint8)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Improve local contrast with CLAHE."""
    image_uint8 = to_uint8(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image_uint8)
    return normalize_intensity(enhanced)


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Stretch the global contrast using histogram equalization."""
    image_uint8 = to_uint8(image)
    enhanced = cv2.equalizeHist(image_uint8)
    return normalize_intensity(enhanced)


def apply_fft_high_pass_filter(image: np.ndarray, radius: int = 20) -> np.ndarray:
    """Highlight edges and fine structures using a simple FFT high-pass filter."""
    image = normalize_intensity(image)

    frequency = np.fft.fft2(image)
    shifted = np.fft.fftshift(frequency)

    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    mask = np.ones((rows, cols), dtype=np.float32)
    y, x = np.ogrid[:rows, :cols]
    low_freq_region = (x - center_col) ** 2 + (y - center_row) ** 2 <= radius**2
    mask[low_freq_region] = 0.0

    filtered_shifted = shifted * mask
    inverse_shifted = np.fft.ifftshift(filtered_shifted)
    reconstructed = np.fft.ifft2(inverse_shifted)
    magnitude = np.abs(reconstructed)

    return normalize_intensity(magnitude)


def visualize_enhancements(image: np.ndarray) -> None:
    """Show the original image next to all enhancement outputs."""
    clahe_image = apply_clahe(image)
    hist_eq_image = apply_histogram_equalization(image)
    fft_image = apply_fft_high_pass_filter(image)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    images = [image, clahe_image, hist_eq_image, fft_image]
    titles = ["Original", "CLAHE", "Histogram Equalization", "FFT High-Pass"]

    for axis, current_image, title in zip(axes, images, titles):
        axis.imshow(current_image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def run_smoke_test(image_path: str | Path) -> None:
    """Simple module-level test snippet for quick verification."""
    image = load_and_preprocess_image(image_path)

    clahe_image = apply_clahe(image)
    hist_eq_image = apply_histogram_equalization(image)
    fft_image = apply_fft_high_pass_filter(image)

    print(
        f"Original: shape={image.shape}, range=({image.min():.3f}, {image.max():.3f})"
    )
    print(
        f"CLAHE: shape={clahe_image.shape}, range=({clahe_image.min():.3f}, {clahe_image.max():.3f})"
    )
    print(
        f"HistEq: shape={hist_eq_image.shape}, range=({hist_eq_image.min():.3f}, {hist_eq_image.max():.3f})"
    )
    print(
        f"FFT HPF: shape={fft_image.shape}, range=({fft_image.min():.3f}, {fft_image.max():.3f})"
    )

    visualize_enhancements(image)


if __name__ == "__main__":
    sample_path = Path("data/raw/busi/benign/benign (1).png")
    run_smoke_test(sample_path)
