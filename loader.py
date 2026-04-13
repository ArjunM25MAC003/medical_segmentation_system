from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    import pydicom
except ImportError:  # pragma: no cover - optional dependency at runtime
    pydicom = None


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}


def validate_image_path(image_path: str | Path) -> Path:
    """Check that the image exists and has a supported extension."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return path


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Scale image intensities to the [0, 1] range."""
    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())

    if max_value == min_value:
        return np.zeros_like(image, dtype=np.float32)

    return (image - min_value) / (max_value - min_value)


def load_dicom_image(image_path: str | Path) -> np.ndarray:
    """Load a DICOM image and return a float32 pixel array."""
    if pydicom is None:
        raise ImportError(
            "pydicom is required to load DICOM files. Install it with `pip install pydicom`."
        )

    dicom_data = pydicom.dcmread(str(image_path))
    image = dicom_data.pixel_array.astype(np.float32)
    return image


def load_standard_image(image_path: str | Path, grayscale: bool) -> np.ndarray:
    """Load PNG/JPG images with OpenCV."""
    read_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    image = cv2.imread(str(image_path), read_flag)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image.astype(np.float32)


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR style images to grayscale when needed."""
    if image.ndim == 2:
        return image

    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_image(image: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize image to the target spatial size."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def load_and_preprocess_image(
    image_path: str | Path,
    size: Tuple[int, int] = (256, 256),
    grayscale: bool = True,
) -> np.ndarray:
    """
    Load an image, convert to grayscale if requested, resize it, and normalize intensities.
    """
    path = validate_image_path(image_path)

    if path.suffix.lower() in {".dcm", ".dicom"}:
        image = load_dicom_image(path)
    else:
        image = load_standard_image(path, grayscale=grayscale)

    if grayscale:
        image = ensure_grayscale(image)

    image = resize_image(image, size=size)
    image = normalize_intensity(image)
    return image


def visualize_image(
    image: np.ndarray,
    title: str = "Preprocessed Image",
    cmap: str = "gray",
) -> None:
    """Display a single image using matplotlib."""
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sample_path = Path("data/raw/busi/benign/benign (1).png")
    processed_image = load_and_preprocess_image(sample_path)
    print(
        f"Loaded image with shape={processed_image.shape}, "
        f"dtype={processed_image.dtype}, range=({processed_image.min():.3f}, {processed_image.max():.3f})"
    )
    visualize_image(processed_image, title="Loader Smoke Test")
