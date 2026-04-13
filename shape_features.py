from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from classical_pipeline.segmentation import watershed_segmentation
from preprocessing.enhancement import apply_clahe
from preprocessing.loader import load_and_preprocess_image


def get_largest_contour(mask: np.ndarray) -> np.ndarray | None:
    """Return the dominant contour from a binary mask."""
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def extract_shape_features(mask: np.ndarray) -> dict[str, float]:
    """Compute a few basic geometric properties from the mask."""
    contour = get_largest_contour(mask)

    if contour is None:
        return {"area": 0.0, "perimeter": 0.0, "circularity": 0.0}

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))
    circularity = 0.0

    if perimeter > 0:
        circularity = float((4.0 * np.pi * area) / (perimeter**2))

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
    }


if __name__ == "__main__":
    sample_path = Path("data/raw/busi/benign/benign (1).png")
    image = load_and_preprocess_image(sample_path)
    enhanced = apply_clahe(image)
    mask, _ = watershed_segmentation(enhanced)
    print(extract_shape_features(mask))
