from __future__ import annotations

from pathlib import Path

import numpy as np

from classical_pipeline.segmentation import watershed_segmentation
from features.shape_features import get_largest_contour
from preprocessing.enhancement import apply_clahe
from preprocessing.loader import load_and_preprocess_image


def extract_fourier_descriptors(mask: np.ndarray, num_descriptors: int = 10) -> dict[str, float]:
    """Describe contour shape using normalized Fourier coefficients."""
    contour = get_largest_contour(mask)

    if contour is None:
        return {f"fourier_descriptor_{index + 1}": 0.0 for index in range(num_descriptors)}

    contour_points = contour.squeeze(axis=1).astype(np.float32)
    if contour_points.ndim != 2 or contour_points.shape[0] < 3:
        return {f"fourier_descriptor_{index + 1}": 0.0 for index in range(num_descriptors)}

    complex_contour = contour_points[:, 0] + 1j * contour_points[:, 1]
    descriptors = np.fft.fft(complex_contour)
    descriptors = np.abs(descriptors[1 : num_descriptors + 1])

    if descriptors.size == 0:
        return {f"fourier_descriptor_{index + 1}": 0.0 for index in range(num_descriptors)}

    normalization_term = descriptors[0] if descriptors[0] > 0 else 1.0
    descriptors = descriptors / normalization_term

    feature_dict: dict[str, float] = {}
    for index in range(num_descriptors):
        value = float(descriptors[index]) if index < len(descriptors) else 0.0
        feature_dict[f"fourier_descriptor_{index + 1}"] = value

    return feature_dict


if __name__ == "__main__":
    sample_path = Path("data/raw/busi/benign/benign (1).png")
    image = load_and_preprocess_image(sample_path)
    enhanced = apply_clahe(image)
    mask, _ = watershed_segmentation(enhanced)
    print(extract_fourier_descriptors(mask, num_descriptors=5))
