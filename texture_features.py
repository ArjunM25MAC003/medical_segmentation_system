from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from classical_pipeline.segmentation import watershed_segmentation
from preprocessing.enhancement import apply_clahe
from preprocessing.loader import load_and_preprocess_image


def extract_glcm_features(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
) -> dict[str, float]:
    """Extract a compact set of GLCM texture statistics."""
    distances = distances or [1]
    angles = angles or [0.0]

    working_image = np.clip(image, 0.0, 1.0)
    image_uint8 = (working_image * 255).astype(np.uint8)

    if mask is not None:
        image_uint8 = image_uint8.copy()
        image_uint8[mask == 0] = 0

    glcm = graycomatrix(
        image_uint8,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    return {
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
    }


if __name__ == "__main__":
    sample_path = Path("data/raw/busi/benign/benign (1).png")
    image = load_and_preprocess_image(sample_path)
    enhanced = apply_clahe(image)
    mask, _ = watershed_segmentation(enhanced)
    print(extract_glcm_features(enhanced, mask))
