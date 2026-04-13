from __future__ import annotations

from pathlib import Path

from classical_pipeline.segmentation import watershed_segmentation
from features.fourier_descriptors import extract_fourier_descriptors
from features.shape_features import extract_shape_features
from features.texture_features import extract_glcm_features
from preprocessing.enhancement import apply_clahe
from preprocessing.loader import load_and_preprocess_image


def extract_all_features(image, mask) -> dict[str, float]:
    """Combine shape, texture, and Fourier descriptors in one feature vector."""
    features: dict[str, float] = {}
    features.update(extract_shape_features(mask))
    features.update(extract_glcm_features(image, mask))
    features.update(extract_fourier_descriptors(mask))
    return features


def extract_features_from_image(image) -> dict[str, float]:
    """Run the classical preprocessing path starting from an in-memory image."""
    enhanced = apply_clahe(image)
    mask, _ = watershed_segmentation(enhanced)
    return extract_all_features(enhanced, mask)


def extract_features_from_path(image_path: str | Path) -> dict[str, float]:
    """Run the classical preprocessing path and return a feature dictionary."""
    image = load_and_preprocess_image(image_path)
    return extract_features_from_image(image)


if __name__ == "__main__":
    sample_path = Path("data/raw/busi/benign/benign (1).png")
    feature_dict = extract_features_from_path(sample_path)
    for name, value in feature_dict.items():
        print(f"{name}: {value:.4f}")
