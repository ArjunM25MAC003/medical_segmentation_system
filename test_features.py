from pathlib import Path

import numpy as np

from classical_pipeline.segmentation import watershed_segmentation
from features.feature_extractor import extract_all_features
from preprocessing.enhancement import apply_clahe
from preprocessing.loader import load_and_preprocess_image


def main() -> None:
    sample_path = Path("data/raw/busi/benign/benign (1).png")
    image = load_and_preprocess_image(sample_path)
    enhanced = apply_clahe(image)
    mask, regions = watershed_segmentation(enhanced)
    features = extract_all_features(enhanced, mask)
    region_count = int(np.sum(np.unique(regions) > 0))

    print(f"Sample: {sample_path.name}")
    print(f"Mask foreground pixels: {int(mask.sum())}")
    print(f"Watershed regions: {region_count}")
    print("Extracted features:")

    for name, value in features.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
