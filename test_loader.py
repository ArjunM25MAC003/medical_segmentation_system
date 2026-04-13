from pathlib import Path

from preprocessing.loader import load_and_preprocess_image, visualize_image


def main() -> None:
    sample_paths = [
        Path("data/raw/busi/benign/benign (1).png"),
        Path("data/raw/busi/malignant/malignant (1).png"),
        Path("data/raw/busi/normal/normal (1).png"),
    ]

    for image_path in sample_paths:
        processed = load_and_preprocess_image(image_path)
        print(
            f"{image_path.name}: shape={processed.shape}, "
            f"dtype={processed.dtype}, min={processed.min():.3f}, max={processed.max():.3f}"
        )
        visualize_image(processed, title=f"Preview: {image_path.name}")


if __name__ == "__main__":
    main()
