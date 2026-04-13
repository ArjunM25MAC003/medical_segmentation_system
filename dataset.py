from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.loader import load_and_preprocess_image, normalize_intensity, validate_image_path


def load_mask(mask_path: str | Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Load a binary segmentation mask and keep edges crisp during resize."""
    path = validate_image_path(mask_path)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Failed to load mask: {path}")

    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.float32)
    return mask


def merge_mask_variants(image_path: str | Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Merge one or more BUSI mask files into a single binary target."""
    image_path = Path(image_path)
    mask_candidates = sorted(image_path.parent.glob(f"{image_path.stem}_mask*.png"))

    if not mask_candidates:
        return np.zeros(size, dtype=np.float32)

    merged_mask = np.zeros(size, dtype=np.float32)
    for mask_path in mask_candidates:
        merged_mask = np.maximum(merged_mask, load_mask(mask_path, size=size))

    return merged_mask


def collect_segmentation_pairs(
    image_root: str | Path = "data/raw/busi",
) -> list[tuple[Path, Path | None]]:
    """Collect image paths and optional paired mask paths."""
    root = Path(image_root)
    pairs: list[tuple[Path, Path | None]] = []

    for class_name in ("benign", "malignant", "normal"):
        class_dir = root / class_name
        if not class_dir.exists():
            continue

        for image_path in sorted(class_dir.glob("*.png")):
            if "_mask" in image_path.stem:
                continue

            primary_mask = class_dir / f"{image_path.stem}_mask.png"
            pairs.append((image_path, primary_mask if primary_mask.exists() else None))

    if not pairs:
        raise ValueError(f"No segmentation images found under: {root}")

    return pairs


class MedicalSegmentationDataset(Dataset):
    """Dataset for grayscale image and binary mask pairs."""

    def __init__(
        self,
        image_paths: list[str | Path],
        size: tuple[int, int] = (256, 256),
    ) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.size = size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        image = load_and_preprocess_image(image_path, size=self.size, grayscale=True)
        mask = merge_mask_variants(image_path, size=self.size)

        image = normalize_intensity(image).astype(np.float32)
        mask = mask.astype(np.float32)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
        }


if __name__ == "__main__":
    pairs = collect_segmentation_pairs()
    dataset = MedicalSegmentationDataset([image_path for image_path, _ in pairs[:2]])
    sample = dataset[0]
    print(sample["image"].shape, sample["mask"].shape, sample["image_path"])
