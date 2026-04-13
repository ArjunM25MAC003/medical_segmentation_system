from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from classical_pipeline.segmentation import watershed_segmentation
from dl_pipeline.dataset import merge_mask_variants
from preprocessing.enhancement import apply_clahe
from preprocessing.loader import load_and_preprocess_image


def binarize_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert masks to clean binary format."""
    return (mask >= threshold).astype(np.float32)


def dice_score(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1.0) -> float:
    """Compute Dice overlap between two binary masks."""
    y_true = binarize_mask(y_true).ravel()
    y_pred = binarize_mask(y_pred).ravel()

    intersection = float((y_true * y_pred).sum())
    denominator = float(y_true.sum() + y_pred.sum())
    return (2.0 * intersection + smooth) / (denominator + smooth)


def iou_score(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1.0) -> float:
    """Compute intersection-over-union for binary masks."""
    y_true = binarize_mask(y_true).ravel()
    y_pred = binarize_mask(y_pred).ravel()

    intersection = float((y_true * y_pred).sum())
    union = float(y_true.sum() + y_pred.sum() - intersection)
    return (intersection + smooth) / (union + smooth)


def psnr_score(y_true: np.ndarray, y_pred: np.ndarray, max_pixel: float = 1.0) -> float:
    """Compute PSNR between two normalized masks."""
    y_true = np.clip(y_true.astype(np.float32), 0.0, 1.0)
    y_pred = np.clip(y_pred.astype(np.float32), 0.0, 1.0)

    mse = float(np.mean((y_true - y_pred) ** 2))
    if mse == 0.0:
        return float("inf")

    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))


def evaluate_segmentation(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a compact dictionary of segmentation metrics."""
    return {
        "dice_score": dice_score(y_true, y_pred),
        "iou_score": iou_score(y_true, y_pred),
        "psnr": psnr_score(y_true, y_pred),
    }


def plot_segmentation_comparison(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, float] | None = None,
) -> None:
    """Show image, target mask, prediction, and a metrics bar chart."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Image")

    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground Truth Mask")

    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title("Predicted Mask")

    if metrics is None:
        metrics = evaluate_segmentation(ground_truth, prediction)

    metric_names = list(metrics.keys())
    metric_values = [
        0.0 if np.isinf(value) else float(value)
        for value in metrics.values()
    ]
    axes[3].bar(metric_names, metric_values, color=["#2f6bff", "#45a049", "#d87d2f"])
    axes[3].set_title("Metrics")
    axes[3].tick_params(axis="x", rotation=20)

    for axis in axes[:3]:
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def run_smoke_test(image_path: str | Path) -> None:
    """Small test snippet using classical segmentation against BUSI masks."""
    image = load_and_preprocess_image(image_path)
    enhanced = apply_clahe(image)
    predicted_mask, _ = watershed_segmentation(enhanced)
    ground_truth_mask = merge_mask_variants(image_path)

    metrics = evaluate_segmentation(ground_truth_mask, predicted_mask)

    print(f"Dice Score: {metrics['dice_score']:.4f}")
    print(f"IoU Score: {metrics['iou_score']:.4f}")
    print(f"PSNR: {metrics['psnr']:.4f}")

    plot_segmentation_comparison(image, ground_truth_mask, predicted_mask, metrics)


if __name__ == "__main__":
    sample_path = Path("data/raw/busi/benign/benign (1).png")
    run_smoke_test(sample_path)
