from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from classical_pipeline.segmentation import watershed_segmentation
from dl_pipeline.dataset import merge_mask_variants
from dl_pipeline.gradcam import generate_gradcam_for_image
from dl_pipeline.train import train_unet
from evaluation.metrics import evaluate_segmentation
from preprocessing.enhancement import apply_clahe
from preprocessing.loader import load_and_preprocess_image


def array_to_image_reader(image: np.ndarray) -> ImageReader:
    """Convert a numpy image into a ReportLab-friendly in-memory image."""
    image = np.asarray(image)

    if image.ndim == 2:
        image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        success, encoded = cv2.imencode(".png", image_uint8)
    elif image.ndim == 3:
        image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".png", image_bgr)
    else:
        raise ValueError(f"Unsupported image shape for report export: {image.shape}")

    if not success:
        raise ValueError("Failed to encode report image.")

    return ImageReader(BytesIO(encoded.tobytes()))


def build_diagnosis_summary(metrics: dict[str, float]) -> str:
    """Create a short human-readable diagnosis summary."""
    dice = metrics.get("dice_score", 0.0)
    iou = metrics.get("iou_score", 0.0)

    if dice >= 0.75 and iou >= 0.60:
        return "Segmentation quality looks strong. The detected region is well aligned with the reference mask."
    if dice >= 0.40 and iou >= 0.25:
        return "Segmentation quality is moderate. The detected region overlaps the reference mask but may need review."
    return "Segmentation quality is limited. The detected region should be reviewed carefully before clinical interpretation."


def generate_pdf_report(
    output_path: str | Path,
    original_image: np.ndarray,
    segmentation_mask: np.ndarray,
    gradcam_heatmap: np.ndarray,
    metrics: dict[str, float],
    diagnosis_summary: str,
    analysis_result: dict[str, str | float | int] | None = None,
    title: str = "Medical Image Segmentation Report",
) -> str:
    """Generate a PDF report with images, metrics, and summary text."""
    output_path = Path(output_path)
    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    page_width, page_height = A4

    pdf.setTitle(title)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, page_height - 40, title)

    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, page_height - 60, f"Generated for: {output_path.name}")

    analysis_y = page_height - 90
    if analysis_result is not None:
        predicted_class = str(analysis_result.get("predicted_class", "unknown")).replace("_", " ")
        tumor_detected = str(analysis_result.get("tumor_detected", "unknown"))
        confidence = analysis_result.get("anomaly_probability")

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, analysis_y, "Analysis")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(50, analysis_y - 18, f"Tumor Detected: {tumor_detected}")
        pdf.drawString(50, analysis_y - 34, f"Predicted Class: {predicted_class.title()}")
        if confidence is not None:
            pdf.drawString(50, analysis_y - 50, f"Anomaly Probability: {float(confidence):.4f}")

    image_width = 160
    image_height = 160
    y_top = page_height - 290 if analysis_result is not None else page_height - 250

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(40, y_top + 170, "Original Image")
    pdf.drawString(220, y_top + 170, "Segmentation Mask")
    pdf.drawString(400, y_top + 170, "Grad-CAM Heatmap")

    pdf.drawImage(array_to_image_reader(original_image), 40, y_top, width=image_width, height=image_height)
    pdf.drawImage(array_to_image_reader(segmentation_mask), 220, y_top, width=image_width, height=image_height)
    pdf.drawImage(array_to_image_reader(gradcam_heatmap), 400, y_top, width=image_width, height=image_height)

    metric_y = y_top - 40
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, metric_y, "Metrics")

    pdf.setFont("Helvetica", 10)
    line_y = metric_y - 20
    for metric_name, value in metrics.items():
        formatted_value = "inf" if np.isinf(value) else f"{value:.4f}"
        pdf.drawString(50, line_y, f"{metric_name}: {formatted_value}")
        line_y -= 16

    summary_y = line_y - 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, summary_y, "Diagnosis Summary")

    pdf.setFont("Helvetica", 10)
    text_object = pdf.beginText(50, summary_y - 20)
    for line in split_text(diagnosis_summary, max_chars=95):
        text_object.textLine(line)
    pdf.drawText(text_object)

    pdf.save()
    return str(output_path)


def split_text(text: str, max_chars: int = 90) -> list[str]:
    """Wrap text into short lines for the PDF canvas."""
    words = text.split()
    lines: list[str] = []
    current_line: list[str] = []

    for word in words:
        trial = " ".join(current_line + [word])
        if len(trial) <= max_chars:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def run_smoke_test(
    image_path: str | Path,
    output_path: str | Path = "reports/sample_report.pdf",
) -> None:
    """Create a report using the current classical and DL helper outputs."""
    image = load_and_preprocess_image(image_path)
    enhanced = apply_clahe(image)
    segmentation_mask, _ = watershed_segmentation(enhanced)
    ground_truth_mask = merge_mask_variants(image_path)
    metrics = evaluate_segmentation(ground_truth_mask, segmentation_mask)

    gradcam_model = train_unet(
        epochs=1,
        batch_size=2,
        max_samples=8,
        model_output_path=None,
        device="cpu",
    )["model"]
    _, gradcam_heatmap, _ = generate_gradcam_for_image(gradcam_model, image_path, device="cpu")

    diagnosis_summary = build_diagnosis_summary(metrics)

    report_path = generate_pdf_report(
        output_path=output_path,
        original_image=image,
        segmentation_mask=segmentation_mask,
        gradcam_heatmap=gradcam_heatmap,
        metrics=metrics,
        diagnosis_summary=diagnosis_summary,
        analysis_result=None,
    )

    print(f"Report path: {report_path}")
    print(f"Dice Score: {metrics['dice_score']:.4f}")
    print(f"IoU Score: {metrics['iou_score']:.4f}")
    print(f"PSNR: {metrics['psnr']:.4f}")
    print(f"Diagnosis summary: {diagnosis_summary}")


if __name__ == "__main__":
    sample_path = Path("data/raw/busi/benign/benign (1).png")
    run_smoke_test(sample_path)
