from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

try:
    import pydicom
except ImportError:  # pragma: no cover - optional dependency at runtime
    pydicom = None

from classical_pipeline.ml_classifier import predict_image_array, train_classifier
from classical_pipeline.segmentation import watershed_segmentation
from dl_pipeline.gradcam import UNetGradCAM, overlay_heatmap_on_image
from dl_pipeline.model import UNet
from dl_pipeline.train import train_unet
from evaluation.metrics import evaluate_segmentation
from preprocessing.enhancement import apply_clahe
from preprocessing.loader import ensure_grayscale, normalize_intensity, resize_image
from reports.report_generator import build_diagnosis_summary, generate_pdf_report


app = FastAPI(
    title="Medical Segmentation API",
    description="FastAPI service for classical and U-Net medical image segmentation.",
    version="1.0.0",
)


def decode_uploaded_image(file_bytes: bytes, filename: str) -> np.ndarray:
    """Decode PNG/JPG/DICOM bytes into a normalized grayscale image."""
    suffix = Path(filename).suffix.lower()

    if suffix in {".dcm", ".dicom"}:
        if pydicom is None:
            raise HTTPException(status_code=400, detail="DICOM upload requires `pydicom`.")

        dataset = pydicom.dcmread(BytesIO(file_bytes))
        image = dataset.pixel_array.astype(np.float32)
    else:
        buffer = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode uploaded image.")
        image = image.astype(np.float32)

    image = ensure_grayscale(image)
    image = resize_image(image, size=(256, 256))
    image = normalize_intensity(image)
    return image


def encode_image_base64(image: np.ndarray, color: bool = False) -> str:
    """Encode a numpy image as a base64 PNG."""
    if color:
        image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".png", image_bgr)
    else:
        image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        success, encoded = cv2.imencode(".png", image_uint8)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image output.")

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


@lru_cache(maxsize=1)
def get_dl_model() -> UNet:
    """Load or train a small U-Net model for API inference."""
    results = train_unet(
        epochs=1,
        batch_size=2,
        max_samples=16,
        model_output_path=None,
        device="cpu",
    )
    model: UNet = results["model"]
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_classifier_package() -> dict[str, object]:
    """Train and cache the classical anomaly classifier for analysis output."""
    return train_classifier(model_output_path=None)


def run_classical_pipeline(image: np.ndarray) -> dict[str, np.ndarray]:
    """Run classical segmentation and create a simple heatmap view."""
    enhanced = apply_clahe(image)
    segmentation_mask, _ = watershed_segmentation(enhanced)
    heatmap = segmentation_mask.astype(np.float32)
    overlay = overlay_heatmap_on_image(image, heatmap)

    return {
        "image": image,
        "segmentation_mask": segmentation_mask.astype(np.float32),
        "heatmap": heatmap,
        "overlay": overlay.astype(np.float32) / 255.0 if overlay.dtype == np.uint8 else overlay,
    }


def run_deep_learning_pipeline(image: np.ndarray) -> dict[str, np.ndarray]:
    """Run U-Net inference and Grad-CAM on an in-memory image."""
    model = get_dl_model()
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
        segmentation_mask = (probabilities >= 0.5).float().squeeze().cpu().numpy()

    gradcam = UNetGradCAM(model, model.encoder3)
    heatmap = gradcam.generate(image_tensor)
    overlay = overlay_heatmap_on_image(image, heatmap)

    return {
        "image": image,
        "segmentation_mask": segmentation_mask.astype(np.float32),
        "heatmap": heatmap.astype(np.float32),
        "overlay": overlay.astype(np.float32) / 255.0 if overlay.dtype == np.uint8 else overlay,
    }


def build_report_summary(metrics: dict[str, float] | None) -> str:
    """Create a report summary that stays honest when metrics are unavailable."""
    if metrics:
        return build_diagnosis_summary(metrics)

    return (
        "No reference mask was provided, so quantitative segmentation metrics were not computed. "
        "Please review the visual outputs before using the result for downstream decisions."
    )


def build_analysis_result(image: np.ndarray) -> dict[str, str | float | int]:
    """Return a user-facing tumor/anomaly prediction for the input image."""
    prediction = predict_image_array(image, model_package=get_classifier_package())
    tumor_detected = "Yes" if prediction["predicted_label"] == 1 else "No"
    diagnosis_label = "Tumor/Anomaly Detected" if tumor_detected == "Yes" else "No Tumor Detected"

    return {
        **prediction,
        "tumor_detected": tumor_detected,
        "diagnosis_label": diagnosis_label,
    }


def build_report(
    image: np.ndarray,
    segmentation_mask: np.ndarray,
    heatmap: np.ndarray,
    metrics: dict[str, float] | None,
    analysis_result: dict[str, str | float | int],
) -> str:
    """Generate a PDF report and return its path."""
    report_path = Path("reports") / f"api_report_{uuid4().hex[:8]}.pdf"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    generate_pdf_report(
        output_path=report_path,
        original_image=image,
        segmentation_mask=segmentation_mask,
        gradcam_heatmap=heatmap,
        metrics=metrics or {},
        diagnosis_summary=(
            f"{analysis_result['diagnosis_label']}. {build_report_summary(metrics)}"
        ),
        analysis_result=analysis_result,
        title="Medical Segmentation API Report",
    )
    return str(report_path)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple health endpoint."""
    return {"status": "ok"}


@app.get("/reports/{report_name}")
def download_report(report_name: str) -> FileResponse:
    """Download a previously generated report."""
    report_path = Path("reports") / report_name
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(report_path, media_type="application/pdf", filename=report_path.name)


@app.post("/segment")
async def segment_image(
    image: UploadFile = File(...),
    pipeline: str = Form(...),
    reference_mask: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    """Upload an image, choose a pipeline, and receive segmentation outputs."""
    pipeline = pipeline.lower().strip()
    if pipeline not in {"classical", "deep", "deep_learning"}:
        raise HTTPException(status_code=400, detail="Pipeline must be `classical` or `deep`.")

    image_bytes = await image.read()
    processed_image = decode_uploaded_image(image_bytes, image.filename or "uploaded_image.png")

    if pipeline == "classical":
        outputs = run_classical_pipeline(processed_image)
    else:
        outputs = run_deep_learning_pipeline(processed_image)

    analysis_result = build_analysis_result(processed_image)

    metrics: dict[str, float] | None = None
    if reference_mask is not None:
        mask_bytes = await reference_mask.read()
        ground_truth_mask = decode_uploaded_image(
            mask_bytes, reference_mask.filename or "reference_mask.png"
        )
        metrics = evaluate_segmentation(ground_truth_mask, outputs["segmentation_mask"])

    report_path = build_report(
        image=outputs["image"],
        segmentation_mask=outputs["segmentation_mask"],
        heatmap=outputs["heatmap"],
        metrics=metrics,
        analysis_result=analysis_result,
    )

    return {
        "pipeline": pipeline,
        "analysis": analysis_result,
        "metrics": metrics,
        "segmentation": encode_image_base64(outputs["segmentation_mask"]),
        "heatmap": encode_image_base64(outputs["heatmap"]),
        "overlay": encode_image_base64(outputs["overlay"], color=True),
        "report_path": report_path,
        "report_download_url": f"/reports/{Path(report_path).name}",
        "note": (
            "Metrics are returned only when a reference mask is uploaded."
            if metrics is None
            else "Metrics computed against the uploaded reference mask."
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
