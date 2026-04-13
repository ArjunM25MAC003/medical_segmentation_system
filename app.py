from __future__ import annotations

from pathlib import Path
import sys
from uuid import uuid4

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import (
    build_analysis_result,
    build_report_summary,
    decode_uploaded_image,
    get_dl_model,
    run_classical_pipeline,
    run_deep_learning_pipeline,
)
from evaluation.metrics import evaluate_segmentation
from reports.report_generator import generate_pdf_report


st.set_page_config(
    page_title="Medical Tumor Segmentation System",
    page_icon=":material/biotech:",
    layout="wide",
)


@st.cache_resource
def load_cached_dl_model():
    """Keep the U-Net model warm across reruns."""
    return get_dl_model()


def generate_ui_report(
    image,
    segmentation_mask,
    heatmap,
    metrics,
    analysis_result,
) -> Path:
    """Generate a PDF report and return the saved path."""
    report_path = Path("reports") / f"ui_report_{uuid4().hex[:8]}.pdf"
    generate_pdf_report(
        output_path=report_path,
        original_image=image,
        segmentation_mask=segmentation_mask,
        gradcam_heatmap=heatmap,
        metrics=metrics or {},
        diagnosis_summary=f"{analysis_result['diagnosis_label']}. {build_report_summary(metrics)}",
        analysis_result=analysis_result,
        title="Medical Segmentation UI Report",
    )
    return report_path


def format_metrics(metrics: dict[str, float] | None) -> dict[str, str]:
    """Format metric values for the Streamlit metric widgets."""
    if not metrics:
        return {}

    formatted: dict[str, str] = {}
    for name, value in metrics.items():
        formatted[name] = "inf" if value == float("inf") else f"{value:.4f}"
    return formatted


def main() -> None:
    st.title("Medical Image Analysis & Tumor Segmentation")
    st.caption(
        "Upload a medical image, switch between the classical and U-Net pipelines, "
        "inspect the segmentation visually, and export a report."
    )

    with st.sidebar:
        st.header("Controls")
        pipeline_label = st.radio(
            "Pipeline",
            options=["Classical", "Deep Learning"],
            horizontal=False,
        )
        show_gradcam = st.toggle("Grad-CAM ON/OFF", value=True)
        uploaded_image = st.file_uploader(
            "Upload medical image",
            type=["png", "jpg", "jpeg", "dcm", "dicom"],
        )
        uploaded_mask = st.file_uploader(
            "Optional reference mask",
            type=["png", "jpg", "jpeg"],
        )

        if pipeline_label == "Deep Learning":
            st.caption("The first DL run may take longer while the cached U-Net model warms up.")

    if uploaded_image is None:
        st.info("Upload an image to start the analysis.")
        return

    image_bytes = uploaded_image.getvalue()
    image = decode_uploaded_image(image_bytes, uploaded_image.name)

    if pipeline_label == "Deep Learning":
        load_cached_dl_model()
        outputs = run_deep_learning_pipeline(image)
    else:
        outputs = run_classical_pipeline(image)

    analysis_result = build_analysis_result(image)

    metrics = None
    if uploaded_mask is not None:
        reference_mask = decode_uploaded_image(uploaded_mask.getvalue(), uploaded_mask.name)
        metrics = evaluate_segmentation(reference_mask, outputs["segmentation_mask"])

    st.subheader("Analysis")
    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
    analysis_col1.metric("Tumor Detected", str(analysis_result["tumor_detected"]))
    analysis_col2.metric("Predicted Class", str(analysis_result["predicted_class"]).title())
    analysis_col3.metric(
        "Anomaly Probability",
        f"{float(analysis_result['anomaly_probability']):.4f}",
    )
    st.caption(str(analysis_result["diagnosis_label"]))

    st.subheader("Visual Results")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input Image", clamp=True)
    with col2:
        st.image(outputs["segmentation_mask"], caption="Segmentation Output", clamp=True)

    st.subheader("Side-by-Side Comparison")
    compare_col1, compare_col2 = st.columns(2)
    with compare_col1:
        st.image(image, caption="Original", clamp=True)
    with compare_col2:
        if pipeline_label == "Deep Learning" and show_gradcam:
            st.image(outputs["overlay"], caption="Original + Grad-CAM Overlay", clamp=True)
        else:
            st.image(outputs["overlay"], caption="Pipeline Overlay", clamp=True)

    if pipeline_label == "Deep Learning":
        st.subheader("Explainability")
        if show_gradcam:
            heatmap_col1, heatmap_col2 = st.columns(2)
            with heatmap_col1:
                st.image(outputs["heatmap"], caption="Grad-CAM Heatmap", clamp=True)
            with heatmap_col2:
                st.image(outputs["overlay"], caption="Grad-CAM Overlay", clamp=True)
        else:
            st.caption("Grad-CAM is turned off.")

    formatted_metrics = format_metrics(metrics)
    st.subheader("Metrics")
    if formatted_metrics:
        metric_columns = st.columns(len(formatted_metrics))
        for column, (metric_name, metric_value) in zip(metric_columns, formatted_metrics.items()):
            column.metric(metric_name.replace("_", " ").title(), metric_value)
    else:
        st.caption("Upload a reference mask to compute Dice, IoU, and PSNR.")

    report_path = generate_ui_report(
        image=image,
        segmentation_mask=outputs["segmentation_mask"],
        heatmap=outputs["heatmap"],
        metrics=metrics,
        analysis_result=analysis_result,
    )

    st.subheader("Report")
    st.caption(f"{analysis_result['diagnosis_label']}. {build_report_summary(metrics)}")
    with report_path.open("rb") as report_file:
        st.download_button(
            label="Download Report",
            data=report_file.read(),
            file_name=report_path.name,
            mime="application/pdf",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
