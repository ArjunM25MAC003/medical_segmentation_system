from pathlib import Path

from api.main import build_analysis_result, decode_uploaded_image, run_classical_pipeline
from evaluation.metrics import evaluate_segmentation


def main() -> None:
    image_path = Path("data/raw/busi/benign/benign (1).png")
    mask_path = Path("data/raw/busi/benign/benign (1)_mask.png")

    image = decode_uploaded_image(image_path.read_bytes(), image_path.name)
    reference_mask = decode_uploaded_image(mask_path.read_bytes(), mask_path.name)
    outputs = run_classical_pipeline(image)
    analysis = build_analysis_result(image)
    metrics = evaluate_segmentation(reference_mask, outputs["segmentation_mask"])

    print(f"Image shape: {image.shape}")
    print(f"Segmentation shape: {outputs['segmentation_mask'].shape}")
    print(f"Overlay shape: {outputs['overlay'].shape}")
    print(f"Analysis: {analysis}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
