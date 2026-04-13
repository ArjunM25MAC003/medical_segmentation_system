# Medical Image Analysis & Tumor Segmentation System

Network link: "http://10.23.0.136:8501/"
Production-style end-to-end medical image analysis project built with:

- Python
- PyTorch
- OpenCV
- scikit-image
- scikit-learn
- FastAPI
- Streamlit

This system supports two segmentation paths:

- Classical computer vision pipeline
- Deep learning pipeline based on U-Net

It also includes:

- preprocessing utilities
- feature extraction
- anomaly classification
- Grad-CAM explainability
- evaluation metrics
- PDF report generation
- FastAPI inference API
- Streamlit UI

## Project Structure

```text
medical_segmentation_system/
├── api/
├── classical_pipeline/
├── data/
├── dl_pipeline/
├── evaluation/
├── features/
├── preprocessing/
├── reports/
├── ui/
├── test_api.py
├── test_enhancement.py
├── test_features.py
├── test_gradcam.py
├── test_loader.py
├── test_metrics.py
├── test_ml_classifier.py
├── test_morphology.py
├── test_report_generator.py
├── test_segmentation.py
├── test_ui.py
└── test_unet.py
```

## Implemented Modules

### 1. Preprocessing

- [`preprocessing/loader.py`](./preprocessing/loader.py)
  - PNG, JPG, JPEG, DICOM loading
  - grayscale conversion
  - resize to `256 x 256`
  - intensity normalization to `[0, 1]`
  - visualization helper

- [`preprocessing/enhancement.py`](./preprocessing/enhancement.py)
  - CLAHE
  - histogram equalization
  - FFT high-pass filtering
  - before/after visualization

### 2. Classical Pipeline

- [`classical_pipeline/morphology.py`](./classical_pipeline/morphology.py)
  - erosion
  - dilation
  - opening
  - closing

- [`classical_pipeline/segmentation.py`](./classical_pipeline/segmentation.py)
  - Otsu thresholding
  - watershed segmentation
  - binary mask generation
  - separated region labeling

- [`classical_pipeline/ml_classifier.py`](./classical_pipeline/ml_classifier.py)
  - feature table creation from BUSI dataset
  - anomaly vs normal classification
  - Random Forest training
  - single-image prediction

### 3. Feature Extraction

- [`features/shape_features.py`](./features/shape_features.py)
  - area
  - perimeter
  - circularity

- [`features/texture_features.py`](./features/texture_features.py)
  - GLCM contrast
  - GLCM energy
  - GLCM homogeneity

- [`features/fourier_descriptors.py`](./features/fourier_descriptors.py)
  - normalized Fourier contour descriptors

- [`features/feature_extractor.py`](./features/feature_extractor.py)
  - unified feature vector builder

### 4. Deep Learning Pipeline

- [`dl_pipeline/dataset.py`](./dl_pipeline/dataset.py)
  - BUSI image/mask dataset loader
  - support for merged mask variants like `_mask`, `_mask_1`

- [`dl_pipeline/model.py`](./dl_pipeline/model.py)
  - compact U-Net implementation

- [`dl_pipeline/loss.py`](./dl_pipeline/loss.py)
  - Dice loss

- [`dl_pipeline/train.py`](./dl_pipeline/train.py)
  - train/validation split
  - training loop
  - validation loop
  - optional model save support

- [`dl_pipeline/gradcam.py`](./dl_pipeline/gradcam.py)
  - Grad-CAM on U-Net encoder
  - heatmap generation
  - overlay on input image

### 5. Evaluation

- [`evaluation/metrics.py`](./evaluation/metrics.py)
  - Dice score
  - IoU
  - PSNR
  - comparison plotting

### 6. Reporting

- [`reports/report_generator.py`](./reports/report_generator.py)
  - PDF export using `reportlab`
  - original image
  - segmentation mask
  - Grad-CAM heatmap
  - metrics
  - diagnosis summary

### 7. Serving

- [`api/main.py`](./api/main.py)
  - FastAPI app
  - image upload
  - classical / deep pipeline switch
  - optional reference mask upload
  - segmentation, heatmap, overlay, metrics, report response

- [`ui/app.py`](./ui/app.py)
  - Streamlit app
  - upload image
  - choose pipeline
  - Grad-CAM toggle
  - side-by-side comparisons
  - report download

## Dataset

This project is currently wired against the BUSI ultrasound dataset layout already present in this repo:

```text
data/raw/busi/
├── benign/
├── malignant/
└── normal/
```

The code assumes image names like:

- `benign (1).png`
- `benign (1)_mask.png`
- `benign (1)_mask_1.png`

For classical ML classification:

- `normal` is mapped to class `0`
- `benign` and `malignant` are mapped to class `1` as anomaly

## Installation

Create and activate your environment, then install the required libraries.

```bash
pip install numpy pandas matplotlib opencv-python scikit-image scikit-learn torch torchvision fastapi uvicorn streamlit reportlab joblib pydicom python-multipart
```

## How To Run

### Run the Streamlit UI

```bash
streamlit run ui/app.py
```

### Run the FastAPI app

```bash
uvicorn api.main:app --reload
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- API health: `http://127.0.0.1:8000/health`

### Run individual smoke tests

```bash
python test_loader.py
python test_enhancement.py
python test_morphology.py
python test_segmentation.py
python test_features.py
python test_ml_classifier.py
python test_unet.py
python test_gradcam.py
python test_metrics.py
python test_report_generator.py
python test_api.py
python test_ui.py
```

## Example API Usage

`POST /segment`

Form fields:

- `image`: uploaded image file
- `pipeline`: `classical` or `deep`
- `reference_mask`: optional mask file for metrics

Response includes:

- `segmentation`
- `heatmap`
- `overlay`
- `metrics`
- `report_path`
- `report_download_url`

## Notes About Current Behavior

- The DL pipeline uses a small U-Net and a lightweight training pass for local smoke testing.
- Inference is functional, but the model is not fully trained for production-grade medical performance yet.
- The classical segmentation pipeline is intentionally simple and can show low overlap against ground-truth masks without further tuning.
- Metrics are only meaningful when a reference mask is provided.
- Some save operations were made optional or graceful because the current sandbox environment can block certain file writes during tests.

## Output Artifacts

Generated reports are saved under [`reports/`](./reports/), for example:

- [`reports/sample_report.pdf`](./reports/sample_report.pdf)
- [`reports/api_report_e00a4047.pdf`](./reports/api_report_e00a4047.pdf)

## Suggested Next Improvements

- add a pinned `requirements.txt`
- add proper training checkpoints for the U-Net
- add a dedicated inference module for loading saved DL weights
- improve classical segmentation with dataset-specific tuning
- add batch inference
- add Docker support
- add unit tests and CI

## License

Use this project for research, learning, and prototyping. For real clinical deployment, model validation, dataset governance, regulatory review, and medical safety checks are still required.
