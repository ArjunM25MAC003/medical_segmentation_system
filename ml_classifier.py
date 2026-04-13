from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.feature_extractor import extract_features_from_image, extract_features_from_path


CLASS_LABELS = {
    "normal": 0,
    "benign": 1,
    "malignant": 1,
}


def collect_dataset(image_root: str | Path) -> tuple[pd.DataFrame, np.ndarray, list[Path]]:
    """Build a tabular feature dataset from the BUSI image folders."""
    root = Path(image_root)
    feature_rows: list[dict[str, float]] = []
    labels: list[int] = []
    image_paths: list[Path] = []

    for class_name, label in CLASS_LABELS.items():
        class_dir = root / class_name
        if not class_dir.exists():
            continue

        for image_path in sorted(class_dir.glob("*.png")):
            if "_mask" in image_path.stem:
                continue

            feature_dict = extract_features_from_path(image_path)
            feature_rows.append(feature_dict)
            labels.append(label)
            image_paths.append(image_path)

    if not feature_rows:
        raise ValueError(f"No training images found under: {root}")

    feature_frame = pd.DataFrame(feature_rows).fillna(0.0)
    return feature_frame, np.array(labels, dtype=np.int64), image_paths


def build_classifier(random_state: int = 42) -> Pipeline:
    """Create a simple but production-friendly classical ML pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=1,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_classifier(
    image_root: str | Path = "data/raw/busi",
    model_output_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    """Train the anomaly-vs-normal classifier and save it to disk."""
    features, labels, image_paths = collect_dataset(image_root)

    x_train, x_test, y_train, y_test, train_paths, test_paths = train_test_split(
        features,
        labels,
        image_paths,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    model = build_classifier(random_state=random_state)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["normal", "anomaly"],
            zero_division=0,
        ),
        "num_train_samples": int(len(x_train)),
        "num_test_samples": int(len(x_test)),
        "feature_names": list(features.columns),
        "test_image_paths": [str(path) for path in test_paths],
        "test_probabilities": probabilities.tolist(),
    }

    if model_output_path is not None:
        output_path = Path(model_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "feature_names": list(features.columns),
                "class_mapping": {"normal": 0, "anomaly": 1},
            },
            output_path,
        )

    metrics["model"] = model
    metrics["class_mapping"] = {"normal": 0, "anomaly": 1}

    return metrics


def load_trained_model(model_path: str | Path) -> dict[str, object]:
    """Load a previously saved classifier package."""
    return joblib.load(model_path)


def predict_image(
    image_path: str | Path,
    model_path: str | Path | None = None,
    model_package: dict[str, object] | None = None,
) -> dict[str, float | int | str]:
    """Run anomaly-vs-normal prediction for a single image."""
    if model_package is None:
        if model_path is None:
            raise ValueError("Provide either `model_path` or an in-memory `model_package`.")
        model_package = load_trained_model(model_path)

    model: Pipeline = model_package["model"]
    feature_names: list[str] = model_package["feature_names"]

    feature_dict = extract_features_from_path(image_path)
    feature_frame = pd.DataFrame([feature_dict]).reindex(columns=feature_names, fill_value=0.0)

    predicted_label = int(model.predict(feature_frame)[0])
    anomaly_probability = float(model.predict_proba(feature_frame)[0, 1])

    return {
        "predicted_label": predicted_label,
        "predicted_class": "anomaly" if predicted_label == 1 else "normal",
        "anomaly_probability": anomaly_probability,
    }


def predict_image_array(
    image: np.ndarray,
    model_path: str | Path | None = None,
    model_package: dict[str, object] | None = None,
) -> dict[str, float | int | str]:
    """Run anomaly-vs-normal prediction for an in-memory normalized image."""
    if model_package is None:
        if model_path is None:
            raise ValueError("Provide either `model_path` or an in-memory `model_package`.")
        model_package = load_trained_model(model_path)

    model: Pipeline = model_package["model"]
    feature_names: list[str] = model_package["feature_names"]

    feature_dict = extract_features_from_image(image)
    feature_frame = pd.DataFrame([feature_dict]).reindex(columns=feature_names, fill_value=0.0)

    predicted_label = int(model.predict(feature_frame)[0])
    anomaly_probability = float(model.predict_proba(feature_frame)[0, 1])

    return {
        "predicted_label": predicted_label,
        "predicted_class": "anomaly" if predicted_label == 1 else "normal",
        "anomaly_probability": anomaly_probability,
    }


if __name__ == "__main__":
    training_metrics = train_classifier()
    print(f"Accuracy: {training_metrics['accuracy']:.4f}")
    print(training_metrics["classification_report"])
