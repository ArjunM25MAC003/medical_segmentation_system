from pathlib import Path

from classical_pipeline.ml_classifier import predict_image, train_classifier


def main() -> None:
    metrics = train_classifier()
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Train samples: {metrics['num_train_samples']}")
    print(f"Test samples: {metrics['num_test_samples']}")
    print("Confusion matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")

    sample_path = Path("data/raw/busi/normal/normal (1).png")
    prediction = predict_image(
        sample_path,
        model_package={
            "model": metrics["model"],
            "feature_names": metrics["feature_names"],
            "class_mapping": metrics["class_mapping"],
        },
    )
    print(f"Sample prediction for {sample_path.name}:")
    print(f"  class={prediction['predicted_class']}")
    print(f"  anomaly_probability={prediction['anomaly_probability']:.4f}")


if __name__ == "__main__":
    main()
