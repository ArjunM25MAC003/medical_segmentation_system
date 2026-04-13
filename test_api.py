from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


def main() -> None:
    client = TestClient(app)

    image_path = Path("data/raw/busi/benign/benign (1).png")
    mask_path = Path("data/raw/busi/benign/benign (1)_mask.png")

    with image_path.open("rb") as image_file, mask_path.open("rb") as mask_file:
        response = client.post(
            "/segment",
            data={"pipeline": "classical"},
            files={
                "image": (image_path.name, image_file, "image/png"),
                "reference_mask": (mask_path.name, mask_file, "image/png"),
            },
        )

    print(f"Status code: {response.status_code}")
    payload = response.json()
    print(f"Pipeline: {payload['pipeline']}")
    print(f"Analysis: {payload['analysis']}")
    print(f"Metrics keys: {list(payload['metrics'].keys())}")
    print(f"Has segmentation output: {bool(payload['segmentation'])}")
    print(f"Has heatmap output: {bool(payload['heatmap'])}")
    print(f"Report path: {payload['report_path']}")


if __name__ == "__main__":
    main()
