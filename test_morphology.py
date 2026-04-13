from pathlib import Path

from classical_pipeline.morphology import run_smoke_test


if __name__ == "__main__":
    sample_path = Path("data/raw/busi/benign/benign (1).png")
    run_smoke_test(sample_path)
