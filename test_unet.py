from pathlib import Path

import torch

from dl_pipeline.dataset import MedicalSegmentationDataset, collect_segmentation_pairs
from dl_pipeline.model import UNet
from dl_pipeline.train import train_unet


def main() -> None:
    pairs = collect_segmentation_pairs()
    dataset = MedicalSegmentationDataset([image_path for image_path, _ in pairs[:2]])
    sample = dataset[0]
    print(f"Sample image shape: {tuple(sample['image'].shape)}")
    print(f"Sample mask shape: {tuple(sample['mask'].shape)}")

    model = UNet()
    dummy_output = model(sample["image"].unsqueeze(0))
    print(f"Forward pass output shape: {tuple(dummy_output.shape)}")

    results = train_unet(
        epochs=1,
        batch_size=2,
        max_samples=12,
        model_output_path=Path("dl_pipeline/unet_smoke_test.pt"),
        device="cpu",
    )
    print(f"Training device: {results['device']}")
    print(f"Train batches: {results['num_train_batches']}")
    print(f"Validation batches: {results['num_val_batches']}")
    print(f"Final train loss: {results['history']['train_loss'][-1]:.4f}")
    print(f"Final val loss: {results['history']['val_loss'][-1]:.4f}")
    print(f"Model output path: {results['model_output_path']}")
    print(f"Save error: {results['save_error']}")
    print(f"Saved model exists: {Path('dl_pipeline/unet_smoke_test.pt').exists()}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
