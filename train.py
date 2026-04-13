from __future__ import annotations

from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dl_pipeline.dataset import MedicalSegmentationDataset, collect_segmentation_pairs
from dl_pipeline.loss import DiceLoss
from dl_pipeline.model import UNet


def build_dataloaders(
    image_root: str | Path = "data/raw/busi",
    batch_size: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    pairs = collect_segmentation_pairs(image_root)
    image_paths = [image_path for image_path, _ in pairs]

    if max_samples is not None:
        image_paths = image_paths[:max_samples]

    train_paths, val_paths = train_test_split(
        image_paths,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    train_dataset = MedicalSegmentationDataset(train_paths)
    val_dataset = MedicalSegmentationDataset(val_paths)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: UNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DiceLoss,
    device: torch.device,
) -> float:
    """Run one training epoch."""
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())

    return running_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate_one_epoch(
    model: UNet,
    dataloader: DataLoader,
    loss_fn: DiceLoss,
    device: torch.device,
) -> float:
    """Run one validation epoch."""
    model.eval()
    running_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)
        running_loss += float(loss.item())

    return running_loss / max(len(dataloader), 1)


def save_model(model: UNet, model_path: str | Path) -> None:
    """Save the trained model state dict."""
    torch.save(model.state_dict(), str(model_path))


def train_unet(
    image_root: str | Path = "data/raw/busi",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    model_output_path: str | Path | None = "dl_pipeline/unet_model.pt",
    device: str | None = None,
    max_samples: int | None = None,
) -> dict[str, object]:
    """Train a U-Net model for binary mask prediction."""
    chosen_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_loader, val_loader = build_dataloaders(
        image_root=image_root,
        batch_size=batch_size,
        max_samples=max_samples,
    )

    model = UNet().to(chosen_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = DiceLoss()

    history = {"train_loss": [], "val_loss": []}
    save_error: str | None = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, chosen_device)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, chosen_device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

    if model_output_path is not None:
        try:
            save_model(model, model_output_path)
        except (OSError, RuntimeError) as error:
            save_error = str(error)

    return {
        "model": model,
        "history": history,
        "device": str(chosen_device),
        "num_train_batches": len(train_loader),
        "num_val_batches": len(val_loader),
        "model_output_path": str(model_output_path) if model_output_path is not None else None,
        "save_error": save_error,
    }


if __name__ == "__main__":
    results = train_unet(epochs=1, batch_size=2, max_samples=12)
    print(results["history"])
