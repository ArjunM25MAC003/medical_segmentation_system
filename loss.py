from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.contiguous().view(probabilities.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice_score.mean()


if __name__ == "__main__":
    loss_fn = DiceLoss()
    logits = torch.randn(2, 1, 256, 256)
    masks = torch.randint(0, 2, (2, 1, 256, 256)).float()
    print(f"Dice loss: {loss_fn(logits, masks).item():.4f}")
