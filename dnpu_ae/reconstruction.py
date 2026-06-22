"""Losses, metrics, and sample images for CIFAR reconstruction."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image


def reconstruction_loss(logits, x, loss_type):
    """Compute the selected optimization loss with the original semantics."""
    recon = torch.sigmoid(logits)

    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits, x)
    if loss_type == "mse":
        return F.mse_loss(recon, x)
    if loss_type == "l1":
        return F.l1_loss(recon, x)
    if loss_type == "bce_l1":
        return F.binary_cross_entropy_with_logits(logits, x) + F.l1_loss(recon, x)
    raise ValueError(f"Unknown loss type: {loss_type}")


@torch.no_grad()
def evaluate_reconstruction(model, loader, device, loss_type, max_batches=None):
    """Return reconstruction metrics normalized by the number of pixels."""
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_pixels = 0

    for batch_idx, (x, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        logits, _ = model(x)
        recon = torch.sigmoid(logits)

        loss = reconstruction_loss(logits, x, loss_type)
        bce = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
        mse = F.mse_loss(recon, x, reduction="sum")
        mae = F.l1_loss(recon, x, reduction="sum")

        total_loss += loss.item() * x.numel()
        total_bce += bce.item()
        total_mse += mse.item()
        total_mae += mae.item()
        total_pixels += x.numel()

    return {
        "loss_per_pixel": total_loss / total_pixels,
        "bce_per_pixel": total_bce / total_pixels,
        "mse_per_pixel": total_mse / total_pixels,
        "mae_per_pixel": total_mae / total_pixels,
    }


@torch.no_grad()
def save_reconstruction_sample(model, loader, device, save_path, n=8):
    """Save a grid containing inputs followed by their reconstructions."""
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    logits, _ = model(x)
    recon = torch.sigmoid(logits)

    panel = torch.cat([x.cpu(), recon.cpu()], dim=0)
    grid = make_grid(panel, nrow=n, padding=2)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, save_path)
    return save_path

