import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from brainspy.processors.processor import Processor
from brainspy.processors.modules.conv import DNPUConv2d


def make_processor():
    ckpt = torch.load(Path("surrogate_model.pt"), map_location="cpu")

    processor = Processor(
        configs={
            "processor_type": "simulation",
            "waveform": {
                "plateau_length": 1,
                "slope_length": 0,
            },
        },
        info=ckpt["info"],
        model_state_dict=ckpt["model_state_dict"],
    )

    return processor


class DNPUConvCIFARAutoencoder(nn.Module):
    """
    Minimal CIFAR autoencoder.

    Input:
        grayscale CIFAR image in [0, 1], shape (B, 1, 32, 32)

    DNPU input:
        shifted to voltage range [-0.5, 0.5]

    Architecture:
        DNPUConv2d 1x32x32 -> conv_channels x16x16
        BatchNorm/ReLU
        digital downsample
        digital bottleneck
        digital decoder
    """

    def __init__(
        self,
        processor,
        conv_channels=8,
        latent_dim=64,
    ):
        super().__init__()

        self.dnpu_conv = DNPUConv2d(
            processor=processor,
            data_input_indices=[[0, 1, 2, 3]],
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            forward_pass_type="vec",
        )

        self.encoder = nn.Sequential(
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(conv_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.to_latent = nn.Linear(16 * 8 * 8, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 16 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        # CIFAR image [0,1] -> DNPU voltage range [-0.5,0.5]
        x_voltage = x - 0.5

        h = self.dnpu_conv(x_voltage)
        h = self.encoder(h)

        h_flat = h.flatten(start_dim=1)
        z = self.to_latent(h_flat)

        h_dec = self.from_latent(z)
        h_dec = h_dec.reshape(x.shape[0], 16, 8, 8)

        logits = self.decoder(h_dec)

        return logits, z


def reconstruction_loss(logits, x, loss_type):
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
def evaluate(model, loader, device, loss_type, max_batches=None):
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


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data_cifar")
    parser.add_argument("--results-dir", type=str, default="results_cifar")

    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--conv-channels", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=64)

    parser.add_argument(
        "--loss",
        choices=["bce", "mse", "l1", "bce_l1"],
        default="bce",
        help="Reconstruction loss used for optimization.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_full = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_full = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_subset = Subset(train_full, list(range(args.subset_size)))
    test_subset = Subset(test_full, list(range(min(args.subset_size, len(test_full)))))

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    processor = make_processor()

    model = DNPUConvCIFARAutoencoder(
        processor=processor,
        conv_channels=args.conv_channels,
        latent_dim=args.latent_dim,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("DNPUConv CIFAR autoencoder")
    print(f"  subset_size:      {args.subset_size}")
    print(f"  batch_size:       {args.batch_size}")
    print(f"  conv_channels:    {args.conv_channels}")
    print(f"  latent_dim:       {args.latent_dim}")
    print(f"  loss:             {args.loss}")
    print(f"  total params:     {total_params}")
    print(f"  trainable params: {trainable_params}")
    print(f"  device:           {device}")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        running_pixels = 0

        for x, _ in train_loader:
            x = x.to(device)

            logits, _ = model(x)
            loss = reconstruction_loss(logits, x, args.loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.numel()
            running_pixels += x.numel()

        train_loss = running_loss / running_pixels

        test_metrics = evaluate(
            model,
            test_loader,
            device,
            loss_type=args.loss,
            max_batches=20,
        )

        print(
            f"epoch {epoch:4d} | "
            f"train_loss {train_loss:.6f} | "
            f"test_loss {test_metrics['loss_per_pixel']:.6f} | "
            f"test_bce {test_metrics['bce_per_pixel']:.6f} | "
            f"test_mse {test_metrics['mse_per_pixel']:.6f} | "
            f"test_mae {test_metrics['mae_per_pixel']:.6f}"
        )

        if epoch == 1 or epoch == args.epochs:
            path = save_reconstruction_sample(
                model=model,
                loader=test_loader,
                device=device,
                save_path=results_dir / f"cifar_recon_epoch{epoch:04d}.png",
                n=8,
            )
            print("  saved:", path)

    checkpoint_path = results_dir / "cifar_dnpuconv_autoencoder.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
        },
        checkpoint_path,
    )

    print("\nSaved checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
