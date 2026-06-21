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


def parse_channel_list(s):
    channels = [int(x.strip()) for x in s.split(",") if x.strip()]

    if len(channels) == 0:
        raise ValueError("--dnpu-channels must contain at least one integer")

    if any(c <= 0 for c in channels):
        raise ValueError("--dnpu-channels must contain only positive integers")

    return channels


class DNPUStackCIFARAutoencoder(nn.Module):
    """
    CIFAR grayscale autoencoder with either a hybrid encoder or a universal
    DNPUConv stack encoder.

    Input:
        grayscale CIFAR image in [0, 1], shape (B, 1, 32, 32)

    Encoder modes:
        hybrid:
            DNPUConv2d 1 x 32 x 32 -> C1 x 16 x 16
            BatchNorm/ReLU
            digital Conv2d C1 x 16 x 16 -> 16 x 8 x 8
            BatchNorm/ReLU

        dnpu:
            DNPUConv stack with stride-2 layers specified by --dnpu-channels.
            Example --dnpu-channels 16,8,4:
                1 x 32 x 32
                -> 16 x 16 x 16
                -> 8 x 8 x 8
                -> 4 x 4 x 4

    Latent modes:
        raw:
            z = flatten(encoder_output)

        linear:
            z_raw = flatten(encoder_output)
            z = Linear(z_raw_dim -> latent_dim)

    Decoder:
        Linear(z_dim -> 16*8*8)
        ConvTranspose2d decoder to 1 x 32 x 32
    """

    def __init__(
        self,
        processor,
        encoder_type="dnpu",
        dnpu_channels=None,
        latent_mode="raw",
        latent_dim=64,
    ):
        super().__init__()

        if dnpu_channels is None:
            dnpu_channels = [16, 1]

        if encoder_type not in ["hybrid", "dnpu"]:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        if latent_mode not in ["raw", "linear"]:
            raise ValueError(f"Unknown latent_mode: {latent_mode}")

        if len(dnpu_channels) > 5:
            raise ValueError(
                "Too many DNPU layers for 32x32 input with kernel=2, stride=2. "
                "Maximum is 5: 32 -> 16 -> 8 -> 4 -> 2 -> 1."
            )

        self.encoder_type = encoder_type
        self.dnpu_channels = list(dnpu_channels)
        self.latent_mode = latent_mode
        self.requested_latent_dim = latent_dim

        self.dnpu_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        in_channels = 1
        spatial_size = 32

        if encoder_type == "hybrid":
            # One DNPUConv layer, followed by a digital downsampling layer.
            c1 = dnpu_channels[0]

            self.dnpu_layers.append(
                DNPUConv2d(
                    processor=processor,
                    data_input_indices=[[0, 1, 2, 3]],
                    in_channels=1,
                    out_channels=c1,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    forward_pass_type="vec",
                )
            )
            self.norm_layers.append(nn.BatchNorm2d(c1))

            self.encoder2 = nn.Sequential(
                nn.Conv2d(c1, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )

            self.raw_channels = 16
            self.raw_spatial_size = 8

        else:
            # Fully DNPUConv encoder stack.
            for out_channels in dnpu_channels:
                self.dnpu_layers.append(
                    DNPUConv2d(
                        processor=processor,
                        data_input_indices=[[0, 1, 2, 3]],
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        forward_pass_type="vec",
                    )
                )

                self.norm_layers.append(nn.BatchNorm2d(out_channels))

                in_channels = out_channels
                spatial_size = spatial_size // 2

            self.raw_channels = in_channels
            self.raw_spatial_size = spatial_size

        self.raw_latent_dim = (
            self.raw_channels * self.raw_spatial_size * self.raw_spatial_size
        )

        if latent_mode == "raw":
            self.to_latent = nn.Identity()
            self.latent_dim = self.raw_latent_dim
        else:
            self.to_latent = nn.Linear(self.raw_latent_dim, latent_dim)
            self.latent_dim = latent_dim

        self.from_latent = nn.Linear(self.latent_dim, 16 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode_features(self, x):
        # CIFAR image [0,1] -> DNPU voltage range [-0.5,0.5]
        h = x - 0.5

        if self.encoder_type == "hybrid":
            h = self.dnpu_layers[0](h)
            h = self.norm_layers[0](h)
            h = F.relu(h)

            h = self.encoder2(h)
            return h

        for dnpu_layer, norm_layer in zip(self.dnpu_layers, self.norm_layers):
            h = dnpu_layer(h)
            h = norm_layer(h)
            h = F.relu(h)

        return h

    def forward(self, x):
        h = self.encode_features(x)

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


def freeze_dnpu_parameters(model):
    frozen_trainable = 0

    for name, param in model.named_parameters():
        if "dnpu_layers" in name and param.requires_grad:
            frozen_trainable += param.numel()
            param.requires_grad = False

    return frozen_trainable


def freeze_batchnorm_parameters(model):
    frozen_trainable = 0

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                if param.requires_grad:
                    frozen_trainable += param.numel()
                    param.requires_grad = False

    return frozen_trainable


def freeze_encoder_parameters(model):
    frozen_trainable = 0

    encoder_prefixes = (
        "dnpu_layers",
        "norm_layers",
        "encoder2",
        "to_latent",
    )

    for name, param in model.named_parameters():
        if name.startswith(encoder_prefixes) and param.requires_grad:
            frozen_trainable += param.numel()
            param.requires_grad = False

    return frozen_trainable


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

    parser.add_argument(
        "--encoder-type",
        choices=["hybrid", "dnpu"],
        default="dnpu",
        help="hybrid = DNPUConv + digital Conv2d; dnpu = arbitrary DNPUConv stack.",
    )
    parser.add_argument(
        "--dnpu-channels",
        type=str,
        default="16,1",
        help="Comma-separated DNPUConv output channels, e.g. 16,1 or 16,8,4.",
    )

    parser.add_argument(
        "--latent-mode",
        choices=["raw", "linear"],
        default="raw",
        help="raw = use flattened encoder output as latent; linear = apply Linear(raw_dim -> latent_dim).",
    )
    parser.add_argument("--latent-dim", type=int, default=64)

    parser.add_argument(
        "--loss",
        choices=["bce", "mse", "l1", "bce_l1"],
        default="l1",
        help="Reconstruction loss used for optimization.",
    )

    parser.add_argument(
        "--freeze-dnpu",
        action="store_true",
        help="Freeze DNPUConv trainable control voltages.",
    )
    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        help="Freeze BatchNorm affine parameters while keeping DNPU controls trainable.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the full encoder up to the latent representation. Decoder remains trainable.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.freeze_encoder and (args.freeze_dnpu or args.freeze_bn):
        raise ValueError(
            "Use --freeze-encoder alone, not together with --freeze-dnpu or --freeze-bn."
        )

    if args.freeze_dnpu and args.freeze_bn:
        raise ValueError("Use either --freeze-dnpu or --freeze-bn, not both.")

    dnpu_channels = parse_channel_list(args.dnpu_channels)

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

    model = DNPUStackCIFARAutoencoder(
        processor=processor,
        encoder_type=args.encoder_type,
        dnpu_channels=dnpu_channels,
        latent_mode=args.latent_mode,
        latent_dim=args.latent_dim,
    ).to(device)

    frozen_dnpu_params = 0
    frozen_bn_params = 0
    frozen_encoder_params = 0

    if args.freeze_dnpu:
        frozen_dnpu_params = freeze_dnpu_parameters(model)

    if args.freeze_bn:
        frozen_bn_params = freeze_batchnorm_parameters(model)

    if args.freeze_encoder:
        frozen_encoder_params = freeze_encoder_parameters(model)

    total_params, trainable_params = count_parameters(model)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("DNPU stack CIFAR autoencoder")
    print(f"  subset_size:       {args.subset_size}")
    print(f"  batch_size:        {args.batch_size}")
    print(f"  encoder_type:      {args.encoder_type}")
    print(f"  dnpu_channels:     {dnpu_channels}")
    print(f"  raw_shape:         {model.raw_channels} x {model.raw_spatial_size} x {model.raw_spatial_size}")
    print(f"  latent_mode:       {args.latent_mode}")
    print(f"  latent_dim:        {model.latent_dim}")
    print(f"  raw_latent_dim:    {model.raw_latent_dim}")
    print(f"  loss:              {args.loss}")
    print(f"  freeze_dnpu:       {args.freeze_dnpu}")
    print(f"  frozen DNPU pars:  {frozen_dnpu_params}")
    print(f"  freeze_bn:         {args.freeze_bn}")
    print(f"  frozen BN pars:    {frozen_bn_params}")
    print(f"  freeze_encoder:    {args.freeze_encoder}")
    print(f"  frozen enc pars:   {frozen_encoder_params}")
    print(f"  total params:      {total_params}")
    print(f"  trainable params:  {trainable_params}")
    print(f"  device:            {device}")
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

    checkpoint_path = results_dir / "cifar_dnpu_stack_autoencoder.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "dnpu_channels": dnpu_channels,
            "raw_channels": model.raw_channels,
            "raw_spatial_size": model.raw_spatial_size,
            "raw_latent_dim": model.raw_latent_dim,
            "latent_dim": model.latent_dim,
        },
        checkpoint_path,
    )

    print("\nSaved checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
