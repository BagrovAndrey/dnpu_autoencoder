import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from train_cifar_dnpu_stack_autoencoder import (
    make_processor,
    parse_channel_list,
    DNPUStackCIFARAutoencoder,
    reconstruction_loss,
)


class FixedRandomDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.from_latent = nn.Linear(latent_dim, 16 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        h = self.from_latent(z)
        h = h.reshape(z.shape[0], 16, 8, 8)
        logits = self.decoder(h)
        return logits


class DigitalEncoderFixedDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.to_latent = nn.Linear(16 * 8 * 8, latent_dim)
        self.fixed_decoder = FixedRandomDecoder(latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.to_latent(h.flatten(start_dim=1))
        logits = self.fixed_decoder(z)
        return logits, z


def reset_module_parameters(module, seed):
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    torch.random.set_rng_state(state)


def freeze_module(module):
    frozen_trainable = 0

    for p in module.parameters():
        if p.requires_grad:
            frozen_trainable += p.numel()
            p.requires_grad = False

    return frozen_trainable


def freeze_dnpu_decoder(model):
    frozen_trainable = 0

    for name, p in model.named_parameters():
        if name.startswith("from_latent") or name.startswith("decoder"):
            if p.requires_grad:
                frozen_trainable += p.numel()
                p.requires_grad = False

    return frozen_trainable


def reinitialize_dnpu_decoder(model, decoder_seed):
    state = torch.random.get_rng_state()
    torch.manual_seed(decoder_seed)

    for m in model.from_latent.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    for m in model.decoder.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    torch.random.set_rng_state(state)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


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
def oracle_batch_metrics(decoder, z, x):
    decoder.eval()

    logits = decoder(z)
    recon = torch.sigmoid(logits)

    bce = F.binary_cross_entropy_with_logits(
        logits, x, reduction="sum"
    ).item() / x.numel()

    mse = F.mse_loss(recon, x, reduction="sum").item() / x.numel()
    mae = F.l1_loss(recon, x, reduction="sum").item() / x.numel()

    return bce, mse, mae


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


def run_trainable_encoder(model, train_loader, test_loader, args, device, results_dir):
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    initial_metrics = evaluate(
        model,
        test_loader,
        device=device,
        loss_type=args.loss,
        max_batches=args.eval_batches,
    )

    print(
        f"epoch {0:4d} | "
        f"train_loss {'nan':>8} | "
        f"test_loss {initial_metrics['loss_per_pixel']:.6f} | "
        f"test_bce {initial_metrics['bce_per_pixel']:.6f} | "
        f"test_mse {initial_metrics['mse_per_pixel']:.6f} | "
        f"test_mae {initial_metrics['mae_per_pixel']:.6f}",
        flush=True,
    )

    path = save_reconstruction_sample(
        model=model,
        loader=test_loader,
        device=device,
        save_path=results_dir / "recon_epoch0000.png",
        n=8,
    )
    print("  saved:", path, flush=True)

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
            device=device,
            loss_type=args.loss,
            max_batches=args.eval_batches,
        )

        print(
            f"epoch {epoch:4d} | "
            f"train_loss {train_loss:.6f} | "
            f"test_loss {test_metrics['loss_per_pixel']:.6f} | "
            f"test_bce {test_metrics['bce_per_pixel']:.6f} | "
            f"test_mse {test_metrics['mse_per_pixel']:.6f} | "
            f"test_mae {test_metrics['mae_per_pixel']:.6f}",
            flush=True,
        )

        if epoch == 1 or epoch == args.epochs:
            path = save_reconstruction_sample(
                model=model,
                loader=test_loader,
                device=device,
                save_path=results_dir / f"recon_epoch{epoch:04d}.png",
                n=8,
            )
            print("  saved:", path, flush=True)


def run_frozen_random(model, test_loader, args, device, results_dir):
    metrics = evaluate(
        model,
        test_loader,
        device=device,
        loss_type=args.loss,
        max_batches=args.eval_batches,
    )

    print(
        f"frozen_random | "
        f"test_loss {metrics['loss_per_pixel']:.6f} | "
        f"test_bce {metrics['bce_per_pixel']:.6f} | "
        f"test_mse {metrics['mse_per_pixel']:.6f} | "
        f"test_mae {metrics['mae_per_pixel']:.6f}",
        flush=True,
    )

    path = save_reconstruction_sample(
        model=model,
        loader=test_loader,
        device=device,
        save_path=results_dir / "recon_frozen_random.png",
        n=8,
    )
    print("  saved:", path, flush=True)


def run_oracle_z(test_loader, args, device, results_dir):
    decoder = FixedRandomDecoder(args.latent_dim).to(device)
    reset_module_parameters(decoder, args.decoder_seed)
    freeze_module(decoder)

    decoder.eval()

    total_bce = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_pixels = 0

    saved_sample = False

    for batch_idx, (x, _) in enumerate(test_loader):
        if args.oracle_batches is not None and batch_idx >= args.oracle_batches:
            break

        x = x.to(device)

        z = torch.randn(
            x.shape[0],
            args.latent_dim,
            device=device,
            requires_grad=True,
        )

        optimizer = torch.optim.Adam([z], lr=args.oracle_lr)

        bce0, mse0, mae0 = oracle_batch_metrics(decoder, z, x)

        print(
            f"oracle batch {batch_idx + 1:4d} | "
            f"step {0:5d} | "
            f"bce {bce0:.6f} | "
            f"mse {mse0:.6f} | "
            f"mae {mae0:.6f}",
            flush=True,
        )

        for step in range(1, args.oracle_steps + 1):
            logits = decoder(z)
            loss = reconstruction_loss(logits, x, args.loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            should_log = (
                step == args.oracle_steps
                or (
                    args.oracle_log_every > 0
                    and step % args.oracle_log_every == 0
                )
            )

            if should_log:
                bce_step, mse_step, mae_step = oracle_batch_metrics(decoder, z, x)

                print(
                    f"oracle batch {batch_idx + 1:4d} | "
                    f"step {step:5d} | "
                    f"bce {bce_step:.6f} | "
                    f"mse {mse_step:.6f} | "
                    f"mae {mae_step:.6f}",
                    flush=True,
                )

        with torch.no_grad():
            logits = decoder(z)
            recon = torch.sigmoid(logits)

            bce = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
            mse = F.mse_loss(recon, x, reduction="sum")
            mae = F.l1_loss(recon, x, reduction="sum")

            total_bce += bce.item()
            total_mse += mse.item()
            total_mae += mae.item()
            total_pixels += x.numel()

            if not saved_sample:
                n = min(8, x.shape[0])
                panel = torch.cat([x[:n].cpu(), recon[:n].cpu()], dim=0)
                grid = make_grid(panel, nrow=n, padding=2)
                path = results_dir / "recon_oracle_z.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                save_image(grid, path)
                print("  saved:", path, flush=True)
                saved_sample = True

    print(
        f"oracle_z | "
        f"test_bce {total_bce / total_pixels:.6f} | "
        f"test_mse {total_mse / total_pixels:.6f} | "
        f"test_mae {total_mae / total_pixels:.6f}",
        flush=True,
    )


def build_dnpu_model(args, device):
    dnpu_channels = parse_channel_list(args.dnpu_channels)

    processor = make_processor()

    model = DNPUStackCIFARAutoencoder(
        processor=processor,
        encoder_type=args.encoder_type,
        dnpu_channels=dnpu_channels,
        latent_mode=args.latent_mode,
        latent_dim=args.latent_dim,
    ).to(device)

    reinitialize_dnpu_decoder(model, args.decoder_seed)
    frozen_decoder_params = freeze_dnpu_decoder(model)

    return model, dnpu_channels, frozen_decoder_params


def build_digital_model(args, device):
    model = DigitalEncoderFixedDecoder(latent_dim=args.latent_dim).to(device)

    reset_module_parameters(model.fixed_decoder, args.decoder_seed)
    frozen_decoder_params = freeze_module(model.fixed_decoder)

    return model, frozen_decoder_params


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "frozen_random",
            "train_dnpu_encoder",
            "train_digital_encoder",
            "oracle_z",
        ],
        required=True,
    )

    parser.add_argument("--data-dir", type=str, default="data_cifar")
    parser.add_argument("--results-dir", type=str, default="results_fixed_decoder")

    parser.add_argument("--subset-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument(
        "--encoder-type",
        choices=["hybrid", "dnpu"],
        default="dnpu",
        help="Used for DNPU modes only.",
    )
    parser.add_argument(
        "--dnpu-channels",
        type=str,
        default="16,1",
        help="Used for DNPU modes only. Example: 16,1 or 8,4,4.",
    )
    parser.add_argument(
        "--latent-mode",
        choices=["raw", "linear"],
        default="raw",
        help="Used for DNPU modes only.",
    )
    parser.add_argument("--latent-dim", type=int, default=64)

    parser.add_argument(
        "--loss",
        choices=["bce", "mse", "l1", "bce_l1"],
        default="l1",
    )

    parser.add_argument("--oracle-steps", type=int, default=500)
    parser.add_argument("--oracle-lr", type=float, default=1e-2)
    parser.add_argument("--oracle-batches", type=int, default=20)
    parser.add_argument(
        "--oracle-log-every",
        type=int,
        default=50,
        help="Print oracle metrics every N latent-optimization steps. Use 0 for start/end only.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decoder-seed", type=int, default=12345)
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
    test_subset = Subset(test_full, list(range(min(args.test_size, len(test_full)))))

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

    print("Fixed random decoder CIFAR hierarchy", flush=True)
    print(f"  mode:              {args.mode}", flush=True)
    print(f"  subset_size:       {args.subset_size}", flush=True)
    print(f"  test_size:         {args.test_size}", flush=True)
    print(f"  batch_size:        {args.batch_size}", flush=True)
    print(f"  loss:              {args.loss}", flush=True)
    print(f"  seed:              {args.seed}", flush=True)
    print(f"  decoder_seed:      {args.decoder_seed}", flush=True)
    print(f"  device:            {device}", flush=True)

    if args.mode in ["frozen_random", "train_dnpu_encoder"]:
        model, dnpu_channels, frozen_decoder_params = build_dnpu_model(args, device)

        if args.mode == "frozen_random":
            frozen_all_params = freeze_module(model)
        else:
            frozen_all_params = 0

        total_params, trainable_params = count_parameters(model)

        print(f"  encoder_type:      {args.encoder_type}", flush=True)
        print(f"  dnpu_channels:     {dnpu_channels}", flush=True)
        print(f"  latent_mode:       {args.latent_mode}", flush=True)
        print(f"  latent_dim:        {model.latent_dim}", flush=True)
        print(f"  raw_latent_dim:    {model.raw_latent_dim}", flush=True)
        print(f"  frozen decoder:    {frozen_decoder_params}", flush=True)
        print(f"  frozen all:        {frozen_all_params}", flush=True)
        print(f"  total params:      {total_params}", flush=True)
        print(f"  trainable params:  {trainable_params}", flush=True)
        print(flush=True)

        if args.mode == "frozen_random":
            run_frozen_random(model, test_loader, args, device, results_dir)
        else:
            run_trainable_encoder(model, train_loader, test_loader, args, device, results_dir)

    elif args.mode == "train_digital_encoder":
        model, frozen_decoder_params = build_digital_model(args, device)

        total_params, trainable_params = count_parameters(model)

        print(f"  digital latent_dim:{args.latent_dim}", flush=True)
        print(f"  frozen decoder:    {frozen_decoder_params}", flush=True)
        print(f"  total params:      {total_params}", flush=True)
        print(f"  trainable params:  {trainable_params}", flush=True)
        print(flush=True)

        run_trainable_encoder(model, train_loader, test_loader, args, device, results_dir)

    elif args.mode == "oracle_z":
        print(f"  oracle latent_dim: {args.latent_dim}", flush=True)
        print(f"  oracle_steps:      {args.oracle_steps}", flush=True)
        print(f"  oracle_lr:         {args.oracle_lr}", flush=True)
        print(f"  oracle_batches:    {args.oracle_batches}", flush=True)
        print(f"  oracle_log_every:  {args.oracle_log_every}", flush=True)
        print(flush=True)

        run_oracle_z(test_loader, args, device, results_dir)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
