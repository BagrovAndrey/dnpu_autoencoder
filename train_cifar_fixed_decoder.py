"""Fixed-decoder hierarchy experiments and oracle-z sanity check.

The oracle optimizes latent vectors directly to measure the ceiling imposed by
the frozen random digital decoder, independently of encoder quality.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from dnpu_ae.cifar_data import make_grayscale_cifar_loaders
from dnpu_ae.cifar_models import (
    DigitalEncoderFixedDecoder,
    DNPUStackCIFARAutoencoder,
    FixedRandomDecoder,
)
from dnpu_ae.model_utils import (
    count_parameters,
    freeze_dnpu_decoder,
    freeze_module,
    parse_channel_list,
    reinitialize_dnpu_decoder,
    reset_module_parameters,
)
from dnpu_ae.processor import make_processor
from dnpu_ae.reconstruction import (
    evaluate_reconstruction as evaluate,
    reconstruction_loss,
    save_reconstruction_sample,
)


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

    train_loader, test_loader = make_grayscale_cifar_loaders(
        data_dir=args.data_dir,
        train_size=args.subset_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        train_shuffle=True,
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
