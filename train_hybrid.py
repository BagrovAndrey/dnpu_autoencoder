import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from brainspy.processors.processor import Processor

from data import make_bars_stripes, pixels_to_voltages
from models import PlanarDNPUEncoderDigitalDecoder
from visualize import plot_reconstructions


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

    voltage_ranges = ckpt["info"]["electrode_info"]["activation_electrodes"]["voltage_ranges"]
    return processor, voltage_ranges


def reconstruction_accuracy(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    pixel_acc = (pred == target).float().mean()
    pattern_acc = (pred == target).all(dim=1).float().mean()
    return pixel_acc.item(), pattern_acc.item()


def make_run_name(args):
    return (
        f"hybrid_"
        f"{args.image_size}x{args.image_size}_"
        f"ndata{args.n_data}_"
        f"nctrl{args.n_control}_"
        f"{args.group_type}_"
        f"seed{args.seed}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a planar DNPU encoder with a minimal digital decoder."
    )

    # Architecture knobs.
    parser.add_argument("--image-size", type=int, default=4)
    parser.add_argument("--n-data", type=int, default=4)
    parser.add_argument("--n-control", type=int, default=3)
    parser.add_argument(
        "--group-type",
        type=str,
        default="auto",
        choices=["auto", "2x2", "horizontal", "vertical"],
    )

    # Training knobs.
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    # Hardware-aware latent-output penalty.
    parser.add_argument("--lambda-z", type=float, default=1e-3)
    parser.add_argument("--z0", type=float, default=10.0)

    # Input voltage encoding.
    parser.add_argument("--v-low", type=float, default=-0.5)
    parser.add_argument("--v-high", type=float, default=0.5)

    # Output.
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.n_data + args.n_control != 7:
        raise ValueError("--n-data + --n-control must be 7.")

    torch.manual_seed(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    run_name = args.run_name if args.run_name is not None else make_run_name(args)

    x_pixels = make_bars_stripes(n=args.image_size)
    x_volt = pixels_to_voltages(x_pixels, low=args.v_low, high=args.v_high)

    processor, voltage_ranges = make_processor()

    model = PlanarDNPUEncoderDigitalDecoder(
        processor=processor,
        voltage_ranges=voltage_ranges,
        image_size=args.image_size,
        n_data=args.n_data,
        n_control=args.n_control,
        group_type=args.group_type,
        init="center",
        freeze_encoder=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Hybrid DNPU autoencoder training")
    print(f"  run_name:     {run_name}")
    print(f"  image_size:   {args.image_size}x{args.image_size}")
    print(f"  patterns:     {x_pixels.shape[0]}")
    print(f"  n_data:       {args.n_data}")
    print(f"  n_control:    {args.n_control}")
    print(f"  group_type:   {args.group_type}")
    print(f"  latent_dim:   {model.latent_dim}")
    print(f"  epochs:       {args.epochs}")
    print(f"  lr:           {args.lr}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  lambda_z:     {args.lambda_z}")
    print(f"  z0:           {args.z0}")
    print()

    for epoch in range(1, args.epochs + 1):
        logits, z = model(x_volt)

        loss_recon = F.binary_cross_entropy_with_logits(
            logits,
            x_pixels,
        )

        z_penalty = torch.relu(z.abs() - args.z0).pow(2).mean()
        loss = loss_recon + args.lambda_z * z_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clip_controls_()

        if epoch == 1 or epoch % 500 == 0:
            pixel_acc, pattern_acc = reconstruction_accuracy(logits, x_pixels)
            z_mean = z.detach().mean().item()
            z_std = z.detach().std().item()
            z_abs_max = z.detach().abs().max().item()

            print(
                f"epoch {epoch:5d} | "
                f"loss {loss.item():.6f} | "
                f"rec {loss_recon.item():.6f} | "
                f"z_pen {z_penalty.item():.6f} | "
                f"pixel_acc {pixel_acc:.3f} | "
                f"pattern_acc {pattern_acc:.3f} | "
                f"z_mean {z_mean:.3f} | "
                f"z_std {z_std:.3f} | "
                f"z_abs_max {z_abs_max:.3f}"
            )

    checkpoint_path = results_dir / f"{run_name}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "loss": loss.item(),
            "loss_recon": loss_recon.item(),
            "z_penalty": z_penalty.item(),
            "args": vars(args),
            "run_name": run_name,
            "input_groups": model.input_groups,
            "latent_dim": model.latent_dim,
        },
        checkpoint_path,
    )

    print("\nSaved:", checkpoint_path)

    print("\nEncoder control voltages:")
    for i, unit in enumerate(model.encoder.units):
        print(f"  unit {i}: {unit.control_voltages.detach().cpu().numpy()}")

    fig_path = plot_reconstructions(
        model=model,
        x_input=x_volt,
        x_target=x_pixels,
        n=args.image_size,
        max_patterns=x_pixels.shape[0],
        save_path=results_dir / f"{run_name}_reconstructions.png",
    )

    print("\nSaved reconstruction figure:", fig_path)


if __name__ == "__main__":
    main()
