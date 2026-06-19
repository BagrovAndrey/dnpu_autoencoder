import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from brainspy.processors.processor import Processor

from data import make_bars_stripes, pixels_to_voltages
from models import PlanarDNPUEncoderDigitalDecoder
from visualize import save_reconstruction_cases, plot_reconstruction_sample


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


@torch.no_grad()
def evaluate(model, x_volt, x_pixels, z0):
    model.eval()

    logits, z = model(x_volt)
    z_raw = model.last_z_raw

    rec = F.binary_cross_entropy_with_logits(logits, x_pixels)
    z_penalty = torch.relu(z_raw.abs() - z0).pow(2).mean()

    pred = (torch.sigmoid(logits) > 0.5).float()
    pixel_acc = (pred == x_pixels).float().mean()
    pattern_acc = (pred == x_pixels).all(dim=1).float().mean()

    return {
        "rec": rec.item(),
        "z_penalty": z_penalty.item(),
        "pixel_acc": pixel_acc.item(),
        "pattern_acc": pattern_acc.item(),
        "z_raw_std": z_raw.std().item(),
        "z_raw_abs_max": z_raw.abs().max().item(),
        "z_std": z.std().item(),
    }


def make_run_name(args):
    return (
        f"gen_"
        f"{args.image_size}x{args.image_size}_"
        f"ndata{args.n_data}_"
        f"nctrl{args.n_control}_"
        f"latent{args.latent_dim if args.latent_dim is not None else 'raw'}_"
        f"train{args.train_size}_"
        f"{args.group_type}_"
        f"seed{args.seed}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/test generalization for planar DNPU autoencoder."
    )

    # Architecture.
    parser.add_argument("--image-size", type=int, default=4)
    parser.add_argument("--n-data", type=int, default=4)
    parser.add_argument("--n-control", type=int, default=3)
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="Compressed digital latent dimension after DNPU readouts. Default: no compression.",
    )
    parser.add_argument(
        "--group-type",
        type=str,
        default="auto",
        choices=["auto", "2x2", "horizontal", "vertical"],
    )

    # Split.
    parser.add_argument("--train-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)

    # Training.
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-z", type=float, default=1e-3)
    parser.add_argument("--z0", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    # Input voltage encoding.
    parser.add_argument("--v-low", type=float, default=-0.5)
    parser.add_argument("--v-high", type=float, default=0.5)

    # Output.
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-cases", action="store_true")

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

    num_patterns = x_pixels.shape[0]

    if args.train_size <= 0 or args.train_size >= num_patterns:
        raise ValueError(
            f"--train-size must be between 1 and {num_patterns - 1}, "
            f"got {args.train_size}."
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    perm = torch.randperm(num_patterns, generator=generator)

    train_idx = perm[:args.train_size]
    test_idx = perm[args.train_size:]

    x_train_volt = x_volt[train_idx]
    x_train_pixels = x_pixels[train_idx]

    x_test_volt = x_volt[test_idx]
    x_test_pixels = x_pixels[test_idx]

    processor, voltage_ranges = make_processor()

    model = PlanarDNPUEncoderDigitalDecoder(
        processor=processor,
        voltage_ranges=voltage_ranges,
        image_size=args.image_size,
        n_data=args.n_data,
        n_control=args.n_control,
        latent_dim=args.latent_dim,
        group_type=args.group_type,
        init="center",
        freeze_encoder=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("DNPU autoencoder generalization test")
    print(f"  run_name:       {run_name}")
    print(f"  image_size:     {args.image_size}x{args.image_size}")
    print(f"  patterns total: {num_patterns}")
    print(f"  train_size:     {len(train_idx)}")
    print(f"  test_size:      {len(test_idx)}")
    print(f"  n_data:         {args.n_data}")
    print(f"  n_control:      {args.n_control}")
    print(f"  group_type:     {args.group_type}")
    print(f"  raw_latent_dim: {model.raw_latent_dim}")
    print(f"  latent_dim:     {model.latent_dim}")
    print(f"  epochs:         {args.epochs}")
    print(f"  lr:             {args.lr}")
    print(f"  weight_decay:   {args.weight_decay}")
    print(f"  lambda_z:       {args.lambda_z}")
    print(f"  z0:             {args.z0}")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()

        logits, z = model(x_train_volt)
        z_raw = model.last_z_raw

        loss_recon = F.binary_cross_entropy_with_logits(logits, x_train_pixels)
        z_penalty = torch.relu(z_raw.abs() - args.z0).pow(2).mean()
        loss = loss_recon + args.lambda_z * z_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clip_controls_()

        if epoch == 1 or epoch % 500 == 0:
            train_metrics = evaluate(model, x_train_volt, x_train_pixels, args.z0)
            test_metrics = evaluate(model, x_test_volt, x_test_pixels, args.z0)

            print(
                f"epoch {epoch:5d} | "
                f"train_rec {train_metrics['rec']:.6f} | "
                f"test_rec {test_metrics['rec']:.6f} | "
                f"train_pix {train_metrics['pixel_acc']:.3f} | "
                f"test_pix {test_metrics['pixel_acc']:.3f} | "
                f"train_pat {train_metrics['pattern_acc']:.3f} | "
                f"test_pat {test_metrics['pattern_acc']:.3f} | "
                f"z_raw_abs_max {train_metrics['z_raw_abs_max']:.3f}"
            )

    train_metrics = evaluate(model, x_train_volt, x_train_pixels, args.z0)
    test_metrics = evaluate(model, x_test_volt, x_test_pixels, args.z0)
    all_metrics = evaluate(model, x_volt, x_pixels, args.z0)

    checkpoint_path = results_dir / f"{run_name}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "run_name": run_name,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "input_groups": model.input_groups,
            "raw_latent_dim": model.raw_latent_dim,
            "latent_dim": model.latent_dim,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "all_metrics": all_metrics,
        },
        checkpoint_path,
    )

    print("\nFinal metrics:")
    print("  train:", train_metrics)
    print("  test: ", test_metrics)
    print("  all:  ", all_metrics)
    print("\nSaved:", checkpoint_path)

    sample_path, sample_indices = plot_reconstruction_sample(
        model=model,
        x_input=x_test_volt,
        x_target=x_test_pixels,
        n=args.image_size,
        num_examples=8,
        seed=args.seed,
        save_path=results_dir / f"{run_name}_test_sample.png",
    )

    print("Saved test sample figure:", sample_path)
    print("Test sample local indices:", sample_indices)

    if args.save_cases:
        cases_dir = results_dir / f"{run_name}_test_cases"
        saved_cases = save_reconstruction_cases(
            model=model,
            x_input=x_test_volt,
            x_target=x_test_pixels,
            n=args.image_size,
            out_dir=cases_dir,
            prefix=run_name,
        )
        print("Saved individual test cases:", cases_dir)
        print("Number of case files:", len(saved_cases))


if __name__ == "__main__":
    main()
