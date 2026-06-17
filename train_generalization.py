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


def reconstruction_metrics(logits, target):
    loss = F.binary_cross_entropy_with_logits(logits, target).item()
    pred = (torch.sigmoid(logits) > 0.5).float()
    pixel_acc = (pred == target).float().mean().item()
    pattern_acc = (pred == target).all(dim=1).float().mean().item()
    return loss, pixel_acc, pattern_acc


def make_split(num_patterns, train_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)

    perm = torch.randperm(num_patterns, generator=g)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    return train_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-z", type=float, default=1e-3)
    parser.add_argument("--z0", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    x_pixels_all = make_bars_stripes(n=4)
    x_volt_all = pixels_to_voltages(x_pixels_all, low=-0.5, high=0.5)

    num_patterns = x_pixels_all.shape[0]
    train_idx, test_idx = make_split(num_patterns, args.train_size, args.seed)

    x_pixels_train = x_pixels_all[train_idx]
    x_volt_train = x_volt_all[train_idx]

    x_pixels_test = x_pixels_all[test_idx]
    x_volt_test = x_volt_all[test_idx]

    processor, voltage_ranges = make_processor()

    model = PlanarDNPUEncoderDigitalDecoder(
        processor=processor,
        voltage_ranges=voltage_ranges,
        init="center",
        freeze_encoder=False,
    )

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Generalization experiment")
    print(f"  train_size: {args.train_size}")
    print(f"  test_size:  {len(test_idx)}")
    print(f"  seed:       {args.seed}")
    print(f"  train_idx:  {train_idx.tolist()}")
    print(f"  test_idx:   {test_idx.tolist()}")

    for epoch in range(1, args.epochs + 1):
        logits_train, z_train = model(x_volt_train)

        loss_recon = F.binary_cross_entropy_with_logits(
            logits_train,
            x_pixels_train,
        )

        z_penalty = torch.relu(z_train.abs() - args.z0).pow(2).mean()
        loss = loss_recon + args.lambda_z * z_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clip_controls_()

        if epoch == 1 or epoch % 500 == 0:
            with torch.no_grad():
                logits_all, z_all = model(x_volt_all)
                logits_test, z_test = model(x_volt_test)

                train_rec, train_pix, train_pat = reconstruction_metrics(
                    logits_train, x_pixels_train
                )
                test_rec, test_pix, test_pat = reconstruction_metrics(
                    logits_test, x_pixels_test
                )
                all_rec, all_pix, all_pat = reconstruction_metrics(
                    logits_all, x_pixels_all
                )

                z_mean = z_all.mean().item()
                z_std = z_all.std().item()
                z_abs_max = z_all.abs().max().item()

            print(
                f"epoch {epoch:5d} | "
                f"loss {loss.item():.6f} | "
                f"train_rec {train_rec:.6f} | "
                f"test_rec {test_rec:.6f} | "
                f"all_rec {all_rec:.6f} | "
                f"train_pix {train_pix:.3f} | "
                f"test_pix {test_pix:.3f} | "
                f"all_pix {all_pix:.3f} | "
                f"train_pat {train_pat:.3f} | "
                f"test_pat {test_pat:.3f} | "
                f"all_pat {all_pat:.3f} | "
                f"z_std {z_std:.3f} | "
                f"z_abs_max {z_abs_max:.3f}"
            )

    checkpoint_path = results_dir / f"generalization_train{args.train_size}_seed{args.seed}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_size": args.train_size,
            "seed": args.seed,
            "lambda_z": args.lambda_z,
            "z0": args.z0,
        },
        checkpoint_path,
    )

    print("\nSaved:", checkpoint_path)

    print("\nEncoder control voltages:")
    for i, unit in enumerate(model.encoder.units):
        print(f"  unit {i}: {unit.control_voltages.detach().cpu().numpy()}")

    fig_path = plot_reconstructions(
        model=model,
        x_input=x_volt_all,
        x_target=x_pixels_all,
        n=4,
        max_patterns=30,
        save_path=results_dir / f"generalization_train{args.train_size}_seed{args.seed}.png",
    )

    print("\nSaved reconstruction figure:", fig_path)


if __name__ == "__main__":
    main()
