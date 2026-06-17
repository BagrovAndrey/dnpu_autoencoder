import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from brainspy.processors.processor import Processor

from data import make_bars_stripes, pixels_to_voltages
from models import PlanarDNPUEncoderDigitalDecoder
from visualize import plot_reconstructions, plot_noisy_reconstructions


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
    rec = F.binary_cross_entropy_with_logits(logits, target).item()
    pred = (torch.sigmoid(logits) > 0.5).float()
    pixel_acc = (pred == target).float().mean().item()
    pattern_acc = (pred == target).all(dim=1).float().mean().item()
    return rec, pixel_acc, pattern_acc


def add_voltage_noise(x_volt, sigma, low=-0.5, high=0.5):
    noisy = x_volt + sigma * torch.randn_like(x_volt)
    return noisy.clamp(low, high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/hybrid_model.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20],
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    processor, voltage_ranges = make_processor()

    model = PlanarDNPUEncoderDigitalDecoder(
        processor=processor,
        voltage_ranges=voltage_ranges,
        init="center",
        freeze_encoder=False,
    )

    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_pixels = make_bars_stripes(n=4)
    x_volt = pixels_to_voltages(x_pixels, low=-0.5, high=0.5)

    print("Noise robustness test")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  repeats:    {args.repeats}")
    print(f"  sigmas:     {args.sigmas}")
    print()

    print(
        "sigma      rec_mean   rec_std    pixel_acc_mean pixel_acc_std  pattern_acc_mean pattern_acc_std  z_std_mean z_abs_max_mean"
    )

    for sigma in args.sigmas:
        rec_values = []
        pixel_values = []
        pattern_values = []
        z_std_values = []
        z_abs_max_values = []

        for _ in range(args.repeats):
            if sigma == 0.0:
                x_eval = x_volt
            else:
                x_eval = add_voltage_noise(x_volt, sigma=sigma, low=-0.5, high=0.5)

            with torch.no_grad():
                logits, z = model(x_eval)

            rec, pixel_acc, pattern_acc = reconstruction_metrics(logits, x_pixels)

            rec_values.append(rec)
            pixel_values.append(pixel_acc)
            pattern_values.append(pattern_acc)
            z_std_values.append(z.std().item())
            z_abs_max_values.append(z.abs().max().item())

        rec_t = torch.tensor(rec_values)
        pix_t = torch.tensor(pixel_values)
        pat_t = torch.tensor(pattern_values)
        zstd_t = torch.tensor(z_std_values)
        zmax_t = torch.tensor(z_abs_max_values)

        print(
            f"{sigma:<10.3f}"
            f"{rec_t.mean().item():<11.4f}"
            f"{rec_t.std(unbiased=False).item():<11.4f}"
            f"{pix_t.mean().item():<15.3f}"
            f"{pix_t.std(unbiased=False).item():<15.3f}"
            f"{pat_t.mean().item():<17.3f}"
            f"{pat_t.std(unbiased=False).item():<17.3f}"
            f"{zstd_t.mean().item():<11.3f}"
            f"{zmax_t.mean().item():<.3f}"
        )

    # Save one visual example at moderate noise.
    sigma_vis = 0.1
    x_noisy = add_voltage_noise(x_volt, sigma=sigma_vis, low=-0.5, high=0.5)

    fig_path = plot_noisy_reconstructions(
        model=model,
        x_noisy=x_noisy,
        x_target=x_pixels,
        n=4,
        max_patterns=30,
        save_path=results_dir / f"noise_robustness_sigma{sigma_vis:.2f}.png",
        input_low=-0.5,
        input_high=0.5,
    )

    print()
    print("Saved visual example:", fig_path)


if __name__ == "__main__":
    main()
