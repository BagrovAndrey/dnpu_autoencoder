import torch
from pathlib import Path
from visualize import plot_reconstructions
from brainspy.processors.processor import Processor
from data import make_bars_stripes, pixels_to_voltages
from models import PlanarDNPUEncoderDigitalDecoder


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


def main():
    torch.manual_seed(0)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    x_pixels = make_bars_stripes(n=4)
    x_volt = pixels_to_voltages(x_pixels, low=-0.5, high=0.5)

    processor, voltage_ranges = make_processor()

    model = PlanarDNPUEncoderDigitalDecoder(
        processor=processor,
        voltage_ranges=voltage_ranges,
        init="center",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 5000

    for epoch in range(1, n_epochs + 1):
        logits, z = model(x_volt)

        #loss = torch.nn.functional.binary_cross_entropy_with_logits(
        #    logits,
        #    x_pixels,
        #)
        
        #logits, z = model(x_volt)

        loss_recon = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            x_pixels,
        )

        z_penalty = z.pow(2).mean()

        lambda_z = 1e-4
        loss = loss_recon + lambda_z * z_penalty
        z_abs_max = z.detach().abs().max().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clip_controls_()

        if epoch == 1 or epoch % 500 == 0:
            pixel_acc, pattern_acc = reconstruction_accuracy(logits, x_pixels)
            z_mean = z.detach().mean().item()
            z_std = z.detach().std().item()

            print(
                f"epoch {epoch:5d} | "
                f"loss {loss.item():.6f} | "
                f"rec {loss_recon.item():.6f} | "
                f"z_pen {z_penalty.item():.6f} | "
                f"pixel_acc {pixel_acc:.3f} | "
                f"pattern_acc {pattern_acc:.3f} | "
                f"z_mean {z_mean:.3f} | "
                f"z_std {z_std:.3f}"
                f"z_abs_max {z_abs_max:.3f}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "loss": loss.item(),
        },
        results_dir / "hybrid_model.pt",
    )

    print("\nSaved:", results_dir / "hybrid_model.pt")

    print("\nEncoder control voltages:")
    for i, unit in enumerate(model.encoder.units):
        print(f"  unit {i}: {unit.control_voltages.detach().cpu().numpy()}")
        
    fig_path = plot_reconstructions(
        model=model,
        x_input=x_volt,
        x_target=x_pixels,
        n=4,
        max_patterns=30,
        save_path=results_dir / "hybrid_reconstructions.png",
    )

    print("\nSaved reconstruction figure:", fig_path)


if __name__ == "__main__":
    main()
