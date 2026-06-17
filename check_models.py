import torch
from pathlib import Path

from brainspy.processors.processor import Processor

from data import make_bars_stripes, pixels_to_voltages
from models import PlanarDNPUEncoderDigitalDecoder, DigitalBaselineAutoencoder


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


def check_model(name, model, x_volt, x_target):
    logits, z = model(x_volt)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, x_target)
    loss.backward()

    print(f"\n{name}")
    print("  input shape:", tuple(x_volt.shape))
    print("  latent shape:", tuple(z.shape))
    print("  logits shape:", tuple(logits.shape))
    print("  loss:", loss.item())

    grad_ok = True
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grad_ok = grad_ok and torch.isfinite(p.grad).all().item()

    print("  gradients finite:", grad_ok)


def main():
    x_pixels = make_bars_stripes(n=4)
    x_volt = pixels_to_voltages(x_pixels, low=-0.5, high=0.5)

    processor, voltage_ranges = make_processor()

    hybrid = PlanarDNPUEncoderDigitalDecoder(
        processor=processor,
        voltage_ranges=voltage_ranges,
        init="center",
    )

    baseline = DigitalBaselineAutoencoder()

    check_model("Hybrid DNPU encoder + digital decoder", hybrid, x_volt, x_pixels)
    hybrid.clip_controls_()

    check_model("Digital baseline", baseline, x_pixels, x_pixels)


if __name__ == "__main__":
    main()
