import torch
from pathlib import Path

from brainspy.processors.processor import Processor


def main():
    checkpoint_path = Path("surrogate_model.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find {checkpoint_path.resolve()}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

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

    ranges = torch.tensor(
        ckpt["info"]["electrode_info"]["activation_electrodes"]["voltage_ranges"],
        dtype=torch.float32,
    )

    v_min = ranges[:, 0]
    v_max = ranges[:, 1]

    print("Voltage ranges:")
    for i, (lo, hi) in enumerate(zip(v_min, v_max)):
        print(f"  electrode {i}: [{lo.item():.4f}, {hi.item():.4f}]")

    # One batch of 5 random 7-electrode voltage vectors.
    x = v_min + torch.rand(5, 7) * (v_max - v_min)
    x.requires_grad_(True)

    y = processor(x)

    print("\nForward pass:")
    print("  x shape:", tuple(x.shape))
    print("  y shape:", tuple(y.shape))
    print("  y:", y.detach().cpu())

    loss = y.pow(2).mean()
    loss.backward()

    print("\nGradient check:")
    print("  x.grad shape:", tuple(x.grad.shape))
    print("  x.grad finite:", torch.isfinite(x.grad).all().item())
    print("  x.grad mean abs:", x.grad.abs().mean().item())


if __name__ == "__main__":
    main()
