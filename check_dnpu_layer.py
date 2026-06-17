import torch
from pathlib import Path

from brainspy.processors.processor import Processor
from dnpu_layer import DNPULayer


def main():
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

    # Four 2x2 patches of a flattened 4x4 image:
    #
    #  0  1 |  2  3
    #  4  5 |  6  7
    # ------+------
    #  8  9 | 10 11
    # 12 13 | 14 15
    patch_groups = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]

    layer = DNPULayer(
        processor=processor,
        voltage_ranges=voltage_ranges,
        input_groups=patch_groups,
        init="center",
    )

    # For this shape test, use random voltages in a conservative common range.
    # Later we will map binary pixels to actual electrode-specific voltages.
    x = -0.5 + torch.rand(10, 16)

    z = layer(x)
    loss = z.pow(2).mean()
    loss.backward()

    print("x shape:", tuple(x.shape))
    print("z shape:", tuple(z.shape))
    print("z sample:", z[:3].detach())

    for i, unit in enumerate(layer.units):
        grad = unit.control_voltages.grad
        print(f"unit {i} controls:", unit.control_voltages.detach())
        print(f"unit {i} grad finite:", torch.isfinite(grad).all().item())
        print(f"unit {i} grad mean abs:", grad.abs().mean().item())

    layer.clip_controls_()
    print("clipping ok")


if __name__ == "__main__":
    main()
