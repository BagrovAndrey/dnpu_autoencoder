import torch
from pathlib import Path

from brainspy.processors.processor import Processor
from dnpu_layer import DNPUUnit


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

    unit = DNPUUnit(
        processor=processor,
        voltage_ranges=voltage_ranges,
        data_indices=(0, 1, 2, 3),
        control_indices=(4, 5, 6),
        init="center",
    )

    ranges = torch.tensor(voltage_ranges, dtype=torch.float32)
    data_idx = torch.tensor([0, 1, 2, 3])
    lo = ranges[data_idx, 0]
    hi = ranges[data_idx, 1]

    # Random data voltages within the ranges of electrodes 0..3.
    x_data = lo + torch.rand(8, 4) * (hi - lo)

    y = unit(x_data)
    loss = y.pow(2).mean()

    loss.backward()

    print("x_data shape:", tuple(x_data.shape))
    print("y shape:", tuple(y.shape))
    print("initial controls:", unit.control_voltages.detach())
    print("control grad:", unit.control_voltages.grad)
    print("control grad finite:", torch.isfinite(unit.control_voltages.grad).all().item())

    # Test clipping.
    with torch.no_grad():
        unit.control_voltages[:] = torch.tensor([999.0, -999.0, 999.0])
    unit.clip_controls_()
    print("controls after clipping:", unit.control_voltages.detach())


if __name__ == "__main__":
    main()
