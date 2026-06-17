import torch
import torch.nn as nn


class DNPUUnit(nn.Module):
    """
    One DNPU call:
        4 data voltages + 3 trainable control voltages -> 1 scalar output.

    This assumes the surrogate processor expects 7 activation electrodes.
    By default:
        electrodes 0,1,2,3 are data inputs
        electrodes 4,5,6 are trainable controls
    """

    def __init__(
        self,
        processor,
        voltage_ranges,
        data_indices=(0, 1, 2, 3),
        control_indices=(4, 5, 6),
        init="center",
    ):
        super().__init__()

        self.processor = processor
        self.data_indices = list(data_indices)
        self.control_indices = list(control_indices)

        voltage_ranges = torch.as_tensor(voltage_ranges, dtype=torch.float32)
        if voltage_ranges.shape != (7, 2):
            raise ValueError(f"Expected voltage_ranges shape (7, 2), got {voltage_ranges.shape}")

        self.register_buffer("voltage_ranges", voltage_ranges)
        self.register_buffer("v_min", voltage_ranges[:, 0])
        self.register_buffer("v_max", voltage_ranges[:, 1])

        if len(self.data_indices) != 4:
            raise ValueError("This MVP DNPUUnit expects exactly 4 data electrodes.")
        if len(self.control_indices) != 3:
            raise ValueError("This MVP DNPUUnit expects exactly 3 control electrodes.")

        if init == "center":
            ctrl0 = 0.5 * (
                self.v_min[self.control_indices] + self.v_max[self.control_indices]
            )
        elif init == "random":
            lo = self.v_min[self.control_indices]
            hi = self.v_max[self.control_indices]
            ctrl0 = lo + torch.rand(3) * (hi - lo)
        else:
            raise ValueError(f"Unknown init: {init}")

        self.control_voltages = nn.Parameter(ctrl0.clone())

    @torch.no_grad()
    def clip_controls_(self):
        lo = self.v_min[self.control_indices]
        hi = self.v_max[self.control_indices]
        self.control_voltages.clamp_(lo, hi)

    def forward(self, x_data):
        """
        x_data: tensor of shape (batch, 4)
        returns: tensor of shape (batch, 1)
        """
        if x_data.ndim != 2 or x_data.shape[1] != 4:
            raise ValueError(f"Expected x_data shape (batch, 4), got {tuple(x_data.shape)}")

        batch_size = x_data.shape[0]
        device = x_data.device
        dtype = x_data.dtype

        full_x = torch.zeros(batch_size, 7, device=device, dtype=dtype)

        # Fill data electrodes.
        full_x[:, self.data_indices] = x_data

        # Fill control electrodes. Same controls for every batch item.
        controls = self.control_voltages.to(device=device, dtype=dtype)
        full_x[:, self.control_indices] = controls.unsqueeze(0).expand(batch_size, -1)

        return self.processor(full_x)


class DNPULayer(nn.Module):
    """
    Simulated planar array of several physically distinct DNPU units.

    In software, all units are evaluated using the same calibrated surrogate
    model. Experimentally, they should be interpreted as separate local DNPU
    devices in a planar encoder layout, not as repeated time-multiplexed calls
    to a single physical device.

    Each unit receives 4 selected coordinates from the input vector and has
    its own 3 trainable control voltages.

    input_groups:
        list of lists/tuples, length n_units.
        Each group contains exactly 4 indices into the input vector.

    Example:
        input_groups = [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15],
        ]
    """

    def __init__(
        self,
        processor,
        voltage_ranges,
        input_groups,
        data_indices=(0, 1, 2, 3),
        control_indices=(4, 5, 6),
        init="center",
    ):
        super().__init__()

        if len(input_groups) == 0:
            raise ValueError("input_groups must contain at least one group.")

        for group in input_groups:
            if len(group) != 4:
                raise ValueError(f"Each input group must have length 4, got {group}")

        self.input_groups = [list(group) for group in input_groups]

        self.units = nn.ModuleList([
            DNPUUnit(
                processor=processor,
                voltage_ranges=voltage_ranges,
                data_indices=data_indices,
                control_indices=control_indices,
                init=init,
            )
            for _ in self.input_groups
        ])

    @torch.no_grad()
    def clip_controls_(self):
        for unit in self.units:
            unit.clip_controls_()

    def forward(self, x):
        """
        x: tensor of shape (batch, input_dim)
        returns: tensor of shape (batch, n_units)
        """
        if x.ndim != 2:
            raise ValueError(f"Expected x shape (batch, input_dim), got {tuple(x.shape)}")

        outputs = []

        for group, unit in zip(self.input_groups, self.units):
            x_group = x[:, group]
            y = unit(x_group)  # (batch, 1)
            outputs.append(y)

        return torch.cat(outputs, dim=1)
