import torch
import torch.nn as nn


class DNPUUnit(nn.Module):
    """
    One DNPU call:

        n_data data voltages + n_control trainable control voltages -> 1 scalar output

    The calibrated surrogate processor expects exactly 7 activation electrodes,
    so n_data + n_control must be 7.

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

        if len(self.data_indices) + len(self.control_indices) != 7:
            raise ValueError("data_indices + control_indices must contain 7 electrodes.")

        if sorted(self.data_indices + self.control_indices) != list(range(7)):
            raise ValueError(
                "data_indices and control_indices must be a non-overlapping "
                "partition of electrodes 0..6."
            )

        n_control = len(self.control_indices)

        if init == "center":
            ctrl0 = 0.5 * (
                self.v_min[self.control_indices] + self.v_max[self.control_indices]
            )
        elif init == "random":
            lo = self.v_min[self.control_indices]
            hi = self.v_max[self.control_indices]
            ctrl0 = lo + torch.rand(n_control) * (hi - lo)
        else:
            raise ValueError(f"Unknown init: {init}")

        self.control_voltages = nn.Parameter(ctrl0.clone())

    @property
    def n_data(self):
        return len(self.data_indices)

    @property
    def n_control(self):
        return len(self.control_indices)

    @torch.no_grad()
    def clip_controls_(self):
        lo = self.v_min[self.control_indices]
        hi = self.v_max[self.control_indices]
        self.control_voltages.clamp_(lo, hi)

    def forward(self, x_data):
        """
        x_data: tensor of shape (batch, n_data)
        returns: tensor of shape (batch, 1)
        """
        if x_data.ndim != 2 or x_data.shape[1] != self.n_data:
            raise ValueError(
                f"Expected x_data shape (batch, {self.n_data}), "
                f"got {tuple(x_data.shape)}"
            )

        batch_size = x_data.shape[0]
        device = x_data.device
        dtype = x_data.dtype

        full_x = torch.zeros(batch_size, 7, device=device, dtype=dtype)

        full_x[:, self.data_indices] = x_data

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

    Each unit receives n_data selected coordinates from the input vector and
    has its own trainable control voltages.

    input_groups:
        list of lists/tuples, length n_units.
        Each group contains exactly n_data indices into the input vector.

    Example for 4x4 images and n_data=4:
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

        n_data = len(data_indices)

        for group in input_groups:
            if len(group) != n_data:
                raise ValueError(
                    f"Each input group must have length {n_data}, got {group}"
                )

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

    @property
    def n_units(self):
        return len(self.units)

    @property
    def n_data(self):
        return self.units[0].n_data

    @property
    def n_control(self):
        return self.units[0].n_control

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
            y = unit(x_group)
            outputs.append(y)

        return torch.cat(outputs, dim=1)
