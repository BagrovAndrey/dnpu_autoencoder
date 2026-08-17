import torch
import torch.nn as nn

from brainspy.processors.processor import Processor
from brainspy.processors.dnpu import DNPU

class DNPUUnit_DNPUChild(DNPU):
    """
    A child of brainspy.processors.dnpu.DNPU that exposes the control voltages 
    as a trainable parameter.
    """
    def __init__(
        self,
        processor: Processor,
        data_input_indices: list[int] = [[1, 2, 4, 5]],
        forward_pass_type: str = 'vec',
    ):
        super(DNPUUnit_DNPUChild, self).__init__(
            processor,
            data_input_indices,
            forward_pass_type = forward_pass_type
        )
        self.input_transform = True
        # initializes control voltages as learnable parameters
        self.reset()

    def add_input_transform(self, input_range, strict = True):
        super(DNPUUnit_DNPUChild, self).add_input_transform(input_range, strict)

    def _apply_input_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.unique_transform:
            x = (x * self.scale) + self.offset
        else:
            scale = self.scale.expand_as(x)
            offset = self.offset.expand_as(x)
            x = (x * scale) + offset
        return x
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the input tensor x by concatenating the control voltages to it.
        Important: The indices of the control voltages and inputs are important to be preserved.
            x: tensor of shape (batch, n_data)
            returns: tensor of shape (batch, 1)
        """
        if self.input_transform:
            x = self._apply_input_transform(x)
        return x

    def merge_electrode_data(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge the input data to be fed to the input data electrodes with the
        data to be fed to the control voltage electrodes.

        Parameters
        ----------
        x: torch.tensor
            Input data that will be fed into the input data electrodes.

        Returns
        -------
        data: torch.Tensor
            A tensor with the input data and control voltage data to be fed
            through the activation electrodes, ordered according to the configurations
            of the indices for the data input and control voltage inputs to the
            DNPU convolution architecture. The data is given with a shape of:
            (batch_size,electrode_no).
        """
        # if x.ndim != 2 or x.shape[1] != self.n_data:
        #     raise ValueError(
        #         f"Expected x shape (batch, {self.n_data}), "
        #         f"got {tuple(x.shape)}"
        #     )

        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        full_x = torch.zeros(batch_size, 7, device=device, dtype=dtype)

        full_x[:, self.data_input_indices[0]] = x

        controls = self.control_voltages.to(device=device, dtype=dtype)
        full_x[:, self.control_indices[0]] = controls[0].unsqueeze(0).expand(batch_size, -1)

        return full_x


    def forward(self, x_data):
        """
        x_data: tensor of shape (batch, n_data)
        returns: tensor of shape (batch, 1)
        """
        # if x_data.ndim != 2 or x_data.shape[1] != self.n_data:
        #     raise ValueError(
        #         f"Expected x_data shape (batch, {self.n_data}), "
        #         f"got {tuple(x_data.shape)}"
        #     )

        x_data = self.preprocess(x_data)
        x_data = self.merge_electrode_data(x_data)
        y = self.processor(x_data)
        return y

    @torch.no_grad()
    def clip_controls_(self):
        self.constraint_control_voltages()

    @property
    def n_data(self):
        return len(self.data_input_indices[0])

    @property
    def n_control(self):
        return len(self.control_indices[0])


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

    def get_input_ranges(self):
            pass
    
    def add_input_transform(self, input_range, strict=None):
        pass

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
        DNPUChild = True
    ):
        super().__init__()
        self.DNPUChild = DNPUChild
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
            DNPUUnit_DNPUChild(
                processor=processor,
            ) if self.DNPUChild else DNPUUnit(
                            processor=processor,
                            voltage_ranges=voltage_ranges,
                            data_indices=data_indices,
                            control_indices=control_indices,
                            init=init,
                        )
            for _ in self.input_groups
        ])

        # add input transforms to each unit if is a child of DNPU
        if self.DNPUChild:
            for unit in self.units:
                unit.add_input_transform(unit.get_input_ranges(), strict=True)

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
