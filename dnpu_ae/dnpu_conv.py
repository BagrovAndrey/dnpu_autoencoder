"""DNPU convolution built on top of DNPUUnit_DNPUChild.

This module keeps the CIFAR-facing convolution API compatible with the older
BrainSpy DNPUConv2d usage, while the primitive DNPU call is delegated to
DNPUUnit_DNPUChild from the legacy dnpu_layer module.
"""

import torch
import torch.nn as nn

from dnpu_layer import DNPUUnit_DNPUChild


class DNPUConv2d_DNPUChild(DNPUUnit_DNPUChild):
    """Vectorized conv2d wrapper around DNPUUnit_DNPUChild.

    Each input/output-channel pair owns its own trainable control voltages. The
    convolution is implemented with ``torch.nn.Unfold`` and a batched processor
    call; no Python loop is used over spatial windows.
    """

    def __init__(
        self,
        processor,
        data_input_indices: list,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        forward_pass_type: str = "vec",
    ):
        assert type(in_channels) is int, "in_channels should be integer"
        assert type(out_channels) is int, "out_channels should be integer"
        assert type(kernel_size) is int, (
            "kernel_size should be integer. Only square kernel sizes are supported, "
            "represented by a single number."
        )
        assert type(stride) is int, "stride should be integer"
        assert type(padding) is int, "padding should be integer"
        assert len(data_input_indices) > 0, "data_input_indices must contain at least one DNPU node"
        assert all(
            len(node_indices) == len(data_input_indices[0])
            for node_indices in data_input_indices
        ), "All DNPU nodes must use the same number of data electrodes."
        assert (
            torch.tensor(data_input_indices).numel() == kernel_size**2
        ), (
            "Data input indices should be defined as mapping a single kernel. "
            "E.g., for a 3x3 convolution you need 9 data input indices, "
            "represented as (dnpu_node_no=3, data_input_no_per_dnpu_node=3)."
        )

        self.raw_inputs_list = [list(node_indices) for node_indices in data_input_indices]
        repeated_indices = []
        for _in_channel in range(in_channels):
            for _out_channel in range(out_channels):
                for node_indices in self.raw_inputs_list:
                    repeated_indices.append(list(node_indices))

        super().__init__(
            processor=processor,
            data_input_indices=repeated_indices,
            forward_pass_type=forward_pass_type,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.unfold = nn.Unfold(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.input_transform = False
        self.kernel_node_no = len(self.raw_inputs_list)
        self.data_electrode_no = len(self.raw_inputs_list[0])
        self.pair_node_no = in_channels * out_channels * self.kernel_node_no
        self.activation_electrode_no = max(
            max(max(indices) for indices in self.data_input_indices),
            max(max(indices) for indices in self.control_indices),
        ) + 1

    def get_output_dim(self, input_dim: int) -> int:
        """Return the spatial output size for a square input."""
        return ((input_dim + (2 * self.padding) - self.kernel_size) // self.stride) + 1

    def preprocess(self, x):
        """Extract sliding windows and vectorize channel/pair dimensions."""
        x = self.unfold(x)
        x = x.transpose(1, 2)
        x = x.reshape(
            x.shape[0],
            x.shape[1],
            self.in_channels,
            self.kernel_node_no,
            self.data_electrode_no,
        )

        if self.input_transform:
            x = self._apply_input_transform(x)

        x = x.unsqueeze(3).expand(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            self.out_channels,
            x.shape[3],
            x.shape[4],
        )
        return x

    def merge_electrode_data(self, x):
        """Merge unfolded inputs with per-pair control voltages."""
        data_dim = x.shape
        full_x = x.new_zeros(*data_dim[:-1], self.activation_electrode_no)

        for node_idx, node_indices in enumerate(self.raw_inputs_list):
            full_x[..., node_idx, node_indices] = x[..., node_idx, :]

        controls = self.control_voltages.to(device=x.device, dtype=x.dtype).reshape(
            self.in_channels,
            self.out_channels,
            self.kernel_node_no,
            -1,
        )

        for node_idx, node_control_indices in enumerate(self.control_indices[: self.kernel_node_no]):
            full_x[..., node_idx, node_control_indices] = controls[:, :, node_idx, :]

        return full_x.reshape(-1, self.activation_electrode_no), data_dim

    def postprocess(self, result, data_dim, output_height, output_width):
        """Restore convolution output layout and sum pair/node contributions."""
        result = result.reshape(*data_dim[:-1], -1)
        if result.shape[-1] != 1:
            raise ValueError(
                f"Expected a single readout value per DNPU evaluation, got {result.shape[-1]}."
            )

        result = result.squeeze(-1)
        result = result.sum(dim=2)
        result = result.sum(dim=3)
        result = result.transpose(1, 2)
        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)
        return result

    def forward(self, x):
        """Apply DNPU convolution with one shared processor and per-pair controls."""
        output_height = self.get_output_dim(x.shape[2])
        output_width = self.get_output_dim(x.shape[3])
        x = self.preprocess(x)
        x, original_data_dim = self.merge_electrode_data(x)
        x = self.processor(x)
        x = self.postprocess(x, original_data_dim, output_height, output_width)
        return x
