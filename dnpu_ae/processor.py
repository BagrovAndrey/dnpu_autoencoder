"""BrainSpy backend ownership helpers for the CIFAR experiments.

This module is the single place that constructs the shared BrainSpy
``Processor``. The processor wraps the frozen surrogate backend, while each
``DNPUConv2d_DNPUChild`` created through the backend owns its own trainable
``control_voltages`` parameters.
"""

from pathlib import Path

import torch
from brainspy.processors.processor import Processor

from dnpu_ae.dnpu_conv import DNPUConv2d_DNPUChild


DEFAULT_WAVEFORM_CONFIG = {
    "processor_type": "simulation",
    "waveform": {
        "plateau_length": 1,
        "slope_length": 0,
    },
}


class DNPUBackend:
    """Simulation/hardware abstraction that owns one shared BrainSpy processor.

    The backend is responsible for constructing the Processor exactly once,
    freezing all surrogate parameters, and producing ``DNPUConv2d_DNPUChild`` layers that
    reuse that shared processor while keeping per-layer ``control_voltages``
    trainable.
    """

    def __init__(self, processor, source_path=None):
        self.processor = processor
        self.source_path = Path(source_path) if source_path is not None else None
        self.freeze_surrogate_parameters()

    @classmethod
    def from_surrogate_checkpoint(cls, checkpoint_path="surrogate_model.pt"):
        """Create a backend from the local surrogate checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        processor = Processor(
            configs=DEFAULT_WAVEFORM_CONFIG,
            info=ckpt["info"],
            model_state_dict=ckpt["model_state_dict"],
        )

        return cls(processor=processor, source_path=checkpoint_path)

    def freeze_surrogate_parameters(self):
        """Freeze the shared surrogate backend; only DNPU controls remain trainable."""
        for param in self.processor.parameters():
            param.requires_grad = False

    def conv2d(
        self,
        *,
        data_input_indices,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        forward_pass_type="vec",
    ):
        """Create a DNPU convolution layer that reuses the shared frozen Processor."""
        return DNPUConv2d_DNPUChild(
            processor=self.processor,
            data_input_indices=data_input_indices,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            forward_pass_type=forward_pass_type,
        )


def make_backend(checkpoint_path="surrogate_model.pt"):
    """Compatibility-friendly backend factory for shared Processor ownership."""
    return DNPUBackend.from_surrogate_checkpoint(checkpoint_path=checkpoint_path)


def make_processor():
    """Compatibility wrapper returning the shared Processor from a new backend."""
    return make_backend().processor
