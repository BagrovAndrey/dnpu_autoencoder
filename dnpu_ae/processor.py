"""BrainSPy processor construction used by the CIFAR experiments."""

from pathlib import Path

import torch
from brainspy.processors.processor import Processor


def make_processor():
    """Load the local surrogate checkpoint and construct its simulator."""
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

    return processor

