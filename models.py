import torch
import torch.nn as nn

from dnpu_layer import DNPULayer


def make_2x2_patch_groups_4x4():
    """
    Flattened 4x4 layout:

         0   1 |  2   3
         4   5 |  6   7
        ------+------
         8   9 | 10  11
        12  13 | 14  15

    Returns four local 2x2 patches.
    """
    return [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]


class PlanarDNPUEncoderDigitalDecoder(nn.Module):
    """
    Hybrid autoencoder.

    Encoder:
        simulated planar array of 4 local DNPU units.
        Each unit receives one 2x2 patch of a 4x4 input image.

    Decoder:
        minimal digital linear decoder from 4 latent variables to 16 pixels.

    The decoder outputs logits. Apply sigmoid outside, or use
    BCEWithLogitsLoss during training.
    """

    def __init__(
        self,
        processor,
        voltage_ranges,
        init="center",
        freeze_encoder=False,
    ):
        super().__init__()

        self.encoder = DNPULayer(
            processor=processor,
            voltage_ranges=voltage_ranges,
            input_groups=make_2x2_patch_groups_4x4(),
            init=init,
        )

        self.decoder = nn.Linear(4, 16)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

    @torch.no_grad()
    def clip_controls_(self):
        self.encoder.clip_controls_()

    def forward(self, x):
        """
        x: tensor of shape (batch, 16), in voltage units.

        Returns:
            logits: tensor of shape (batch, 16)
            z:      tensor of shape (batch, 4)
        """
        z = self.encoder(x)
        logits = self.decoder(z)
        return logits, z


class DigitalBaselineAutoencoder(nn.Module):
    """
    Small purely digital baseline with the same bottleneck dimension:

        16 -> 4 -> 16

    This is not a hardware model. It only checks that the reconstruction
    task is solvable with a 4-dimensional latent code.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Linear(16, 4)
        self.decoder = nn.Linear(4, 16)

    def forward(self, x):
        z = torch.tanh(self.encoder(x))
        logits = self.decoder(z)
        return logits, z
