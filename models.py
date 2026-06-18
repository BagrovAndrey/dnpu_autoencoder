import math

import torch
import torch.nn as nn

from dnpu_layer import DNPULayer


def make_2x2_patch_groups(image_size):
    """
    Non-overlapping 2x2 patches for even image_size.

    Example for 4x4 flattened layout:

         0   1 |  2   3
         4   5 |  6   7
        ------+------
         8   9 | 10  11
        12  13 | 14  15

    Returns:
        list of 4-index groups.
    """
    if image_size % 2 != 0:
        raise ValueError("2x2 patch groups require even image_size.")

    groups = []

    for row in range(0, image_size, 2):
        for col in range(0, image_size, 2):
            i00 = row * image_size + col
            i01 = row * image_size + (col + 1)
            i10 = (row + 1) * image_size + col
            i11 = (row + 1) * image_size + (col + 1)
            groups.append([i00, i01, i10, i11])

    return groups


def make_pair_groups(image_size, orientation="horizontal"):
    """
    Non-overlapping local 2-pixel groups.

    This is useful for n_data=2, n_control=5.

    orientation="horizontal":
        pairs adjacent pixels within each row.

    orientation="vertical":
        pairs adjacent pixels within each column.

    For now, image_size must be even.
    """
    if image_size % 2 != 0:
        raise ValueError("Pair groups currently require even image_size.")

    groups = []

    if orientation == "horizontal":
        for row in range(image_size):
            for col in range(0, image_size, 2):
                i0 = row * image_size + col
                i1 = row * image_size + (col + 1)
                groups.append([i0, i1])
    elif orientation == "vertical":
        for row in range(0, image_size, 2):
            for col in range(image_size):
                i0 = row * image_size + col
                i1 = (row + 1) * image_size + col
                groups.append([i0, i1])
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    return groups


def make_local_groups(image_size, n_data, group_type="auto"):
    """
    Build local planar input groups for the DNPU encoder.

    Supported simple cases:
        n_data=4 -> non-overlapping 2x2 patches
        n_data=2 -> non-overlapping adjacent pairs

    group_type:
        "auto"       -> 2x2 for n_data=4, horizontal pairs for n_data=2
        "2x2"        -> force 2x2 patches
        "horizontal" -> horizontal pairs
        "vertical"   -> vertical pairs
    """
    if group_type == "auto":
        if n_data == 4:
            return make_2x2_patch_groups(image_size)
        if n_data == 2:
            return make_pair_groups(image_size, orientation="horizontal")
        raise ValueError("auto group_type currently supports only n_data=2 or n_data=4.")

    if group_type == "2x2":
        if n_data != 4:
            raise ValueError("group_type='2x2' requires n_data=4.")
        return make_2x2_patch_groups(image_size)

    if group_type in ("horizontal", "vertical"):
        if n_data != 2:
            raise ValueError(f"group_type='{group_type}' requires n_data=2.")
        return make_pair_groups(image_size, orientation=group_type)

    raise ValueError(f"Unknown group_type: {group_type}")


class PlanarDNPUEncoderDigitalDecoder(nn.Module):
    """
    Hybrid autoencoder.

    Encoder:
        simulated planar array of local DNPU units.

    Decoder:
        minimal digital linear decoder from latent variables to image pixels.

    The decoder outputs logits. Apply sigmoid outside, or use
    BCEWithLogitsLoss during training.
    """

    def __init__(
        self,
        processor,
        voltage_ranges,
        image_size=4,
        n_data=4,
        n_control=3,
        input_groups=None,
        group_type="auto",
        init="center",
        freeze_encoder=False,
        latent_dim=None,
    ):
        super().__init__()

        if n_data + n_control != 7:
            raise ValueError("n_data + n_control must be 7.")

        self.image_size = image_size
        self.input_dim = image_size * image_size
        self.n_data = n_data
        self.n_control = n_control

        data_indices = tuple(range(n_data))
        control_indices = tuple(range(n_data, 7))

        if input_groups is None:
            input_groups = make_local_groups(
                image_size=image_size,
                n_data=n_data,
                group_type=group_type,
            )

        #self.input_groups = input_groups
        #self.latent_dim = len(input_groups)
        
        self.input_groups = input_groups
        self.raw_latent_dim = len(input_groups)

        if latent_dim is None:
            latent_dim = self.raw_latent_dim

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        self.latent_dim = latent_dim

        self.encoder = DNPULayer(
            processor=processor,
            voltage_ranges=voltage_ranges,
            input_groups=input_groups,
            data_indices=data_indices,
            control_indices=control_indices,
            init=init,
        )
        
        if self.latent_dim == self.raw_latent_dim:
            self.bottleneck = nn.Identity()
        else:
            self.bottleneck = nn.Linear(self.raw_latent_dim, self.latent_dim)

        self.decoder = nn.Linear(self.latent_dim, self.input_dim)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

    @torch.no_grad()
    def clip_controls_(self):
        self.encoder.clip_controls_()

    #def forward(self, x):
    #    """
    #    x: tensor of shape (batch, image_size*image_size), in voltage units.

    #    Returns:
    #        logits: tensor of shape (batch, image_size*image_size)
    #        z:      tensor of shape (batch, latent_dim)
    #    """
    #    z = self.encoder(x)
    #    logits = self.decoder(z)
    #    return logits, z
        
    def forward(self, x):
        """
        x: tensor of shape (batch, image_size*image_size), in voltage units.

        Returns:
            logits: tensor of shape (batch, image_size*image_size)
            z:      compressed latent tensor of shape (batch, latent_dim)

        The raw physical DNPU readouts are stored as self.last_z_raw for diagnostics
        and hardware-aware penalties.
        """
        z_raw = self.encoder(x)
        z = self.bottleneck(z_raw)

        self.last_z_raw = z_raw

        logits = self.decoder(z)
        return logits, z


class DigitalBaselineAutoencoder(nn.Module):
    """
    Small purely digital baseline with configurable bottleneck dimension.

    This is not a hardware model. It only checks that the reconstruction
    task is solvable with the chosen latent dimension.
    """

    def __init__(self, image_size=4, latent_dim=4):
        super().__init__()

        self.image_size = image_size
        self.input_dim = image_size * image_size
        self.latent_dim = latent_dim

        self.encoder = nn.Linear(self.input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, self.input_dim)

    def forward(self, x):
        z = torch.tanh(self.encoder(x))
        logits = self.decoder(z)
        return logits, z
