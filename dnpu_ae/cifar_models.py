"""Model definitions for the grayscale CIFAR-10 experiments."""

import torch.nn as nn
import torch.nn.functional as F
from brainspy.processors.modules.conv import DNPUConv2d

from dnpu_ae.upsampling import (
    DNPUZeroConvDecoder,
    DigitalZeroConvDecoder,
)


class DNPUConvCIFARAutoencoder(nn.Module):
    """Legacy one/two-stage DNPUConv CIFAR autoencoder.

    DNPU controls are hardware-like trainable parameters. BatchNorm provides
    digital readout calibration, and the decoder is entirely digital.
    """

    def __init__(
        self,
        processor,
        encoder_type="hybrid",
        conv_channels=8,
        conv2_channels=1,
        latent_mode="linear",
        latent_dim=64,
    ):
        super().__init__()

        if encoder_type not in ["hybrid", "dnpu2"]:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        if latent_mode not in ["raw", "linear"]:
            raise ValueError(f"Unknown latent_mode: {latent_mode}")

        self.encoder_type = encoder_type
        self.conv_channels = conv_channels
        self.conv2_channels = conv2_channels
        self.latent_mode = latent_mode
        self.requested_latent_dim = latent_dim

        self.dnpu_conv1 = DNPUConv2d(
            processor=processor,
            data_input_indices=[[0, 1, 2, 3]],
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            forward_pass_type="vec",
        )
        self.norm1 = nn.BatchNorm2d(conv_channels)

        if encoder_type == "hybrid":
            self.encoder2 = nn.Sequential(
                nn.Conv2d(conv_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
            self.raw_channels = 16
        else:
            self.dnpu_conv2 = DNPUConv2d(
                processor=processor,
                data_input_indices=[[0, 1, 2, 3]],
                in_channels=conv_channels,
                out_channels=conv2_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                forward_pass_type="vec",
            )
            self.norm2 = nn.BatchNorm2d(conv2_channels)
            self.raw_channels = conv2_channels

        self.raw_latent_dim = self.raw_channels * 8 * 8

        # raw exposes flattened physical readouts; linear adds a digital bottleneck.
        if latent_mode == "raw":
            self.to_latent = nn.Identity()
            self.latent_dim = self.raw_latent_dim
        else:
            self.to_latent = nn.Linear(self.raw_latent_dim, latent_dim)
            self.latent_dim = latent_dim

        self.from_latent = nn.Linear(self.latent_dim, 16 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode_features(self, x):
        x_voltage = x - 0.5
        h = self.dnpu_conv1(x_voltage)
        h = self.norm1(h)
        h = F.relu(h)

        if self.encoder_type == "hybrid":
            h = self.encoder2(h)
        else:
            h = self.dnpu_conv2(h)
            h = self.norm2(h)
            h = F.relu(h)
        return h

    def forward(self, x):
        h = self.encode_features(x)
        z = self.to_latent(h.flatten(start_dim=1))
        h_dec = self.from_latent(z)
        h_dec = h_dec.reshape(x.shape[0], 16, 8, 8)
        logits = self.decoder(h_dec)
        return logits, z


class DNPUStackCIFARAutoencoder(nn.Module):
    """Configurable stride-2 DNPUConv stack with selectable decoder.

    ``dnpu_channels`` gives each physical stage's output width. For example,
    ``[16, 1]`` maps 32x32 -> 16x16 -> 8x8 and yields 64 raw readouts.
    BatchNorm layers act as digital calibration after each DNPUConv stage.
    """

    def __init__(
        self,
        processor,
        encoder_type="dnpu",
        dnpu_channels=None,
        latent_mode="raw",
        latent_dim=64,
        decoder_type="transpose",
        decoder_channels=None,
    ):
        super().__init__()

        if dnpu_channels is None:
            dnpu_channels = [16, 1]
        if decoder_channels is None:
            decoder_channels = [16, 1]
        if encoder_type not in ["hybrid", "dnpu"]:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        if latent_mode not in ["raw", "linear"]:
            raise ValueError(f"Unknown latent_mode: {latent_mode}")
        if decoder_type not in ["transpose", "zero_conv", "dnpu_zero_conv"]:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
        if len(dnpu_channels) > 5:
            raise ValueError(
                "Too many DNPU layers for 32x32 input with kernel=2, stride=2. "
                "Maximum is 5: 32 -> 16 -> 8 -> 4 -> 2 -> 1."
            )
        if decoder_type != "transpose" and latent_mode != "raw":
            raise ValueError(
                f"decoder_type={decoder_type!r} requires --latent-mode raw because "
                "the latent vector is reshaped directly into the encoder output feature map."
            )

        self.encoder_type = encoder_type
        self.dnpu_channels = list(dnpu_channels)
        self.latent_mode = latent_mode
        self.requested_latent_dim = latent_dim
        self.decoder_type = decoder_type
        self.decoder_channels = list(decoder_channels)
        self.dnpu_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        in_channels = 1
        spatial_size = 32

        if encoder_type == "hybrid":
            c1 = dnpu_channels[0]
            self.dnpu_layers.append(
                DNPUConv2d(
                    processor=processor,
                    data_input_indices=[[0, 1, 2, 3]],
                    in_channels=1,
                    out_channels=c1,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    forward_pass_type="vec",
                )
            )
            self.norm_layers.append(nn.BatchNorm2d(c1))
            self.encoder2 = nn.Sequential(
                nn.Conv2d(c1, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
            self.raw_channels = 16
            self.raw_spatial_size = 8
        else:
            for out_channels in dnpu_channels:
                self.dnpu_layers.append(
                    DNPUConv2d(
                        processor=processor,
                        data_input_indices=[[0, 1, 2, 3]],
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        forward_pass_type="vec",
                    )
                )
                self.norm_layers.append(nn.BatchNorm2d(out_channels))
                in_channels = out_channels
                spatial_size = spatial_size // 2

            self.raw_channels = in_channels
            self.raw_spatial_size = spatial_size

        self.raw_latent_dim = (
            self.raw_channels * self.raw_spatial_size * self.raw_spatial_size
        )

        # raw keeps physical readouts; linear learns an additional digital projection.
        if latent_mode == "raw":
            self.to_latent = nn.Identity()
            self.latent_dim = self.raw_latent_dim
        else:
            self.to_latent = nn.Linear(self.raw_latent_dim, latent_dim)
            self.latent_dim = latent_dim

        if decoder_type == "transpose":
            self.from_latent = nn.Linear(self.latent_dim, 16 * 8 * 8)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            )
            self.decoder_stage_shapes = ["16 x 8 x 8", "8 x 16 x 16", "1 x 32 x 32"]
        elif decoder_type == "zero_conv":
            self.from_latent = nn.Identity()
            self.decoder = DigitalZeroConvDecoder(
                raw_channels=self.raw_channels,
                raw_spatial_size=self.raw_spatial_size,
                decoder_channels=self.decoder_channels,
            )
            self.decoder_stage_shapes = self.decoder.stage_shape_strings
        else:
            self.from_latent = nn.Identity()
            self.decoder = DNPUZeroConvDecoder(
                processor=processor,
                raw_channels=self.raw_channels,
                raw_spatial_size=self.raw_spatial_size,
                decoder_channels=self.decoder_channels,
            )
            self.decoder_stage_shapes = self.decoder.stage_shape_strings

        if encoder_type == "hybrid":
            self.encoder_stage_shapes = [f"{self.dnpu_channels[0]} x 16 x 16", "16 x 8 x 8"]
        else:
            self.encoder_stage_shapes = []
            spatial_size = 32
            for out_channels in self.dnpu_channels:
                spatial_size //= 2
                self.encoder_stage_shapes.append(
                    f"{out_channels} x {spatial_size} x {spatial_size}"
                )

    def encode_features(self, x):
        h = x - 0.5
        if self.encoder_type == "hybrid":
            h = self.dnpu_layers[0](h)
            h = self.norm_layers[0](h)
            h = F.relu(h)
            return self.encoder2(h)

        for dnpu_layer, norm_layer in zip(self.dnpu_layers, self.norm_layers):
            h = dnpu_layer(h)
            h = norm_layer(h)
            h = F.relu(h)
        return h

    def forward(self, x):
        h = self.encode_features(x)
        z = self.to_latent(h.flatten(start_dim=1))
        h_dec = self.from_latent(z)
        if self.decoder_type == "transpose":
            h_dec = h_dec.reshape(x.shape[0], 16, 8, 8)
        else:
            h_dec = h_dec.reshape(
                x.shape[0],
                self.raw_channels,
                self.raw_spatial_size,
                self.raw_spatial_size,
            )
        logits = self.decoder(h_dec)
        return logits, z


class FixedRandomDecoder(nn.Module):
    """Digital decoder whose random weights can be frozen for hierarchy tests."""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.from_latent = nn.Linear(latent_dim, 16 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        h = self.from_latent(z)
        h = h.reshape(z.shape[0], 16, 8, 8)
        return self.decoder(h)


class DigitalEncoderFixedDecoder(nn.Module):
    """Digital encoder paired with the same fixed digital decoder baseline."""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.to_latent = nn.Linear(16 * 8 * 8, latent_dim)
        self.fixed_decoder = FixedRandomDecoder(latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.to_latent(h.flatten(start_dim=1))
        logits = self.fixed_decoder(z)
        return logits, z
