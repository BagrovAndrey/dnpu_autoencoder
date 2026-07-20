"""Zero-insertion upsampling helpers and CIFAR decoders."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_insert_upsample_2d(x, scale_factor=2):
    """Upsample by placing input values at even coordinates and zeros elsewhere."""
    if x.ndim != 4:
        raise ValueError(
            f"zero_insert_upsample_2d expects a 4D tensor, got shape {tuple(x.shape)}"
        )
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")

    batch, channels, height, width = x.shape
    out = x.new_zeros(batch, channels, height * scale_factor, width * scale_factor)
    out[..., ::scale_factor, ::scale_factor] = x
    return out


def parse_decoder_stage_specs(raw_spatial_size, raw_channels, decoder_channels, target_size=32):
    """Validate zero-conv decoder arithmetic and return per-stage metadata."""
    if raw_spatial_size <= 0:
        raise ValueError(f"raw_spatial_size must be positive, got {raw_spatial_size}")
    if raw_channels <= 0:
        raise ValueError(f"raw_channels must be positive, got {raw_channels}")
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if len(decoder_channels) == 0:
        raise ValueError("--decoder-channels must contain at least one stage")
    if any(c <= 0 for c in decoder_channels):
        raise ValueError("--decoder-channels must contain only positive integers")

    if target_size % raw_spatial_size != 0:
        raise ValueError(
            f"Raw spatial size {raw_spatial_size} must divide target size {target_size}."
        )

    upsample_ratio = target_size // raw_spatial_size
    if upsample_ratio & (upsample_ratio - 1):
        raise ValueError(
            f"Upsampling ratio {upsample_ratio} from {raw_spatial_size} to {target_size} "
            "must be a power of two."
        )

    required_stages = int(math.log2(upsample_ratio))
    if len(decoder_channels) != required_stages:
        raise ValueError(
            f"--decoder-channels specifies {len(decoder_channels)} stages, but "
            f"{raw_spatial_size} -> {target_size} requires {required_stages} factor-two "
            "upsampling stages."
        )
    if decoder_channels[-1] != 1:
        raise ValueError(
            f"The last decoder channel must be 1, got {decoder_channels[-1]}."
        )

    stage_specs = []
    spatial_size = raw_spatial_size
    in_channels = raw_channels

    for out_channels in decoder_channels:
        out_spatial_size = spatial_size * 2
        stage_specs.append(
            {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "in_spatial_size": spatial_size,
                "upsampled_spatial_size": out_spatial_size,
                "padded_spatial_size": out_spatial_size + 1,
                "out_spatial_size": out_spatial_size,
            }
        )
        in_channels = out_channels
        spatial_size = out_spatial_size

    return stage_specs


def _format_shape(channels, spatial_size):
    return f"{channels} x {spatial_size} x {spatial_size}"


def _pad_kernel2_same_size(x, pad_mode):
    """Pad by one pixel total for a kernel-2 stride-1 layer without fixing one phase."""
    if pad_mode == "bottom_right":
        return F.pad(x, (0, 1, 0, 1))
    if pad_mode == "top_left":
        return F.pad(x, (1, 0, 1, 0))
    raise ValueError(f"Unknown pad_mode: {pad_mode}")


class _ZeroConvDecoderBase(nn.Module):
    """Shared logic for digital and DNPU zero-insertion decoders."""

    def __init__(self, raw_channels, raw_spatial_size, decoder_channels, use_mixing=False):
        super().__init__()
        self.raw_channels = raw_channels
        self.raw_spatial_size = raw_spatial_size
        self.decoder_channels = list(decoder_channels)
        self.use_mixing = use_mixing
        self.stage_specs = parse_decoder_stage_specs(
            raw_spatial_size=raw_spatial_size,
            raw_channels=raw_channels,
            decoder_channels=self.decoder_channels,
            target_size=32,
        )
        self.norm_layers = nn.ModuleList(
            [nn.BatchNorm2d(spec["out_channels"]) for spec in self.stage_specs]
        )
        if self.use_mixing:
            self.mixing_norm_layers = nn.ModuleList(
                [nn.BatchNorm2d(spec["out_channels"]) for spec in self.stage_specs[:-1]]
            )

    def _make_conv_layer(self, in_channels, out_channels):
        raise NotImplementedError

    def _stage_pad_mode(self, idx):
        return "bottom_right" if idx % 2 == 1 else "top_left"

    def _mixing_pad_mode(self, idx):
        return "top_left" if idx % 2 == 1 else "bottom_right"

    def _build_layers(self):
        self.layers = nn.ModuleList(
            [
                self._make_conv_layer(spec["in_channels"], spec["out_channels"])
                for spec in self.stage_specs
            ]
        )
        if self.use_mixing:
            self.mixing_layers = nn.ModuleList(
                [
                    self._make_conv_layer(spec["out_channels"], spec["out_channels"])
                    for spec in self.stage_specs
                ]
            )

    @property
    def stage_shape_strings(self):
        shapes = [_format_shape(self.raw_channels, self.raw_spatial_size)]
        for spec in self.stage_specs:
            shapes.append(_format_shape(spec["out_channels"], spec["out_spatial_size"]))
        return shapes

    def forward(self, x):
        expected_input_shape = (
            x.shape[0],
            self.raw_channels,
            self.raw_spatial_size,
            self.raw_spatial_size,
        )
        if tuple(x.shape) != expected_input_shape:
            raise ValueError(
                "Decoder expected raw latent feature map shape "
                f"{expected_input_shape}, got {tuple(x.shape)}."
            )

        h = x
        for idx, (layer, norm, spec) in enumerate(
            zip(self.layers, self.norm_layers, self.stage_specs),
            start=1,
        ):
            h = zero_insert_upsample_2d(h, scale_factor=2)
            expected_up_shape = (
                x.shape[0],
                spec["in_channels"],
                spec["upsampled_spatial_size"],
                spec["upsampled_spatial_size"],
            )
            if tuple(h.shape) != expected_up_shape:
                raise RuntimeError(
                    f"Decoder stage {idx} zero insertion produced {tuple(h.shape)}, "
                    f"expected {expected_up_shape}."
                )

            h = _pad_kernel2_same_size(h, self._stage_pad_mode(idx))
            expected_padded_shape = (
                x.shape[0],
                spec["in_channels"],
                spec["padded_spatial_size"],
                spec["padded_spatial_size"],
            )
            if tuple(h.shape) != expected_padded_shape:
                raise RuntimeError(
                    f"Decoder stage {idx} padding produced {tuple(h.shape)}, "
                    f"expected {expected_padded_shape}."
                )

            h = layer(h)
            h = norm(h)

            expected_out_shape = (
                x.shape[0],
                spec["out_channels"],
                spec["out_spatial_size"],
                spec["out_spatial_size"],
            )
            if tuple(h.shape) != expected_out_shape:
                raise RuntimeError(
                    f"Decoder stage {idx} convolution produced {tuple(h.shape)}, "
                    f"expected {expected_out_shape}."
                )

            if idx < len(self.stage_specs):
                h = F.relu(h)
                if self.use_mixing:
                    residual = h
                    h = _pad_kernel2_same_size(h, self._mixing_pad_mode(idx))
                    expected_mixing_padded_shape = (
                        x.shape[0],
                        spec["out_channels"],
                        spec["out_spatial_size"] + 1,
                        spec["out_spatial_size"] + 1,
                    )
                    if tuple(h.shape) != expected_mixing_padded_shape:
                        raise RuntimeError(
                            f"Decoder stage {idx} mixing padding produced {tuple(h.shape)}, "
                            f"expected {expected_mixing_padded_shape}."
                        )

                    mixed = self.mixing_layers[idx - 1](h)
                    mixed = self.mixing_norm_layers[idx - 1](mixed)

                    expected_mixing_shape = (
                        x.shape[0],
                        spec["out_channels"],
                        spec["out_spatial_size"],
                        spec["out_spatial_size"],
                    )
                    if tuple(mixed.shape) != expected_mixing_shape:
                        raise RuntimeError(
                            f"Decoder stage {idx} mixing convolution produced {tuple(mixed.shape)}, "
                            f"expected {expected_mixing_shape}."
                        )

                    h = F.relu(residual + mixed)
            elif self.use_mixing:
                residual = h
                h = _pad_kernel2_same_size(h, self._mixing_pad_mode(idx))
                expected_mixing_padded_shape = (
                    x.shape[0],
                    spec["out_channels"],
                    spec["out_spatial_size"] + 1,
                    spec["out_spatial_size"] + 1,
                )
                if tuple(h.shape) != expected_mixing_padded_shape:
                    raise RuntimeError(
                        f"Decoder stage {idx} final mixing padding produced {tuple(h.shape)}, "
                        f"expected {expected_mixing_padded_shape}."
                    )

                mixed = self.mixing_layers[idx - 1](h)

                expected_mixing_shape = (
                    x.shape[0],
                    spec["out_channels"],
                    spec["out_spatial_size"],
                    spec["out_spatial_size"],
                )
                if tuple(mixed.shape) != expected_mixing_shape:
                    raise RuntimeError(
                        f"Decoder stage {idx} final mixing convolution produced {tuple(mixed.shape)}, "
                        f"expected {expected_mixing_shape}."
                    )

                h = residual + mixed

        return h


class DigitalZeroConvDecoder(_ZeroConvDecoderBase):
    """Zero-insertion upsampling decoder built from ordinary Conv2d layers."""

    def __init__(self, raw_channels, raw_spatial_size, decoder_channels, use_mixing=False):
        super().__init__(
            raw_channels=raw_channels,
            raw_spatial_size=raw_spatial_size,
            decoder_channels=decoder_channels,
            use_mixing=use_mixing,
        )
        self._build_layers()

    def _make_conv_layer(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=1,
            padding=0,
        )


class DNPUZeroConvDecoder(_ZeroConvDecoderBase):
    """Zero-insertion upsampling decoder built from DNPUConv2d layers."""

    def __init__(
        self,
        processor,
        raw_channels,
        raw_spatial_size,
        decoder_channels,
        use_mixing=False,
    ):
        super().__init__(
            raw_channels=raw_channels,
            raw_spatial_size=raw_spatial_size,
            decoder_channels=decoder_channels,
            use_mixing=use_mixing,
        )
        self.processor = processor
        self._build_layers()

    def _make_conv_layer(self, in_channels, out_channels):
        from brainspy.processors.modules.conv import DNPUConv2d

        return DNPUConv2d(
            processor=self.processor,
            data_input_indices=[[0, 1, 2, 3]],
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=1,
            padding=0,
            forward_pass_type="vec",
        )
