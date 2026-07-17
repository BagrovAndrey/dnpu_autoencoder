"""Reconstruct CIFAR encoders from experiment checkpoints for latent probes."""

import torch

from dnpu_ae.cifar_models import (
    DNPUConvCIFARAutoencoder,
    DNPUStackCIFARAutoencoder,
)
from dnpu_ae.model_utils import parse_channel_list
from dnpu_ae.processor import make_processor


def get_arg(args_dict, name, default):
    if args_dict is None:
        return default
    return args_dict.get(name, default)


def parse_maybe_channel_list(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, tuple):
        return [int(x) for x in value]
    if isinstance(value, str):
        return parse_channel_list(value)
    raise ValueError(f"Cannot parse channel list from {value!r}")


def build_encoder_from_checkpoint(checkpoint_path, device, random_init=False):
    """Build and freeze the legacy or stack encoder described by a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args_dict = ckpt.get("args", {})
    processor = make_processor()

    is_stack_checkpoint = (
        "dnpu_channels" in ckpt
        or "dnpu_channels" in args_dict
        or "raw_spatial_size" in ckpt
    )

    if is_stack_checkpoint:
        dnpu_channels = ckpt.get(
            "dnpu_channels",
            parse_maybe_channel_list(get_arg(args_dict, "dnpu_channels", "16,1")),
        )
        model = DNPUStackCIFARAutoencoder(
            processor=processor,
            encoder_type=get_arg(args_dict, "encoder_type", "dnpu"),
            dnpu_channels=dnpu_channels,
            latent_mode=get_arg(args_dict, "latent_mode", "raw"),
            latent_dim=int(get_arg(args_dict, "latent_dim", ckpt.get("latent_dim", 64))),
            decoder_type=get_arg(args_dict, "decoder_type", ckpt.get("decoder_type", "transpose")),
            decoder_channels=ckpt.get(
                "decoder_channels",
                parse_maybe_channel_list(get_arg(args_dict, "decoder_channels", "16,1")),
            ),
        )
        model_kind = "stack"
    else:
        model = DNPUConvCIFARAutoencoder(
            processor=processor,
            encoder_type=get_arg(args_dict, "encoder_type", "hybrid"),
            conv_channels=int(get_arg(args_dict, "conv_channels", 8)),
            conv2_channels=int(get_arg(args_dict, "conv2_channels", 1)),
            latent_mode=get_arg(args_dict, "latent_mode", "linear"),
            latent_dim=int(get_arg(args_dict, "latent_dim", 64)),
        )
        model_kind = "legacy"

    if not random_init:
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"],
            strict=False,
        )
        if missing:
            print("WARNING: missing keys while loading checkpoint:")
            for key in missing:
                print("  ", key)
        if unexpected:
            print("WARNING: unexpected keys while loading checkpoint:")
            for key in unexpected:
                print("  ", key)

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, ckpt, model_kind
