"""Small model-management helpers shared by CIFAR experiments."""

import torch
import torch.nn as nn


def parse_channel_list(value):
    """Parse comma-separated DNPU stage widths such as ``16,1`` or ``8,4,4``."""
    channels = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(channels) == 0:
        raise ValueError("--dnpu-channels must contain at least one integer")
    if any(c <= 0 for c in channels):
        raise ValueError("--dnpu-channels must contain only positive integers")
    return channels


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def freeze_dnpu_parameters(model):
    """Freeze hardware-like DNPU control parameters, leaving digital layers trainable."""
    frozen_trainable = 0
    for name, param in model.named_parameters():
        if ("dnpu_conv" in name or "dnpu_layers" in name) and param.requires_grad:
            frozen_trainable += param.numel()
            param.requires_grad = False
    return frozen_trainable


def freeze_batchnorm_parameters(model):
    """Freeze BatchNorm affine calibration parameters without changing mode handling."""
    frozen_trainable = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                if param.requires_grad:
                    frozen_trainable += param.numel()
                    param.requires_grad = False
    return frozen_trainable


def freeze_encoder_parameters(model):
    """Freeze the encoder through its raw/linear latent representation."""
    if hasattr(model, "dnpu_layers"):
        encoder_prefixes = ("dnpu_layers", "norm_layers", "encoder2", "to_latent")
    else:
        encoder_prefixes = (
            "dnpu_conv1",
            "norm1",
            "encoder2",
            "dnpu_conv2",
            "norm2",
            "to_latent",
        )

    frozen_trainable = 0
    for name, param in model.named_parameters():
        if name.startswith(encoder_prefixes) and param.requires_grad:
            frozen_trainable += param.numel()
            param.requires_grad = False
    return frozen_trainable


def freeze_module(module):
    frozen_trainable = 0
    for param in module.parameters():
        if param.requires_grad:
            frozen_trainable += param.numel()
            param.requires_grad = False
    return frozen_trainable


def reset_module_parameters(module, seed):
    """Reset a module deterministically without advancing the caller's RNG state."""
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()
    torch.random.set_rng_state(state)


def freeze_dnpu_decoder(model):
    frozen_trainable = 0
    for name, param in model.named_parameters():
        if name.startswith("from_latent") or name.startswith("decoder"):
            if param.requires_grad:
                frozen_trainable += param.numel()
                param.requires_grad = False
    return frozen_trainable


def reinitialize_dnpu_decoder(model, decoder_seed):
    state = torch.random.get_rng_state()
    torch.manual_seed(decoder_seed)
    for child in model.from_latent.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()
    for child in model.decoder.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()
    torch.random.set_rng_state(state)

