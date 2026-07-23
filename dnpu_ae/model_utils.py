"""Small model-management helpers shared by CIFAR experiments."""

from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn


def parse_channel_list(value, arg_name="--dnpu-channels"):
    """Parse comma-separated stage widths such as ``16,1`` or ``8,4,4``."""
    channels = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(channels) == 0:
        raise ValueError(f"{arg_name} must contain at least one integer")
    if any(c <= 0 for c in channels):
        raise ValueError(f"{arg_name} must contain only positive integers")
    return channels


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def freeze_dnpu_parameters(model):
    """Freeze hardware-like DNPU control parameters, leaving digital layers trainable."""
    frozen_trainable = 0
    for name, param in model.named_parameters():
        if (
            "dnpu_conv" in name
            or name.startswith("dnpu_layers")
            or (
                name.startswith("decoder.layers")
                and getattr(model, "decoder_type", None) in (
                    "dnpu_zero_conv",
                    "dnpu_zero_conv_mixing",
                    "dnpu_nearest_conv",
                    "dnpu_nearest_conv_linear",
                )
            )
            or (
                name.startswith("decoder.mixing_layers")
                and getattr(model, "decoder_type", None) == "dnpu_zero_conv_mixing"
            )
        ) and param.requires_grad:
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


def _batchnorm_parameter_names(model):
    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        for param_name, _ in module.named_parameters(recurse=False):
            names.add(f"{module_name}.{param_name}" if module_name else param_name)
    return names


def count_parameter_breakdown(model):
    """Return grouped parameter counts with DNPU controls separated from surrogate weights."""
    counts = OrderedDict(
        [
            ("processor_surrogate_frozen", {"total": 0, "trainable": 0}),
            ("encoder_dnpu_controls", {"total": 0, "trainable": 0}),
            ("decoder_dnpu_controls", {"total": 0, "trainable": 0}),
            ("encoder_batchnorm", {"total": 0, "trainable": 0}),
            ("decoder_batchnorm", {"total": 0, "trainable": 0}),
            ("other_digital", {"total": 0, "trainable": 0}),
        ]
    )

    batchnorm_names = _batchnorm_parameter_names(model)

    for name, param in model.named_parameters():
        if name.endswith("control_voltages"):
            key = "decoder_dnpu_controls" if name.startswith("decoder.") else "encoder_dnpu_controls"
        elif ".processor." in name or name.startswith("processor."):
            key = "processor_surrogate_frozen"
        elif name in batchnorm_names:
            key = "decoder_batchnorm" if name.startswith("decoder.") else "encoder_batchnorm"
        else:
            key = "other_digital"

        counts[key]["total"] += param.numel()
        if param.requires_grad:
            counts[key]["trainable"] += param.numel()

    totals = defaultdict(int)
    for group_counts in counts.values():
        totals["total"] += group_counts["total"]
        totals["trainable"] += group_counts["trainable"]
    counts["registered_parameters_total"] = totals["total"]
    counts["trainable_parameters_total"] = totals["trainable"]

    return counts
