from pathlib import Path

import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def get_reconstructions(model, x_input):
    """
    Return soft and thresholded reconstructions.
    """
    model.eval()
    logits, _ = model(x_input)
    probs = torch.sigmoid(logits)
    binary = (probs > 0.5).float()
    return probs, binary


def _plot_image(ax, img, title=None):
    ax.imshow(img.cpu(), vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)


@torch.no_grad()
def save_reconstruction_cases(
    model,
    x_input,
    x_target,
    n,
    out_dir,
    input_display=None,
    prefix="case",
):
    """
    Save one file per pattern.

    If input_display is None:
        panel = clean target | reconstruction | thresholded

    If input_display is provided:
        panel = clean target | displayed input | reconstruction | thresholded

    Args:
        model:
            autoencoder returning (logits, z)
        x_input:
            model input, shape (num_patterns, n*n)
        x_target:
            clean binary targets, shape (num_patterns, n*n)
        n:
            image side length
        out_dir:
            directory for individual case files
        input_display:
            optional tensor in pixel scale [0,1], same shape as x_target
        prefix:
            filename prefix
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probs, binary = get_reconstructions(model, x_input)

    num = x_target.shape[0]
    has_input_display = input_display is not None
    ncols = 4 if has_input_display else 3

    saved_paths = []

    for i in range(num):
        fig, axes = plt.subplots(1, ncols, figsize=(2.2 * ncols, 2.4), squeeze=False)
        axes = axes[0]

        col = 0
        _plot_image(axes[col], x_target[i].reshape(n, n), "Clean target")
        col += 1

        if has_input_display:
            _plot_image(axes[col], input_display[i].reshape(n, n), "Input shown")
            col += 1

        _plot_image(axes[col], probs[i].reshape(n, n), "Reconstruction")
        col += 1

        _plot_image(axes[col], binary[i].reshape(n, n), "Thresholded")

        fig.suptitle(f"Pattern {i}", y=1.05)
        plt.tight_layout()

        path = out_dir / f"{prefix}_{i:03d}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(path)

    return saved_paths


@torch.no_grad()
def plot_reconstruction_sample(
    model,
    x_input,
    x_target,
    n,
    save_path,
    input_display=None,
    num_examples=8,
    seed=0,
):
    """
    Save a compact overview figure with a random subset of patterns.

    If input_display is None:
        columns = clean target | reconstruction | thresholded

    If input_display is provided:
        columns = clean target | displayed input | reconstruction | thresholded
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    probs, binary = get_reconstructions(model, x_input)

    total = x_target.shape[0]
    num = min(num_examples, total)

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:num].tolist()

    has_input_display = input_display is not None
    ncols = 4 if has_input_display else 3

    fig, axes = plt.subplots(
        num,
        ncols,
        figsize=(2.2 * ncols, 2.1 * num),
        squeeze=False,
    )

    for row, idx in enumerate(indices):
        col = 0

        _plot_image(axes[row, col], x_target[idx].reshape(n, n), "Clean target" if row == 0 else None)
        col += 1

        if has_input_display:
            _plot_image(axes[row, col], input_display[idx].reshape(n, n), "Input shown" if row == 0 else None)
            col += 1

        _plot_image(axes[row, col], probs[idx].reshape(n, n), "Reconstruction" if row == 0 else None)
        col += 1

        _plot_image(axes[row, col], binary[idx].reshape(n, n), "Thresholded" if row == 0 else None)

        axes[row, 0].set_ylabel(str(idx), rotation=0, labelpad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return save_path, indices


# Backward-compatible wrapper used by older scripts.
@torch.no_grad()
def plot_reconstructions(
    model,
    x_input,
    x_target,
    n=4,
    max_patterns=30,
    save_path="results/reconstructions.png",
):
    """
    Legacy wrapper: save a compact sample instead of a very tall full table.
    """
    return plot_reconstruction_sample(
        model=model,
        x_input=x_input,
        x_target=x_target,
        n=n,
        save_path=save_path,
        input_display=None,
        num_examples=min(8, max_patterns),
        seed=0,
    )[0]


@torch.no_grad()
def plot_noisy_reconstructions(
    model,
    x_noisy,
    x_target,
    n=4,
    max_patterns=30,
    save_path="results/noisy_reconstructions.png",
    input_low=-0.5,
    input_high=0.5,
):
    """
    Legacy noisy wrapper: save compact noisy-input sample.
    """
    x_noisy_pixels = (x_noisy - input_low) / (input_high - input_low)
    x_noisy_pixels = x_noisy_pixels.clamp(0.0, 1.0)

    return plot_reconstruction_sample(
        model=model,
        x_input=x_noisy,
        x_target=x_target,
        n=n,
        save_path=save_path,
        input_display=x_noisy_pixels,
        num_examples=min(8, max_patterns),
        seed=0,
    )[0]
