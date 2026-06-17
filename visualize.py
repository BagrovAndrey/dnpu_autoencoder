import torch
import matplotlib.pyplot as plt


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
    Plot input / reconstructed / thresholded reconstruction table.

    Args:
        model:
            autoencoder returning (logits, z)
        x_input:
            model input, shape (num_patterns, 16), usually voltage-encoded
        x_target:
            binary target images, shape (num_patterns, 16), values 0/1
        n:
            image side length
        max_patterns:
            maximum number of patterns to plot
        save_path:
            output image path
    """
    model.eval()

    logits, _ = model(x_input)
    probs = torch.sigmoid(logits)
    binary = (probs > 0.5).float()

    num = min(max_patterns, x_target.shape[0])

    fig, axes = plt.subplots(
        num,
        3,
        figsize=(6, 1.6 * num),
        squeeze=False,
    )

    for i in range(num):
        imgs = [
            x_target[i].reshape(n, n),
            probs[i].reshape(n, n),
            binary[i].reshape(n, n),
        ]
        titles = ["Input", "Reconstruction", "Thresholded"]

        for j in range(3):
            ax = axes[i, j]
            ax.imshow(imgs[j].cpu(), vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(titles[j])

        axes[i, 0].set_ylabel(str(i), rotation=0, labelpad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


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
    Plot clean target / noisy input / reconstruction / thresholded table.

    Args:
        model:
            autoencoder returning (logits, z)
        x_noisy:
            noisy model input in voltage units, shape (num_patterns, 16)
        x_target:
            clean binary target images, shape (num_patterns, 16), values 0/1
        n:
            image side length
        max_patterns:
            maximum number of patterns to plot
        save_path:
            output image path
        input_low, input_high:
            voltage encoding range used to map noisy input back to pixel scale
            for visualization only.
    """
    model.eval()

    logits, _ = model(x_noisy)
    probs = torch.sigmoid(logits)
    binary = (probs > 0.5).float()

    # Convert noisy voltages back to pixel scale for display.
    x_noisy_pixels = (x_noisy - input_low) / (input_high - input_low)
    x_noisy_pixels = x_noisy_pixels.clamp(0.0, 1.0)

    num = min(max_patterns, x_target.shape[0])

    fig, axes = plt.subplots(
        num,
        4,
        figsize=(8, 1.6 * num),
        squeeze=False,
    )

    for i in range(num):
        imgs = [
            x_target[i].reshape(n, n),
            x_noisy_pixels[i].reshape(n, n),
            probs[i].reshape(n, n),
            binary[i].reshape(n, n),
        ]
        titles = ["Clean target", "Noisy input", "Reconstruction", "Thresholded"]

        for j in range(4):
            ax = axes[i, j]
            ax.imshow(imgs[j].cpu(), vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(titles[j])

        axes[i, 0].set_ylabel(str(i), rotation=0, labelpad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path
