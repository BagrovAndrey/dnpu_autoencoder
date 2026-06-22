"""Grayscale CIFAR-10 data loading shared by CIFAR experiments."""

from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms


def make_grayscale_cifar_subsets(data_dir, train_size, test_size):
    """Return deterministic prefix subsets with the original transform semantics."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_full = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_full = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_subset = Subset(train_full, list(range(train_size)))
    test_subset = Subset(test_full, list(range(min(test_size, len(test_full)))))
    return train_subset, test_subset


def make_grayscale_cifar_loaders(
    data_dir,
    train_size,
    test_size,
    batch_size,
    *,
    train_shuffle=True,
):
    """Build CIFAR loaders while preserving prefix subsets and ``num_workers=0``."""
    train_subset, test_subset = make_grayscale_cifar_subsets(
        data_dir=data_dir,
        train_size=train_size,
        test_size=test_size,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader

