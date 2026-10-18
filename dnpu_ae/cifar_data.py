"""Grayscale CIFAR-10 data loading shared by CIFAR experiments."""

from pathlib import Path
import random

from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms


CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _load_image_folder(root, transform):
    dataset = torchvision.datasets.ImageFolder(root, transform=transform)
    if set(dataset.classes) != set(CIFAR_CLASSES):
        raise ValueError(f"Expected the ten CIFAR-10 class folders under {root}.")
    class_to_idx = {name: i for i, name in enumerate(CIFAR_CLASSES)}
    dataset.samples = [
        (path, class_to_idx[dataset.classes[label]])
        for path, label in dataset.samples
    ]
    # ImageFolder groups by class. Shuffle once so small prefix subsets cover
    # multiple classes, independently of training's RNG and shuffle setting.
    random.Random(0).shuffle(dataset.samples)
    dataset.imgs = dataset.samples
    dataset.targets = [label for _, label in dataset.samples]
    dataset.classes = list(CIFAR_CLASSES)
    dataset.class_to_idx = class_to_idx
    return dataset


def make_grayscale_cifar_subsets(data_dir, train_size, test_size):
    """Load original CIFAR batches or extracted ``cifar10/{train,test}/class`` PNGs.

    Folder samples use a fixed shuffle before prefix selection; their ordering
    differs from the original torchvision CIFAR-10 batch ordering.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    folder_root = Path(data_dir) / "cifar10"
    if (folder_root / "train").is_dir() and (folder_root / "test").is_dir():
        train_full = _load_image_folder(folder_root / "train", transform)
        test_full = _load_image_folder(folder_root / "test", transform)
    else:
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

