"""Train linear or MLP probes on frozen CIFAR encoder latents.

Probes measure class information in z without updating the physical encoder.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dnpu_ae.checkpoints import build_encoder_from_checkpoint
from dnpu_ae.cifar_data import make_grayscale_cifar_loaders
from dnpu_ae.model_utils import count_parameters


@torch.no_grad()
def encode_latent(model, x):
    if hasattr(model, "encode_features") and hasattr(model, "to_latent"):
        h = model.encode_features(x)
        z = model.to_latent(h.flatten(start_dim=1))
        return z

    # Fallback, should normally not be needed.
    _, z = model(x)
    return z


@torch.no_grad()
def cache_latents(model, loader, device):
    model.eval()

    zs = []
    ys = []

    for x, y in loader:
        x = x.to(device)
        z = encode_latent(model, x)

        zs.append(z.cpu())
        ys.append(y.cpu())

    z_all = torch.cat(zs, dim=0)
    y_all = torch.cat(ys, dim=0)

    return z_all, y_all


def make_head(head_type, latent_dim, hidden_dim):
    if head_type == "linear":
        return nn.Linear(latent_dim, 10)

    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )

    raise ValueError(f"Unknown head type: {head_type}")


@torch.no_grad()
def evaluate_head(head, loader, device):
    head.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for z, y in loader:
        z = z.to(device)
        y = y.to(device)

        logits = head(z)
        loss = F.cross_entropy(logits, y, reduction="sum")

        pred = logits.argmax(dim=1)

        total_loss += loss.item()
        total_correct += (pred == y).sum().item()
        total_examples += y.numel()

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_probe(head, train_loader, test_loader, args, device):
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    initial_train = evaluate_head(head, train_loader, device)
    initial_test = evaluate_head(head, test_loader, device)

    print(
        f"epoch {0:4d} | "
        f"train_ce {initial_train['loss']:.6f} | "
        f"train_acc {100 * initial_train['accuracy']:.2f}% | "
        f"test_ce {initial_test['loss']:.6f} | "
        f"test_acc {100 * initial_test['accuracy']:.2f}%",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        head.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for z, y in train_loader:
            z = z.to(device)
            y = y.to(device)

            logits = head(z)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                total_loss += loss.item() * y.numel()
                total_correct += (pred == y).sum().item()
                total_examples += y.numel()

        train_ce = total_loss / total_examples
        train_acc = total_correct / total_examples

        test_metrics = evaluate_head(head, test_loader, device)

        print(
            f"epoch {epoch:4d} | "
            f"train_ce {train_ce:.6f} | "
            f"train_acc {100 * train_acc:.2f}% | "
            f"test_ce {test_metrics['loss']:.6f} | "
            f"test_acc {100 * test_metrics['accuracy']:.2f}%",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--random-init", action="store_true")

    parser.add_argument("--data-dir", type=str, default="data_cifar")
    parser.add_argument("--results-dir", type=str, default="results_probe")

    parser.add_argument("--subset-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--head", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_image_loader, test_image_loader = make_grayscale_cifar_loaders(
        data_dir=args.data_dir,
        train_size=args.subset_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        train_shuffle=False,
    )

    print("CIFAR latent probe")
    print(f"  checkpoint:        {args.checkpoint}")
    print(f"  random_init:       {args.random_init}")
    print(f"  subset_size:       {args.subset_size}")
    print(f"  test_size:         {args.test_size}")
    print(f"  batch_size:        {args.batch_size}")
    print(f"  head:              {args.head}")
    print(f"  hidden_dim:        {args.hidden_dim}")
    print(f"  epochs:            {args.epochs}")
    print(f"  lr:                {args.lr}")
    print(f"  weight_decay:      {args.weight_decay}")
    print(f"  seed:              {args.seed}")
    print(f"  device:            {device}")

    encoder, ckpt, model_kind = build_encoder_from_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        random_init=args.random_init,
    )

    encoder_total, encoder_trainable = count_parameters(encoder)

    print(f"  model_kind:         {model_kind}")
    print(f"  encoder_total:      {encoder_total}")
    print(f"  encoder_trainable:  {encoder_trainable}")
    print(f"  latent_dim:         {encoder.latent_dim}")
    if hasattr(encoder, "raw_latent_dim"):
        print(f"  raw_latent_dim:     {encoder.raw_latent_dim}")
    if "args" in ckpt:
        print(f"  checkpoint_args:    {ckpt['args']}")
    print()

    print("Caching train latents...", flush=True)
    z_train, y_train = cache_latents(encoder, train_image_loader, device)
    print(f"  z_train: {tuple(z_train.shape)}")

    print("Caching test latents...", flush=True)
    z_test, y_test = cache_latents(encoder, test_image_loader, device)
    print(f"  z_test:  {tuple(z_test.shape)}")
    print()

    train_probe_loader = DataLoader(
        TensorDataset(z_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_probe_loader = DataLoader(
        TensorDataset(z_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    head = make_head(
        head_type=args.head,
        latent_dim=z_train.shape[1],
        hidden_dim=args.hidden_dim,
    ).to(device)

    head_total, head_trainable = count_parameters(head)

    print(f"  head_total:         {head_total}")
    print(f"  head_trainable:     {head_trainable}")
    print()

    train_probe(
        head=head,
        train_loader=train_probe_loader,
        test_loader=test_probe_loader,
        args=args,
        device=device,
    )

    output_path = results_dir / "latent_probe_head.pt"

    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "checkpoint": args.checkpoint,
            "random_init": args.random_init,
            "head": args.head,
            "hidden_dim": args.hidden_dim,
            "latent_dim": z_train.shape[1],
            "args": vars(args),
        },
        output_path,
    )

    print()
    print("Saved probe head:", output_path)


if __name__ == "__main__":
    main()
