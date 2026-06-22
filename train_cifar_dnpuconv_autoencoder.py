import argparse
from pathlib import Path

import torch

from dnpu_ae.cifar_data import make_grayscale_cifar_loaders
from dnpu_ae.cifar_models import DNPUConvCIFARAutoencoder
from dnpu_ae.model_utils import (
    count_parameters,
    freeze_batchnorm_parameters,
    freeze_dnpu_parameters,
    freeze_encoder_parameters,
)
from dnpu_ae.processor import make_processor
from dnpu_ae.reconstruction import (
    evaluate_reconstruction as evaluate,
    reconstruction_loss,
    save_reconstruction_sample,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data_cifar")
    parser.add_argument("--results-dir", type=str, default="results_cifar")

    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument(
        "--encoder-type",
        choices=["hybrid", "dnpu2"],
        default="hybrid",
        help="Encoder type: hybrid = DNPUConv + digital Conv2d; dnpu2 = two DNPUConv stages.",
    )
    parser.add_argument("--conv-channels", type=int, default=8)
    parser.add_argument(
        "--conv2-channels",
        type=int,
        default=1,
        help="Number of output channels in the second DNPUConv stage. Used only for --encoder-type dnpu2.",
    )

    parser.add_argument(
        "--latent-mode",
        choices=["raw", "linear"],
        default="linear",
        help="raw = use flattened encoder output as latent; linear = apply Linear(raw_dim -> latent_dim).",
    )
    parser.add_argument("--latent-dim", type=int, default=64)

    parser.add_argument(
        "--loss",
        choices=["bce", "mse", "l1", "bce_l1"],
        default="bce",
        help="Reconstruction loss used for optimization.",
    )

    parser.add_argument(
        "--freeze-dnpu",
        action="store_true",
        help="Freeze DNPUConv trainable control voltages. Digital layers remain trainable.",
    )

    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help=(
            "Freeze the full encoder up to the latent representation. "
            "The decoder remains trainable."
        ),
    )

    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        help="Freeze BatchNorm affine parameters while keeping DNPU controls trainable.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    #if args.freeze_dnpu and args.freeze_encoder:
    #    raise ValueError("Use either --freeze-dnpu or --freeze-encoder, not both.")

    if args.freeze_encoder and (args.freeze_dnpu or args.freeze_bn):
        raise ValueError(
            "Use --freeze-encoder alone, not together with --freeze-dnpu or --freeze-bn."
        )

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = make_grayscale_cifar_loaders(
        data_dir=args.data_dir,
        train_size=args.subset_size,
        test_size=args.subset_size,
        batch_size=args.batch_size,
        train_shuffle=True,
    )

    processor = make_processor()

    model = DNPUConvCIFARAutoencoder(
        processor=processor,
        encoder_type=args.encoder_type,
        conv_channels=args.conv_channels,
        conv2_channels=args.conv2_channels,
        latent_mode=args.latent_mode,
        latent_dim=args.latent_dim,
    ).to(device)

    frozen_dnpu_params = 0
    frozen_bn_params = 0
    frozen_encoder_params = 0

    if args.freeze_dnpu:
        frozen_dnpu_params = freeze_dnpu_parameters(model)

    if args.freeze_bn:
        frozen_bn_params = freeze_batchnorm_parameters(model)

    if args.freeze_encoder:
        frozen_encoder_params = freeze_encoder_parameters(model)

    total_params, trainable_params = count_parameters(model)

    #frozen_dnpu_params = 0
    #if args.freeze_dnpu:
    #    frozen_dnpu_params = freeze_dnpu_parameters(model)

    #total_params, trainable_params = count_parameters(model)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("DNPUConv CIFAR autoencoder")
    print(f"  subset_size:      {args.subset_size}")
    print(f"  batch_size:       {args.batch_size}")
    print(f"  encoder_type:     {args.encoder_type}")
    print(f"  conv_channels:    {args.conv_channels}")
    print(f"  conv2_channels:   {args.conv2_channels}")
    print(f"  latent_mode:      {args.latent_mode}")
    print(f"  latent_dim:       {model.latent_dim}")
    print(f"  raw_latent_dim:   {model.raw_latent_dim}")
    print(f"  loss:             {args.loss}")
    print(f"  freeze_dnpu:      {args.freeze_dnpu}")
    print(f"  frozen DNPU pars: {frozen_dnpu_params}")
    print(f"  freeze_bn:        {args.freeze_bn}")
    print(f"  frozen BN pars:   {frozen_bn_params}")
    print(f"  freeze_encoder:   {args.freeze_encoder}")
    print(f"  frozen enc pars:  {frozen_encoder_params}")
    print(f"  total params:     {total_params}")
    print(f"  trainable params: {trainable_params}")
    print(f"  device:           {device}")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        running_pixels = 0

        for x, _ in train_loader:
            x = x.to(device)

            logits, _ = model(x)
            loss = reconstruction_loss(logits, x, args.loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.numel()
            running_pixels += x.numel()

        train_loss = running_loss / running_pixels

        test_metrics = evaluate(
            model,
            test_loader,
            device,
            loss_type=args.loss,
            max_batches=20,
        )

        print(
            f"epoch {epoch:4d} | "
            f"train_loss {train_loss:.6f} | "
            f"test_loss {test_metrics['loss_per_pixel']:.6f} | "
            f"test_bce {test_metrics['bce_per_pixel']:.6f} | "
            f"test_mse {test_metrics['mse_per_pixel']:.6f} | "
            f"test_mae {test_metrics['mae_per_pixel']:.6f}"
        )

        if epoch == 1 or epoch == args.epochs:
            path = save_reconstruction_sample(
                model=model,
                loader=test_loader,
                device=device,
                save_path=results_dir / f"cifar_recon_epoch{epoch:04d}.png",
                n=8,
            )
            print("  saved:", path)

    checkpoint_path = results_dir / "cifar_dnpuconv_autoencoder.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "raw_latent_dim": model.raw_latent_dim,
            "latent_dim": model.latent_dim,
        },
        checkpoint_path,
    )

    print("\nSaved checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
