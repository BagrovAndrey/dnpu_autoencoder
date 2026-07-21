import argparse
from pathlib import Path

import torch

from dnpu_ae.cifar_data import make_grayscale_cifar_loaders
from dnpu_ae.cifar_models import DNPUStackCIFARAutoencoder
from dnpu_ae.model_utils import (
    count_parameters,
    count_parameter_breakdown,
    freeze_batchnorm_parameters,
    freeze_dnpu_parameters,
    freeze_encoder_parameters,
    parse_channel_list,
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
        choices=["hybrid", "dnpu"],
        default="dnpu",
        help="hybrid = DNPUConv + digital Conv2d; dnpu = arbitrary DNPUConv stack.",
    )
    parser.add_argument(
        "--dnpu-channels",
        type=str,
        default="16,1",
        help="Comma-separated DNPUConv output channels, e.g. 16,1 or 16,8,4.",
    )
    parser.add_argument(
        "--decoder-type",
        choices=[
            "transpose",
            "zero_conv",
            "dnpu_zero_conv",
            "zero_conv_mixing",
            "dnpu_zero_conv_mixing",
            "nearest_conv",
            "dnpu_nearest_conv",
        ],
        default="transpose",
        help="transpose = existing ConvTranspose2d decoder; zero_conv = digital zero-insertion decoder; dnpu_zero_conv = DNPU zero-insertion decoder; *_mixing adds an extra same-size mixing convolution after each upsampling stage; nearest_conv and dnpu_nearest_conv use nearest-neighbor upsampling before each kernel-2 convolution.",
    )
    parser.add_argument(
        "--decoder-channels",
        type=str,
        default="16,1",
        help="Comma-separated zero-insertion decoder output channels, e.g. 16,1.",
    )

    parser.add_argument(
        "--latent-mode",
        choices=["raw", "linear"],
        default="raw",
        help="raw = use flattened encoder output as latent; linear = apply Linear(raw_dim -> latent_dim).",
    )
    parser.add_argument("--latent-dim", type=int, default=64)

    parser.add_argument(
        "--loss",
        choices=["bce", "mse", "l1", "bce_l1"],
        default="l1",
        help="Reconstruction loss used for optimization.",
    )

    parser.add_argument(
        "--freeze-dnpu",
        action="store_true",
        help="Freeze DNPUConv trainable control voltages.",
    )
    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        help="Freeze BatchNorm affine parameters while keeping DNPU controls trainable.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the full encoder up to the latent representation. Decoder remains trainable.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.freeze_encoder and (args.freeze_dnpu or args.freeze_bn):
        raise ValueError(
            "Use --freeze-encoder alone, not together with --freeze-dnpu or --freeze-bn."
        )

    if args.freeze_dnpu and args.freeze_bn:
        raise ValueError("Use either --freeze-dnpu or --freeze-bn, not both.")

    dnpu_channels = parse_channel_list(args.dnpu_channels, arg_name="--dnpu-channels")
    decoder_channels = parse_channel_list(
        args.decoder_channels,
        arg_name="--decoder-channels",
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

    model = DNPUStackCIFARAutoencoder(
        processor=processor,
        encoder_type=args.encoder_type,
        dnpu_channels=dnpu_channels,
        latent_mode=args.latent_mode,
        latent_dim=args.latent_dim,
        decoder_type=args.decoder_type,
        decoder_channels=decoder_channels,
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
    total_breakdown = count_parameter_breakdown(model, trainable_only=False)
    trainable_breakdown = count_parameter_breakdown(model, trainable_only=True)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("DNPU stack CIFAR autoencoder")
    print(f"  subset_size:       {args.subset_size}")
    print(f"  batch_size:        {args.batch_size}")
    print(f"  encoder_type:      {args.encoder_type}")
    print(f"  dnpu_channels:     {dnpu_channels}")
    print(f"  raw_shape:         {model.raw_channels} x {model.raw_spatial_size} x {model.raw_spatial_size}")
    print(f"  latent_mode:       {args.latent_mode}")
    print(f"  latent_dim:        {model.latent_dim}")
    print(f"  raw_latent_dim:    {model.raw_latent_dim}")
    print(f"  decoder_type:      {args.decoder_type}")
    print(f"  decoder_channels:  {decoder_channels}")
    print(f"  encoder_stages:    {' -> '.join(model.encoder_stage_shapes)}")
    print(f"  decoder_stages:    {' -> '.join(model.decoder_stage_shapes)}")
    print(
        "  full_path:         "
        + " -> ".join(["1 x 32 x 32"] + model.encoder_stage_shapes + model.decoder_stage_shapes[1:])
    )
    print(f"  loss:              {args.loss}")
    print(f"  freeze_dnpu:       {args.freeze_dnpu}")
    print(f"  frozen DNPU pars:  {frozen_dnpu_params}")
    print(f"  freeze_bn:         {args.freeze_bn}")
    print(f"  frozen BN pars:    {frozen_bn_params}")
    print(f"  freeze_encoder:    {args.freeze_encoder}")
    print(f"  frozen enc pars:   {frozen_encoder_params}")
    print(f"  total params:      {total_params}")
    print(f"  trainable params:  {trainable_params}")
    print(f"  enc DNPU total:    {total_breakdown['encoder_dnpu_controls']}")
    print(f"  dec DNPU total:    {total_breakdown['decoder_dnpu_controls']}")
    print(f"  enc BN total:      {total_breakdown['encoder_batchnorm']}")
    print(f"  dec BN total:      {total_breakdown['decoder_batchnorm']}")
    print(f"  other digital:     {total_breakdown['other_digital']}")
    print(f"  enc DNPU train:    {trainable_breakdown['encoder_dnpu_controls']}")
    print(f"  dec DNPU train:    {trainable_breakdown['decoder_dnpu_controls']}")
    print(f"  enc BN train:      {trainable_breakdown['encoder_batchnorm']}")
    print(f"  dec BN train:      {trainable_breakdown['decoder_batchnorm']}")
    print(f"  other dig train:   {trainable_breakdown['other_digital']}")
    print(f"  device:            {device}")
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

        path = save_reconstruction_sample(
            model=model,
            loader=test_loader,
            device=device,
            save_path=results_dir / f"cifar_recon_epoch{epoch:04d}.png",
            n=8,
        )
        print("  saved:", path)

    checkpoint_path = results_dir / "cifar_dnpu_stack_autoencoder.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "dnpu_channels": dnpu_channels,
            "raw_channels": model.raw_channels,
            "raw_spatial_size": model.raw_spatial_size,
            "raw_latent_dim": model.raw_latent_dim,
            "latent_dim": model.latent_dim,
            "decoder_type": model.decoder_type,
            "decoder_channels": model.decoder_channels,
        },
        checkpoint_path,
    )

    print("\nSaved checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
