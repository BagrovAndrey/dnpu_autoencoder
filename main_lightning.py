"""
PyTorch Lightning entry point for the DNPU autoencoder project.

Consolidates what used to be four separate scripts:
    train_hybrid.py         -> train on all Bars & Stripes patterns
    train_frozen_encoder.py -> same, with --freeze-encoder
    train_generalization.py -> train/test split via --train-size
    test_noise_robustness.py -> voltage-noise sweep via --noise-sigmas

Usage examples:
    python main_lightning.py
    python main_lightning.py --freeze-encoder
    python main_lightning.py --train-size 15
    python main_lightning.py --noise-sigmas 0.0 0.01 0.02 0.05 0.1 0.2
    python main_lightning.py --baseline --latent-dim 4
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from brainspy.processors.processor import Processor

# importing the brains-py DNPU class -- necessary to make sure surrogate model 
# weights are properly frozen, and only control voltages are the learnable params.
from brainspy.processors.dnpu import DNPU


from data import make_bars_stripes, pixels_to_voltages
from models import PlanarDNPUEncoderDigitalDecoder, DigitalBaselineAutoencoder
from visualize import (
    save_reconstruction_cases,
    plot_reconstruction_sample,
    plot_noisy_reconstructions,
)


def make_processor(checkpoint_path):
    ckpt = torch.load(Path(checkpoint_path), weights_only = False)

    processor = Processor(
        configs={
            "processor_type": "simulation",
            "waveform": {
                "plateau_length": 1,
                "slope_length": 0,
            },
        },
        info=ckpt["info"],
        model_state_dict=ckpt["model_state_dict"],
    )

    voltage_ranges = ckpt["info"]["electrode_info"]["activation_electrodes"]["voltage_ranges"]
    return processor, voltage_ranges


def reconstruction_accuracy(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    pixel_acc = (pred == target).float().mean()
    pattern_acc = (pred == target).all(dim=1).float().mean()
    return pixel_acc, pattern_acc


def add_voltage_noise(x_volt, sigma, low, high):
    noisy = x_volt + sigma * torch.randn_like(x_volt)
    return noisy.clamp(low, high)


class BarsStripesDataModule(pl.LightningDataModule):
    """
    Full-batch Bars & Stripes datamodule.

    train_size=None:
        train/val/test all use the full pattern set (matches train_hybrid.py).
    train_size=k:
        random train/test split (matches train_generalization.py); val reuses
        the test split so held-out metrics are visible during training.
    """

    def __init__(self, image_size=4, v_low=-0.5, v_high=0.5, train_size=None, seed=0):
        super().__init__()
        self.image_size = image_size
        self.v_low = v_low
        self.v_high = v_high
        self.train_size = train_size
        self.seed = seed

    def setup(self, stage=None):
        if hasattr(self, "x_volt"):
            return

        x_pixels = make_bars_stripes(n=self.image_size)
        x_volt = pixels_to_voltages(x_pixels, low=self.v_low, high=self.v_high)
        self.num_patterns = x_pixels.shape[0]

        if self.train_size is None:
            self.train_idx = torch.arange(self.num_patterns)
            self.test_idx = torch.arange(self.num_patterns)
        else:
            if not (0 < self.train_size < self.num_patterns):
                raise ValueError(
                    f"--train-size must be between 1 and {self.num_patterns - 1}, "
                    f"got {self.train_size}."
                )
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            perm = torch.randperm(self.num_patterns, generator=generator)
            self.train_idx = perm[: self.train_size]
            self.test_idx = perm[self.train_size :]

        self.x_pixels = x_pixels
        self.x_volt = x_volt

        self.train_dataset = TensorDataset(x_volt[self.train_idx], x_pixels[self.train_idx])
        self.test_dataset = TensorDataset(x_volt[self.test_idx], x_pixels[self.test_idx])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)


class DNPUAutoencoderModule(pl.LightningModule):
    """
    Wraps PlanarDNPUEncoderDigitalDecoder or DigitalBaselineAutoencoder.

    Loss = BCE-with-logits reconstruction + hardware-aware penalty on the raw
    DNPU readouts (relu(|z_raw| - z0)^2), matching train_hybrid.py /
    train_generalization.py. Falls back to penalizing z itself for models
    (e.g. the digital baseline) that don't expose last_z_raw.
    """

    def __init__(self, model, lr=1e-3, weight_decay=0.0, lambda_z=1e-3, z0=10.0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_z = lambda_z
        self.z0 = z0
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, target = batch
        logits, z = self.model(x)
        z_raw = getattr(self.model, "last_z_raw", z)

        loss_recon = F.binary_cross_entropy_with_logits(logits, target)
        z_penalty = torch.relu(z_raw.abs() - self.z0).pow(2).mean()
        loss = loss_recon + self.lambda_z * z_penalty

        pixel_acc, pattern_acc = reconstruction_accuracy(logits, target)

        metrics = {
            "loss": loss,
            "loss_recon": loss_recon,
            "z_penalty": z_penalty,
            "pixel_acc": pixel_acc,
            "pattern_acc": pattern_acc,
            "z_std": z.std(),
            "z_abs_max": z.abs().max(),
            "z_raw_std": z_raw.std(),
            "z_raw_abs_max": z_raw.abs().max(),
        }
        return loss, metrics

    def _log_metrics(self, metrics, prefix):
        prog_bar_keys = ("loss", "pixel_acc", "pattern_acc")
        for name, value in metrics.items():
            self.log(
                f"{prefix}_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(name in prog_bar_keys),
            )

    def training_step(self, batch, batch_idx):
        loss, metrics = self._step(batch)
        self._log_metrics(metrics, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, metrics = self._step(batch)
        self._log_metrics(metrics, "val")

    def test_step(self, batch, batch_idx):
        _, metrics = self._step(batch)
        self._log_metrics(metrics, "test")

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure)
        # Physical control voltages must stay within calibrated ranges.
        if hasattr(self.model, "clip_controls_"):
            self.model.clip_controls_()


class PeriodicPrintCallback(Callback):
    """
    Reproduces the periodic console table from the original train_*.py
    scripts (epoch 1, then every `every` epochs).
    """

    def __init__(self, every=500):
        self.every = every

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch != 1 and epoch % self.every != 0:
            return

        m = trainer.callback_metrics

        def g(key):
            v = m.get(key)
            return v.item() if v is not None else float("nan")

        msg = (
            f"epoch {epoch:5d} | "
            f"loss {g('train_loss'):.6f} | "
            f"rec {g('train_loss_recon'):.6f} | "
            f"z_pen {g('train_z_penalty'):.6f} | "
            f"train_pix {g('train_pixel_acc'):.3f} | "
            f"train_pat {g('train_pattern_acc'):.3f}"
        )
        if "val_pixel_acc" in m:
            msg += f" | val_pix {g('val_pixel_acc'):.3f} | val_pat {g('val_pattern_acc'):.3f}"

        print(msg)


def run_noise_robustness(
    model, x_volt, x_pixels, sigmas, repeats, v_low, v_high, seed, results_dir, run_name
):
    """
    Voltage-noise sweep, matching test_noise_robustness.py.
    """
    model.eval()
    torch.manual_seed(seed)

    print("\nNoise robustness sweep")
    print(f"  sigmas:  {sigmas}")
    print(f"  repeats: {repeats}\n")
    print(
        "sigma      rec_mean   rec_std    pixel_acc_mean pixel_acc_std  "
        "pattern_acc_mean pattern_acc_std  z_std_mean z_abs_max_mean"
    )

    sweep_results = {}

    for sigma in sigmas:
        rec_values, pixel_values, pattern_values = [], [], []
        z_std_values, z_abs_max_values = [], []

        for _ in range(repeats):
            x_eval = x_volt if sigma == 0.0 else add_voltage_noise(x_volt, sigma, v_low, v_high)

            with torch.no_grad():
                logits, z = model(x_eval)

            rec = F.binary_cross_entropy_with_logits(logits, x_pixels).item()
            pixel_acc, pattern_acc = reconstruction_accuracy(logits, x_pixels)

            rec_values.append(rec)
            pixel_values.append(pixel_acc.item())
            pattern_values.append(pattern_acc.item())
            z_std_values.append(z.std().item())
            z_abs_max_values.append(z.abs().max().item())

        rec_t = torch.tensor(rec_values)
        pix_t = torch.tensor(pixel_values)
        pat_t = torch.tensor(pattern_values)
        zstd_t = torch.tensor(z_std_values)
        zmax_t = torch.tensor(z_abs_max_values)

        sweep_results[sigma] = {
            "rec_mean": rec_t.mean().item(),
            "rec_std": rec_t.std(unbiased=False).item(),
            "pixel_acc_mean": pix_t.mean().item(),
            "pixel_acc_std": pix_t.std(unbiased=False).item(),
            "pattern_acc_mean": pat_t.mean().item(),
            "pattern_acc_std": pat_t.std(unbiased=False).item(),
            "z_std_mean": zstd_t.mean().item(),
            "z_abs_max_mean": zmax_t.mean().item(),
        }

        print(
            f"{sigma:<10.3f}"
            f"{rec_t.mean().item():<11.4f}"
            f"{rec_t.std(unbiased=False).item():<11.4f}"
            f"{pix_t.mean().item():<15.3f}"
            f"{pix_t.std(unbiased=False).item():<15.3f}"
            f"{pat_t.mean().item():<17.3f}"
            f"{pat_t.std(unbiased=False).item():<17.3f}"
            f"{zstd_t.mean().item():<11.3f}"
            f"{zmax_t.mean().item():<.3f}"
        )

    positive_sigmas = [s for s in sigmas if s > 0.0]
    if positive_sigmas:
        sigma_vis = positive_sigmas[len(positive_sigmas) // 2]
        x_noisy = add_voltage_noise(x_volt, sigma_vis, v_low, v_high)
        n = int(round(x_pixels.shape[1] ** 0.5))

        fig_path = plot_noisy_reconstructions(
            model=model,
            x_noisy=x_noisy,
            x_target=x_pixels,
            n=n,
            max_patterns=x_pixels.shape[0],
            save_path=results_dir / f"{run_name}_noise_sigma{sigma_vis:.2f}.png",
            input_low=v_low,
            input_high=v_high,
        )
        print("\nSaved noisy reconstruction figure:", fig_path)

    return sweep_results


def make_run_name(args):
    model_tag = "baseline" if args.baseline else ("frozen" if args.freeze_encoder else "hybrid")
    split_tag = f"train{args.train_size}" if args.train_size is not None else "full"
    return (
        f"{model_tag}_{split_tag}_"
        f"{args.image_size}x{args.image_size}_"
        f"ndata{args.n_data}_nctrl{args.n_control}_"
        f"{args.group_type}_seed{args.seed}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightning training/testing for the DNPU autoencoder."
    )

    # Model choice.
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use the purely digital baseline autoencoder instead of the DNPU hybrid model.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze DNPU control voltages and only train the digital decoder.",
    )
    parser.add_argument("--surrogate-checkpoint", type=str, default="surrogate_model.pt")

    # Architecture knobs.
    parser.add_argument("--image-size", type=int, default=4)
    parser.add_argument("--n-data", type=int, default=4)
    parser.add_argument("--n-control", type=int, default=3)
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="Compressed digital latent dimension. Default: no compression "
        "(hybrid) / 4 (baseline).",
    )
    parser.add_argument(
        "--group-type",
        type=str,
        default="auto",
        choices=["auto", "2x2", "horizontal", "vertical"],
    )

    # Data / split.
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="If set, train/test split like train_generalization.py. "
        "Default: train on all patterns (like train_hybrid.py).",
    )
    parser.add_argument("--v-low", type=float, default=-0.5)
    parser.add_argument("--v-high", type=float, default=0.5)

    # Training knobs.
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-z", type=float, default=1e-3)
    parser.add_argument("--z0", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--accelerator", type=str, default="cpu")

    # Output.
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-cases", action="store_true")
    parser.add_argument("--num-sample", type=int, default=8)

    # Noise robustness (test_noise_robustness.py).
    parser.add_argument(
        "--noise-sigmas",
        type=float,
        nargs="+",
        default=None,
        help="If set, run a voltage-noise robustness sweep after training.",
    )
    parser.add_argument("--noise-repeats", type=int, default=20)

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.baseline and args.n_data + args.n_control != 7:
        raise ValueError("--n-data + --n-control must be 7.")

    pl.seed_everything(args.seed, workers=True)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    run_name = args.run_name if args.run_name is not None else make_run_name(args)
    run_dir = results_dir / run_name

    datamodule = BarsStripesDataModule(
        image_size=args.image_size,
        v_low=args.v_low,
        v_high=args.v_high,
        train_size=args.train_size,
        seed=args.seed,
    )
    datamodule.setup()

    if args.baseline:
        model = DigitalBaselineAutoencoder(
            image_size=args.image_size,
            latent_dim=args.latent_dim if args.latent_dim is not None else 4,
        )
    else:
        processor, voltage_ranges = make_processor(args.surrogate_checkpoint)
        model = PlanarDNPUEncoderDigitalDecoder(
            processor=processor,
            voltage_ranges=voltage_ranges,
            image_size=args.image_size,
            n_data=args.n_data,
            n_control=args.n_control,
            latent_dim=args.latent_dim,
            group_type=args.group_type,
            init="center",
            freeze_encoder=args.freeze_encoder,
        )

    module = DNPUAutoencoderModule(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_z=args.lambda_z,
        z0=args.z0,
    )

    print("DNPU autoencoder (Lightning)")
    print(f"  run_name:      {run_name}")
    print(
        f"  model:         "
        f"{'baseline' if args.baseline else ('frozen-encoder hybrid' if args.freeze_encoder else 'hybrid')}"
    )
    print(f"  image_size:    {args.image_size}x{args.image_size}")
    print(f"  patterns:      {datamodule.num_patterns}")
    print(f"  train_size:    {len(datamodule.train_idx)}")
    print(f"  test_size:     {len(datamodule.test_idx)}")
    if not args.baseline:
        print(f"  n_data:        {args.n_data}")
        print(f"  n_control:     {args.n_control}")
        print(f"  group_type:    {args.group_type}")
        print(f"  raw_latent_dim:{model.raw_latent_dim}")
    print(f"  latent_dim:    {model.latent_dim}")
    print(f"  epochs:        {args.epochs}")
    print(f"  lr:            {args.lr}")
    print(f"  weight_decay:  {args.weight_decay}")
    print(f"  lambda_z:      {args.lambda_z}")
    print(f"  z0:            {args.z0}")
    print()

    csv_logger = CSVLogger(save_dir=str(results_dir), name=run_name, version="")
    tb_logger = TensorBoardLogger(save_dir=str(results_dir), name=run_name, version="")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best",
        monitor="val_pixel_acc",
        mode="max",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=1,
        logger=[csv_logger, tb_logger],
        callbacks=[checkpoint_callback, PeriodicPrintCallback(every=args.eval_every)],
        check_val_every_n_epoch=args.eval_every,
        enable_progress_bar=False,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(module, datamodule=datamodule)

    test_results = trainer.test(module, datamodule=datamodule, verbose=False)
    test_metrics = test_results[0] if test_results else {}

    model.eval()
    with torch.no_grad():
        full_logits, _ = model(datamodule.x_volt)
    full_pixel_acc, full_pattern_acc = reconstruction_accuracy(full_logits, datamodule.x_pixels)

    print("\nFinal metrics:")
    print("  test:", test_metrics)
    print(
        f"  all patterns: pixel_acc {full_pixel_acc.item():.4f} | "
        f"pattern_acc {full_pattern_acc.item():.4f}"
    )

    # Legacy-style checkpoint, compatible with the state-dict layout that
    # test_noise_robustness.py and the check_*.py scripts expect.
    legacy_checkpoint_path = results_dir / f"{run_name}.pt"
    legacy_payload = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "run_name": run_name,
        "latent_dim": model.latent_dim,
        "test_metrics": test_metrics,
    }
    if not args.baseline:
        legacy_payload["input_groups"] = model.input_groups
        legacy_payload["raw_latent_dim"] = model.raw_latent_dim
        legacy_payload["freeze_encoder"] = args.freeze_encoder
    if args.train_size is not None:
        legacy_payload["train_idx"] = datamodule.train_idx
        legacy_payload["test_idx"] = datamodule.test_idx

    torch.save(legacy_payload, legacy_checkpoint_path)
    print("\nSaved legacy checkpoint:", legacy_checkpoint_path)
    print("Saved Lightning checkpoints under:", run_dir / "checkpoints")

    if not args.baseline:
        print("\nEncoder control voltages:")
        for i, unit in enumerate(model.encoder.units):
            print(f"  unit {i}: {unit.control_voltages.detach().cpu().numpy()}")

    # Visualizations, evaluated on the held-out split (== full set when
    # --train-size was not given).
    eval_input, eval_target = datamodule.test_dataset.tensors

    sample_path, sample_indices = plot_reconstruction_sample(
        model=model,
        x_input=eval_input,
        x_target=eval_target,
        n=args.image_size,
        num_examples=min(args.num_sample, eval_target.shape[0]),
        seed=args.seed,
        save_path=results_dir / f"{run_name}_sample.png",
    )
    print("\nSaved sample figure:", sample_path)
    print("Sample indices:", sample_indices)

    if args.save_cases:
        cases_dir = results_dir / f"{run_name}_cases"
        saved_cases = save_reconstruction_cases(
            model=model,
            x_input=eval_input,
            x_target=eval_target,
            n=args.image_size,
            out_dir=cases_dir,
            prefix=run_name,
        )
        print("Saved individual reconstruction cases:", cases_dir)
        print("Number of case files:", len(saved_cases))

    if args.noise_sigmas is not None:
        run_noise_robustness(
            model=model,
            x_volt=datamodule.x_volt,
            x_pixels=datamodule.x_pixels,
            sigmas=args.noise_sigmas,
            repeats=args.noise_repeats,
            v_low=args.v_low,
            v_high=args.v_high,
            seed=args.seed,
            results_dir=results_dir,
            run_name=run_name,
        )


if __name__ == "__main__":
    main()
