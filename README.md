# DNPU autoencoder research prototype

This repository explores DNPUConv-based physical encoders for grayscale CIFAR-10
autoencoding and latent representations. A calibrated BrainSpy surrogate stands
in for the physical DNPU hardware: image patches drive data electrodes, while
trainable control voltages represent hardware-like encoder parameters. Digital
BatchNorm layers calibrate DNPU readouts, and the image decoder is digital.

The code is a research prototype. It favors explicit experiment entry points and
reproducible comparisons over a general training framework.

## Scientific goal

The central question is whether local nonlinear DNPU readouts can form a useful
low-dimensional image representation. The current experiments examine:

- reconstruction from a raw 64-dimensional physical latent;
- encoder ablations that freeze DNPU controls, BatchNorm, or the full encoder;
- deeper configurable DNPUConv stacks;
- the capacity limit imposed by a fixed random decoder;
- class information in frozen latents, measured by linear and MLP probes.

With `latent_mode=raw`, the flattened DNPU stack output is z. With
`latent_mode=linear`, a learned digital linear projection maps the raw physical
readouts to the requested latent dimension.

## Repository layout

- `dnpu_ae/`: shared CIFAR models, processor construction, data loading,
  reconstruction utilities, model helpers, and checkpoint reconstruction.
- `train_cifar_dnpuconv_autoencoder.py`: original hybrid/two-stage CIFAR
  autoencoder entry point and checkpoint compatibility path.
- `train_cifar_dnpu_stack_autoencoder.py`: configurable DNPUConv stack.
- `train_cifar_fixed_decoder.py`: fixed-decoder hierarchy and oracle-z test.
- `train_cifar_latent_probe.py`: linear and MLP probes on frozen encoders.
- `data.py`, `dnpu_layer.py`, `models.py`, `visualize.py`: earlier
  Bars & Stripes prototype.
- `train_hybrid.py`, `train_generalization.py`,
  `train_frozen_encoder.py`: Bars & Stripes experiments.
- `check_*.py`, `inspect_*.py`, `test_*.py`: focused diagnostics.

All experiment scripts remain runnable from the repository root.

## Environment setup

The original environment targets Python 3.10 and is described in
`environment.yml`:

```bash
conda env create -f environment.yml
conda activate dnpu-ae
```

`requirements_cluster.txt` records the cluster environment used for existing
runs. In particular, this project depends on PyTorch, torchvision, and BrainSpy.

## Required surrogate model

Place the calibrated surrogate checkpoint at:

```text
surrogate_model.pt
```

It must be in the repository root when scripts are launched. The CIFAR
`make_processor()` loads that exact relative path on CPU and constructs a
BrainSpy simulation processor from its `info` and `model_state_dict`.
The file is intentionally not tracked by Git.

## CIFAR-10 data

CIFAR-10 is loaded through torchvision under `data_cifar/`. Images are
converted to one grayscale channel and then to tensors in [0, 1]. DNPU models
shift inputs to [-0.5, 0.5] before the first physical layer.

Experiments use deterministic prefix subsets: `--subset-size N` selects the
first N training examples, and test subsets similarly select a prefix. Training
loaders shuffle unless latent caching requires stable traversal. Dataset
download remains enabled, so torchvision will reuse local files or download
missing data.

## Main experiments

Commands below assume execution from the repository root. Choose an explicit
`--device` appropriate for the installed BrainSpy/PyTorch environment.

### Raw physical latent dimension 64

The channel list gives the output width of each stride-2 DNPUConv stage.
`16,1` produces:

```text
1x32x32 -> 16x16x16 -> 1x8x8 -> raw z with 64 values
```

```bash
python train_cifar_dnpu_stack_autoencoder.py \\
  --dnpu-channels 16,1 \\
  --latent-mode raw \\
  --loss l1 \\
  --results-dir results_cifar_dnpu16_1_raw64
```

### Encoder ablations

Freeze only the hardware-like DNPU control voltages:

```bash
python train_cifar_dnpu_stack_autoencoder.py \\
  --dnpu-channels 16,1 --latent-mode raw --loss l1 \\
  --freeze-dnpu \\
  --results-dir results_cifar_dnpu16_1_freeze_dnpu
```

Freeze only BatchNorm affine parameters, which are digital readout calibration:

```bash
python train_cifar_dnpu_stack_autoencoder.py \\
  --dnpu-channels 16,1 --latent-mode raw --loss l1 \\
  --freeze-bn \\
  --results-dir results_cifar_dnpu16_1_freeze_bn
```

Freeze the full encoder through z and train only the digital decoder:

```bash
python train_cifar_dnpu_stack_autoencoder.py \\
  --dnpu-channels 16,1 --latent-mode raw --loss l1 \\
  --freeze-encoder \\
  --results-dir results_cifar_dnpu16_1_freeze_encoder
```

Freezing BatchNorm here freezes its affine parameters. It does not introduce a
new policy for running statistics; existing train/eval mode behavior is kept.

### A deeper DNPU stack

```bash
python train_cifar_dnpu_stack_autoencoder.py \\
  --dnpu-channels 8,4,4 \\
  --latent-mode raw \\
  --loss l1 \\
  --results-dir results_cifar_dnpu_stack_8_4_4_raw64
```

Each value in `--dnpu-channels` adds one kernel-2, stride-2 DNPUConv stage.
For `8,4,4`, the final 4x4x4 output is again a raw 64-dimensional latent.

### Latent probes

First train an autoencoder and retain its
`cifar_dnpu_stack_autoencoder.pt` checkpoint. A linear probe freezes the
reconstructed encoder, caches z, and trains only a ten-class head:

```bash
python train_cifar_latent_probe.py \\
  --checkpoint results_cifar_dnpu16_1_raw64/cifar_dnpu_stack_autoencoder.pt \\
  --head linear \\
  --results-dir results_probe_dnpu16_1_raw64_linear
```

The corresponding MLP probe is:

```bash
python train_cifar_latent_probe.py \\
  --checkpoint results_cifar_dnpu16_1_raw64/cifar_dnpu_stack_autoencoder.pt \\
  --head mlp --hidden-dim 128 \\
  --results-dir results_probe_dnpu16_1_raw64_mlp
```

Use `--random-init` with the same checkpoint metadata for a randomly
initialized encoder baseline.

### Fixed decoder and oracle-z sanity check

The fixed-decoder hierarchy asks whether an encoder can adapt to a randomly
initialized digital decoder whose weights never change. The oracle-z mode
removes the encoder entirely and directly optimizes one latent vector per image.
It therefore estimates the best reconstruction that this frozen decoder can
express:

```bash
python train_cifar_fixed_decoder.py \\
  --mode oracle_z \\
  --latent-dim 64 \\
  --results-dir results_fixed_decoder_raw64_oracle
```

Other modes are `frozen_random`, `train_dnpu_encoder`, and
`train_digital_encoder`.

## Indicative results

The following values summarize current runs and are not formal benchmarks:

- DNPU stack `16,1`, raw physical latent 64: reconstruction test MAE about
  0.077.
- Full frozen encoder: test MAE about 0.19.
- Frozen DNPU controls: test MAE about 0.087.
- Frozen BatchNorm affine parameters: test MAE about 0.110.
- Random-encoder latent probe: roughly 15–17% accuracy.
- Trained-encoder linear probe: roughly 28–29% accuracy.
- Trained-encoder MLP probe: roughly 34–35% accuracy.
- The random frozen-decoder oracle saturates near MAE 0.176.

## Interpretation and caveats

The reconstruction ablations suggest that trainable physical controls help, but
the digital calibration and decoder also carry substantial responsibility.
Probe improvements over a random encoder indicate that the trained latent
contains class-relevant information even though training uses only
reconstruction loss. The MLP/linear gap suggests some of that information is
not linearly organized.

The fixed random decoder is unusually restrictive: even direct per-image
optimization of z saturates around MAE 0.176. It is therefore too harsh to use
as a general reconstruction readout, and poor performance with it should not be
attributed solely to the encoder.

Results depend on subset size, seed, surrogate calibration, software versions,
and optimization duration. The processor is currently a simulation of one
calibrated device rather than a complete physical-array deployment. Reported
figures should be treated as indicative research observations, not final
hardware performance or CIFAR-10 benchmarks.
