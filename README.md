# DNPU autoencoder

Research prototype for grayscale CIFAR-10 autoencoding and latent representation learning with BrainSpy DNPU convolution modules.

The code studies compact autoencoders in which `DNPUConv2d` layers act as trainable nonlinear physical or hardware-like maps. The current system is explicitly **non-autonomous**: DNPU layers provide the nonlinear processing, while routing, normalization, optimization, loss evaluation, and visualization remain digital.

## Scientific motivation

The main question is whether a very small number of trainable physical control voltages can produce a useful low-dimensional representation of natural images.

The current CIFAR experiments study both physical encoders and DNPU-based decoders:

- DNPUConv encoder stacks that compress `1 x 32 x 32` grayscale images into compact latent representations.
- Raw physical bottlenecks versus an additional digital linear bottleneck.
- Zero-insertion upsampling decoders built either from ordinary `Conv2d` or from BrainSpy `DNPUConv2d`.
- Freeze ablations for DNPU controls, BatchNorm parameters, and the full encoder.
- Linear and MLP probes on frozen latent representations.

The zero-insertion decoder is inspired by transposed convolution, but it is **not** mathematically identical to `ConvTranspose2d` for nonlinear DNPU maps. It upsamples by inserting zeros into the feature map and then applies ordinary convolutional DNPU layers.

## Physical and digital components

| Role | Current implementation |
|---|---|
| Physical or hardware-like nonlinear encoder | `DNPUConv2d` |
| Physical or hardware-like nonlinear decoder | `DNPUConv2d` in `--decoder-type dnpu_zero_conv` |
| Trainable physical parameters | DNPU control voltages |
| Zero insertion and routing | Digital tensor operations |
| Tensor reshaping | Digital |
| Readout calibration | `BatchNorm2d` |
| Activations | `ReLU` |
| Optimization | Global digital backpropagation with Adam |
| Losses and metrics | Digital |
| Dataset handling and visualization | Digital |

The project therefore does not claim a fully analog or fully autonomous system.

## Current architecture

Main encoder configuration:

```text
1 x 32 x 32
-> 16 x 16 x 16
-> 1 x 8 x 8
-> raw latent 64
```

Supported decoder modes:

- `transpose`: existing digital `Linear -> ConvTranspose2d -> ConvTranspose2d` decoder.
- `zero_conv`: digital zero-insertion upsampling followed by ordinary `Conv2d`.
- `dnpu_zero_conv`: digital zero insertion plus DNPUConv decoder layers.

For the main raw-latent zero-conv configuration with `--dnpu-channels 16,1` and `--decoder-channels 16,1`, the feature-map path is:

```text
1 x 32 x 32
-> 16 x 16 x 16
-> 1 x 8 x 8
-> 16 x 16 x 16
-> 1 x 32 x 32
```

Intermediate zero-conv decoder stages use:

```text
convolution -> BatchNorm2d -> ReLU
```

The final stage uses:

```text
convolution -> BatchNorm2d
```

The decoder output remains logits. Sigmoid is used only inside the reconstruction losses where needed and for image-space metrics and saved reconstructions.

## Repository layout

Shared CIFAR package:

```text
dnpu_ae/
├── __init__.py
├── checkpoints.py
├── cifar_data.py
├── cifar_models.py
├── model_utils.py
├── processor.py
├── reconstruction.py
└── upsampling.py
```

Main CIFAR entry points:

- `train_cifar_dnpu_stack_autoencoder.py`: configurable DNPU stack autoencoder.
- `train_cifar_dnpuconv_autoencoder.py`: earlier CIFAR DNPUConv implementation kept for compatibility.
- `train_cifar_latent_probe.py`: linear and MLP probes on frozen latents.
- `train_cifar_fixed_decoder.py`: fixed-random-decoder hierarchy and oracle-`z` check.

Earlier prototype code for Bars & Stripes remains in the repository root:

- `data.py`
- `dnpu_layer.py`
- `models.py`
- `train_hybrid.py`
- `train_generalization.py`
- `train_frozen_encoder.py`

## Installation and environment

The development environment used on the cluster is:

```bash
cd /vol/tcm02/bagrov_storage/dnpu_autoencoder
source .venv/bin/activate
```

The repository also includes a conda environment file:

```bash
conda env create -f environment.yml
conda activate dnpu-ae
```

`surrogate_model.pt` must be present in the repository root before running DNPU experiments. It is intentionally not tracked by git.

BrainSpy and the surrounding PyTorch stack are version-sensitive. `requirements_cluster.txt` records the cluster package set used for the reported runs.

## CIFAR data handling

CIFAR-10 is loaded through `torchvision` into `data_cifar/`.

The preprocessing path is:

```text
RGB image -> one grayscale channel -> tensor in [0, 1]
```

Immediately before the first DNPU layer, the image is shifted into the voltage-like range:

```text
[-0.5, 0.5]
```

DataLoaders use `num_workers=0` intentionally for filesystem compatibility.

## Main commands

All commands below are intended to run from the repository root.

### 1. Existing transposed-convolution baseline

```bash
python train_cifar_dnpu_stack_autoencoder.py \
  --data-dir data_cifar \
  --results-dir results_cifar_transpose_16_1_raw64_l1 \
  --subset-size 5000 \
  --batch-size 32 \
  --epochs 10 \
  --encoder-type dnpu \
  --dnpu-channels 16,1 \
  --latent-mode raw \
  --decoder-type transpose \
  --decoder-channels 16,1 \
  --loss l1 \
  --device cpu
```

### 2. Digital zero-convolution decoder

```bash
python train_cifar_dnpu_stack_autoencoder.py \
  --data-dir data_cifar \
  --results-dir results_cifar_zero_conv_16_1_raw64_l1 \
  --subset-size 5000 \
  --batch-size 32 \
  --epochs 10 \
  --encoder-type dnpu \
  --dnpu-channels 16,1 \
  --latent-mode raw \
  --decoder-type zero_conv \
  --decoder-channels 16,1 \
  --loss l1 \
  --device cpu
```

### 3. DNPU zero-convolution decoder

```bash
python train_cifar_dnpu_stack_autoencoder.py \
  --data-dir data_cifar \
  --results-dir results_cifar_dnpu_zero_conv_16_1_raw64_l1 \
  --subset-size 5000 \
  --batch-size 32 \
  --epochs 10 \
  --encoder-type dnpu \
  --dnpu-channels 16,1 \
  --latent-mode raw \
  --decoder-type dnpu_zero_conv \
  --decoder-channels 16,1 \
  --loss l1 \
  --device cpu
```

### 4. Freeze-encoder ablation

```bash
python train_cifar_dnpu_stack_autoencoder.py \
  --data-dir data_cifar \
  --results-dir results_cifar_zero_conv_16_1_raw64_freeze_encoder \
  --subset-size 5000 \
  --batch-size 32 \
  --epochs 10 \
  --encoder-type dnpu \
  --dnpu-channels 16,1 \
  --latent-mode raw \
  --decoder-type zero_conv \
  --decoder-channels 16,1 \
  --loss l1 \
  --freeze-encoder \
  --device cpu
```

### 5. Linear and MLP latent probes

Linear probe:

```bash
python train_cifar_latent_probe.py \
  --checkpoint results_cifar_transpose_16_1_raw64_l1/cifar_dnpu_stack_autoencoder.pt \
  --head linear \
  --subset-size 5000 \
  --test-size 5000 \
  --batch-size 64 \
  --epochs 50 \
  --results-dir results_probe_linear \
  --device cpu
```

MLP probe:

```bash
python train_cifar_latent_probe.py \
  --checkpoint results_cifar_transpose_16_1_raw64_l1/cifar_dnpu_stack_autoencoder.pt \
  --head mlp \
  --hidden-dim 128 \
  --subset-size 5000 \
  --test-size 5000 \
  --batch-size 64 \
  --epochs 50 \
  --results-dir results_probe_mlp \
  --device cpu
```

## Indicative results

These values are representative short-run observations on 5,000 grayscale CIFAR-10 training images. They are not guaranteed benchmarks.

| Experiment | Indicative result |
|---|---:|
| DNPU encoder + digital transpose decoder | test MAE about **0.077-0.08** |
| Frozen complete encoder | test MAE about **0.19** |
| Frozen DNPU controls | test MAE about **0.087** |
| Frozen BatchNorm affine parameters | test MAE about **0.110** |
| Random encoder + linear probe | about **15-17%** accuracy |
| Trained encoder + linear probe | about **28-29%** accuracy |
| Trained encoder + MLP probe | about **34-35%** accuracy |

No numerical claims are made here yet for the new zero-insertion DNPU decoder modes.

## Limitations and open questions

- Autoencoder training is currently run only for about ten epochs in the main experiments.
- The goal is an engineering proof of concept, not state-of-the-art CIFAR reconstruction.
- The surrogate model is not the same as a hardware experiment.
- Zero insertion, routing, tensor reshaping, and BatchNorm calibration remain digital.
- The full system still relies on global backpropagation.
- CIFAR-10 is converted to grayscale.
- The physical decoder performance has not yet been established empirically.

## Status

Active research code. Interfaces and reported conclusions may change as the DNPU-based decoder is evaluated further.
