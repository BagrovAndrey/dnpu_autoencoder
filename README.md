# DNPU autoencoder

Research prototype for grayscale CIFAR-10 autoencoding and latent representation learning with BrainSpy-based DNPU convolution modules.

The code studies compact autoencoders in which DNPU convolution layers act as trainable nonlinear physical or hardware-like maps. The current CIFAR path uses a custom `DNPUConv2d_DNPUChild` wrapper built on top of `DNPUUnit_DNPUChild`, while routing, normalization, optimization, loss evaluation, and visualization remain digital. The system is therefore explicitly **non-autonomous**.

## Scientific motivation

The main question is whether a very small number of trainable physical control voltages can produce a useful low-dimensional representation of natural images.

The current CIFAR experiments study both physical encoders and DNPU-based decoders:

- DNPU convolution encoder stacks that compress `1 x 32 x 32` grayscale images into compact latent representations.
- Raw physical bottlenecks versus an additional digital linear bottleneck.
- Zero-insertion upsampling decoders built either from ordinary `Conv2d` or from DNPU convolution layers backed by a shared BrainSpy `Processor`.
- Zero-insertion upsampling decoders with an additional same-resolution mixing convolution after each upsampling stage.
- Nearest-neighbor upsampling decoders built either from ordinary `Conv2d` or from DNPU convolution layers backed by a shared BrainSpy `Processor`.
- Hybrid decoder variants that first apply a global digital `Linear` map and only then decode with nearest-neighbor upsampling.
- Freeze ablations for DNPU controls, BatchNorm parameters, and the full encoder.
- Fixed-decoder hierarchy experiments, including frozen random decoders and oracle-`z` latent optimization.
- Linear and MLP probes on frozen latent representations.

The zero-insertion decoder is inspired by transposed convolution, but it is **not** mathematically identical to `ConvTranspose2d` for nonlinear DNPU maps. It upsamples by inserting zeros into the feature map and then applies ordinary convolutional DNPU layers.

## Physical and digital components

| Role | Current implementation |
|---|---|
| Physical or hardware-like nonlinear encoder | `DNPUConv2d_DNPUChild` built on `DNPUUnit_DNPUChild` |
| Physical or hardware-like nonlinear decoder | `DNPUConv2d_DNPUChild` in `--decoder-type dnpu_zero_conv`, `dnpu_zero_conv_mixing`, `dnpu_nearest_conv`, and the DNPU convolution stages of `dnpu_nearest_conv_linear` |
| Trainable physical parameters | DNPU control voltages |
| Global latent-to-feature projection | Digital `Linear` in `transpose`, `nearest_conv_linear`, and `dnpu_nearest_conv_linear` |
| Zero insertion and routing | Digital tensor operations |
| Nearest-neighbor upsampling | Digital tensor operations |
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
- `zero_conv_mixing`: `zero_conv` plus an extra ordinary `Conv2d` mixing layer after each upsampling stage.
- `dnpu_zero_conv_mixing`: `dnpu_zero_conv` plus an extra DNPUConv mixing layer after each upsampling stage.
- `nearest_conv`: nearest-neighbor upsampling followed by ordinary `Conv2d`.
- `dnpu_nearest_conv`: nearest-neighbor upsampling followed by DNPUConv decoder layers.
- `nearest_conv_linear`: global digital `Linear` projection to a feature map, then `nearest_conv`.
- `dnpu_nearest_conv_linear`: global digital `Linear` projection to a feature map, then `dnpu_nearest_conv`.

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

For the mixing variants, each upsampling stage adds a second same-resolution `2 x 2` convolution after the first one. In code this is implemented as a residual mixing block. Intermediate stages use:

```text
zero insert -> pad -> convolution -> BatchNorm2d -> ReLU
-> pad -> mixing convolution -> BatchNorm2d
-> residual add -> ReLU
```

The final mixing stage uses:

```text
zero insert -> pad -> convolution -> BatchNorm2d
-> pad -> mixing convolution
-> residual add
```

`zero_conv`, `dnpu_zero_conv`, `zero_conv_mixing`, `dnpu_zero_conv_mixing`, `nearest_conv`, and `dnpu_nearest_conv` require `--latent-mode raw`, because the latent vector is reshaped directly back into the encoder output feature map.
`transpose`, `nearest_conv_linear`, and `dnpu_nearest_conv_linear` are the decoder modes that support `--latent-mode linear`.
For all upsampling decoders, `--decoder-channels` must contain exactly one channel count per factor-two upsampling stage and must end with `1`.

For `nearest_conv_linear` and `dnpu_nearest_conv_linear`, the first decoder step is a large digital `Linear(latent_dim -> C x H x W)` projection. The DNPU variant is therefore not a fully physical decoder: it is a DNPU convolutional decoder preceded by a substantial digital mixing stage.

The configurable DNPU stack supports at most five stride-2 DNPU stages on a `32 x 32` input:

```text
32 -> 16 -> 8 -> 4 -> 2 -> 1
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
├── dnpu_conv.py
├── model_utils.py
├── processor.py
├── reconstruction.py
└── upsampling.py
```

`dnpu_ae/processor.py` is the ownership boundary for BrainSpy integration. It builds one shared simulation backend `Processor`, freezes the surrogate parameters inside that backend, and exposes a small factory API that creates per-layer `DNPUConv2d_DNPUChild` modules with their own trainable `control_voltages`.

`dnpu_ae/dnpu_conv.py` contains the CIFAR-facing convolution wrapper. Its ownership and inheritance chain is:

```text
training script
-> DNPUBackend
-> DNPUConv2d_DNPUChild
-> DNPUUnit_DNPUChild
-> shared BrainSpy Processor
```

This preserves one shared frozen surrogate backend while keeping each convolution layer's `control_voltages` separate and trainable.

The recent refactor also simplified class structure:

- `DNPUBackend` centralizes shared `Processor` ownership and compatibility wrappers.
- `DNPUStackCIFARAutoencoder` and `DNPUConvCIFARAutoencoder` accept either a legacy `processor` or the newer `backend` abstraction.
- The decoder families now share explicit base classes in `dnpu_ae/upsampling.py`, rather than duplicating digital and DNPU implementations stage by stage.

Main CIFAR entry points:

- `train_cifar_dnpu_stack_autoencoder.py`: configurable DNPU stack autoencoder.
- `train_cifar_dnpuconv_autoencoder.py`: earlier CIFAR DNPUConv implementation kept for compatibility.
- `train_cifar_latent_probe.py`: linear and MLP probes on frozen latents.
- `train_cifar_fixed_decoder.py`: fixed-decoder hierarchy, frozen random baselines, DNPU-vs-digital encoder comparisons, and oracle-`z` sanity checks.

Earlier prototype code for Bars & Stripes remains in the repository root:

- `data.py`
- `dnpu_layer.py`
- `models.py`
- `train_hybrid.py`
- `train_generalization.py`
- `train_frozen_encoder.py`

Useful diagnostic and test scripts in the repository root:

- `test_upsampling.py`: unit tests for zero-insertion and nearest-neighbor decoder utilities.
- `test_dnpuconv_forward.py`: forward/backward smoke test for `DNPUConv2d_DNPUChild`.
- `check_surrogate.py`: inspect `surrogate_model.pt` and verify BrainSpy importability.
- `check_processor_forward.py`, `check_dnpu_unit.py`, `check_dnpu_layer.py`: low-level DNPU processor and custom layer checks.
- `check_lightning_parameters.py`: compare local parameter counting against `pytorch_lightning` model summaries and verify that surrogate parameters stay frozen while DNPU `control_voltages` remain trainable.
- `inspect_dnpuconv.py`, `inspect_info.py`: quick introspection helpers for BrainSpy internals and surrogate checkpoint metadata.

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

The repository does not store CIFAR-10 itself. The dataset is downloaded on demand by `torchvision.datasets.CIFAR10(..., download=True)` the first time you run a CIFAR script.

If you want to download the dataset in advance, run:

```bash
python -c "from dnpu_ae.cifar_data import make_grayscale_cifar_subsets; make_grayscale_cifar_subsets('data_cifar', train_size=1, test_size=1)"
```

After that, the local directory will contain the usual `torchvision` CIFAR files under `data_cifar/`.

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

### 4. Digital zero-convolution decoder with mixing

```bash
python train_cifar_dnpu_stack_autoencoder.py \
  --data-dir data_cifar \
  --results-dir results_cifar_zero_conv_mixing_16_1_raw64_l1 \
  --subset-size 5000 \
  --batch-size 32 \
  --epochs 10 \
  --encoder-type dnpu \
  --dnpu-channels 16,1 \
  --latent-mode raw \
  --decoder-type zero_conv_mixing \
  --decoder-channels 16,1 \
  --loss l1 \
  --device cpu
```

### 5. DNPU zero-convolution decoder with mixing

```bash
python train_cifar_dnpu_stack_autoencoder.py \
  --data-dir data_cifar \
  --results-dir results_cifar_dnpu_zero_conv_mixing_16_1_raw64_l1 \
  --subset-size 5000 \
  --batch-size 32 \
  --epochs 10 \
  --encoder-type dnpu \
  --dnpu-channels 16,1 \
  --latent-mode raw \
  --decoder-type dnpu_zero_conv_mixing \
  --decoder-channels 16,1 \
  --loss l1 \
  --device cpu
```

### 6. Spatial-latent resolution check with digital mixing decoder

This run tests whether block artifacts are tied to the coarse `1 x 8 x 8` latent grid by keeping only one stride-2 encoder stage:

```text
1 x 32 x 32 -> 1 x 16 x 16 -> 1 x 32 x 32
```

Command:

```bash
python3 train_cifar_dnpu_stack_autoencoder.py \
  --data-dir data_cifar \
  --results-dir results_cifar_zero_conv_mixing_1x16x16_raw256_l1 \
  --subset-size 5000 \
  --batch-size 32 \
  --epochs 10 \
  --encoder-type dnpu \
  --dnpu-channels 1 \
  --latent-mode raw \
  --decoder-type zero_conv_mixing \
  --decoder-channels 1 \
  --loss l1 \
  --device cpu
```

### 7. Spatial-latent resolution check with DNPU twin decoder

The DNPU twin version of the same `1 x 16 x 16 -> 1 x 32 x 32` test is:

```bash
python3 train_cifar_dnpu_stack_autoencoder.py \
  --data-dir data_cifar \
  --results-dir results_cifar_dnpu_zero_conv_mixing_1x16x16_raw256_l1 \
  --subset-size 5000 \
  --batch-size 32 \
  --epochs 10 \
  --encoder-type dnpu \
  --dnpu-channels 1 \
  --latent-mode raw \
  --decoder-type dnpu_zero_conv_mixing \
  --decoder-channels 1 \
  --loss l1 \
  --device cpu
```

For comparison, the original `1 x 8 x 8 = 64` raw latent uses two encoder stages and therefore:

```bash
--dnpu-channels 16,1 --decoder-channels 16,1
```

### 8. Freeze-encoder ablation

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

The same script also supports `--freeze-dnpu` and `--freeze-bn`. `--freeze-encoder` is mutually exclusive with those partial-freeze modes, and `--freeze-dnpu` cannot be combined with `--freeze-bn`.

### 9. Fixed-decoder hierarchy experiments

Frozen random DNPU encoder+decoder baseline:

```bash
python train_cifar_fixed_decoder.py \
  --mode frozen_random \
  --data-dir data_cifar \
  --results-dir results_fixed_decoder_frozen_random \
  --subset-size 5000 \
  --test-size 5000 \
  --batch-size 32 \
  --encoder-type dnpu \
  --dnpu-channels 16,1 \
  --latent-mode raw \
  --latent-dim 64 \
  --loss l1 \
  --device cpu
```

Train a DNPU encoder against a frozen reinitialized decoder:

```bash
python train_cifar_fixed_decoder.py \
  --mode train_dnpu_encoder \
  --data-dir data_cifar \
  --results-dir results_fixed_decoder_train_dnpu \
  --subset-size 5000 \
  --test-size 5000 \
  --batch-size 32 \
  --epochs 50 \
  --encoder-type dnpu \
  --dnpu-channels 16,1 \
  --latent-mode raw \
  --latent-dim 64 \
  --loss l1 \
  --device cpu
```

Train a digital encoder against the same frozen random decoder:

```bash
python train_cifar_fixed_decoder.py \
  --mode train_digital_encoder \
  --data-dir data_cifar \
  --results-dir results_fixed_decoder_train_digital \
  --subset-size 5000 \
  --test-size 5000 \
  --batch-size 32 \
  --epochs 50 \
  --latent-dim 64 \
  --loss l1 \
  --device cpu
```

Oracle-`z` ceiling for the frozen digital decoder:

```bash
python train_cifar_fixed_decoder.py \
  --mode oracle_z \
  --data-dir data_cifar \
  --results-dir results_fixed_decoder_oracle_z \
  --test-size 5000 \
  --batch-size 32 \
  --latent-dim 64 \
  --loss l1 \
  --oracle-steps 500 \
  --oracle-lr 1e-2 \
  --oracle-batches 20 \
  --device cpu
```

Additional useful fixed-decoder controls:

- `--eval-batches`: number of test batches used for periodic evaluation during training.
- `--decoder-seed`: deterministic reinitialization seed for the frozen decoder.
- `--oracle-log-every`: print oracle metrics every N latent-optimization steps.

### 10. Linear and MLP latent probes

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

Random-initialized control for the same checkpointed architecture:

```bash
python train_cifar_latent_probe.py \
  --checkpoint results_cifar_transpose_16_1_raw64_l1/cifar_dnpu_stack_autoencoder.pt \
  --random-init \
  --head linear \
  --subset-size 5000 \
  --test-size 5000 \
  --batch-size 64 \
  --epochs 50 \
  --results-dir results_probe_linear_random_init \
  --device cpu
```

The probe loader accepts both current stack checkpoints and older legacy CIFAR checkpoints. It reconstructs the frozen encoder from the saved metadata, caches train/test latents once, trains only the probe head, and saves the result as `latent_probe_head.pt`.

## Saved artifacts

`train_cifar_dnpu_stack_autoencoder.py` saves:

- per-epoch reconstruction grids such as `cifar_recon_epoch0001.png`
- a checkpoint `cifar_dnpu_stack_autoencoder.pt` containing `model_state_dict`, CLI `args`, `dnpu_channels`, `raw_channels`, `raw_spatial_size`, `raw_latent_dim`, `latent_dim`, `decoder_type`, and `decoder_channels`

`train_cifar_dnpuconv_autoencoder.py` saves the legacy checkpoint `cifar_dnpuconv_autoencoder.pt`.

`train_cifar_fixed_decoder.py` saves reconstruction grids for the chosen mode:

- `recon_frozen_random.png` for the fully frozen random baseline
- `recon_epoch0000.png` plus per-epoch grids for the trainable-encoder modes
- `recon_oracle_z.png` for the oracle run

`train_cifar_latent_probe.py` saves the trained probe head as `latent_probe_head.pt`.

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

No numerical claims are made here yet for the new zero-insertion decoder modes with or without mixing, or for the fixed-decoder hierarchy experiments.

## Limitations and open questions

- Autoencoder training is currently run only for about ten epochs in the main experiments.
- The goal is an engineering proof of concept, not state-of-the-art CIFAR reconstruction.
- The surrogate model is not the same as a hardware experiment.
- Zero insertion, routing, tensor reshaping, and BatchNorm calibration remain digital.
- Nearest-neighbor upsampling and the `*_nearest_conv_linear` latent-to-feature projection remain digital.
- The mixing variants still rely on digital zero insertion and routing even when the convolution layers themselves are DNPU-based.
- `dnpu_nearest_conv_linear` is not a fully physical decoder because a global digital `Linear` map precedes the DNPU convolutional stages.
- The full system still relies on global backpropagation.
- CIFAR-10 is converted to grayscale.
- The physical decoder performance has not yet been established empirically.

## Status

Active research code. Interfaces and reported conclusions may change as the DNPU-based decoder is evaluated further.
