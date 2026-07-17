# DNPU autoencoder

Research prototype for grayscale CIFAR-10 autoencoding and latent representation learning with convolutional dopant-network processing units (DNPUs).

The project uses the `DNPUConv2d` implementation from [BrainSpy](https://github.com/BraiNEdarwin/brains-py) together with a calibrated surrogate model of a DNPU device. The current system is **non-autonomous**: DNPU layers provide hardware-like nonlinear processing, while data handling, calibration, optimization, and image decoding are performed digitally.

## Scientific question

The main question is whether a very small number of trainable physical control voltages can produce a useful low-dimensional representation of natural images.

The current experiments test:

- grayscale CIFAR-10 reconstruction with one or more `DNPUConv2d` layers;
- raw physical bottlenecks versus an additional digital linear bottleneck;
- ablations in which DNPU controls, BatchNorm parameters, or the complete encoder are frozen;
- the amount of class information retained in the latent representation;
- the expressive ceiling imposed by a fixed random decoder.

This is a proof-of-concept codebase rather than a state-of-the-art CIFAR-10 model.

## Current architecture

The main configuration is a two-stage physical encoder with channel sequence `16,1`:

```text
input                         1 x 32 x 32
DNPUConv2d, stride 2         16 x 16 x 16
BatchNorm2d + ReLU
DNPUConv2d, stride 2          1 x  8 x  8
BatchNorm2d + ReLU
flattened raw latent                     64
digital Linear                          16384
digital ConvTranspose2d      8 x 16 x 16
ReLU
digital ConvTranspose2d      1 x 32 x 32
sigmoid used only for reconstruction metrics/output
```

The physical encoder therefore maps a grayscale image to a raw 64-dimensional latent representation:

```text
1 x 32 x 32 -> 16 x 16 x 16 -> 1 x 8 x 8 -> z in R^64
```

For this architecture, the encoder contains about 130 trainable parameters in the present surrogate configuration:

- 96 hardware-like DNPU control voltages;
- 34 digital BatchNorm affine parameters.

The digital decoder is much larger than the physical encoder.

### Physical and digital components

| Component | Current implementation |
|---|---|
| Nonlinear patch processing | `DNPUConv2d` through a calibrated BrainSpy surrogate |
| Trainable physical parameters | DNPU control voltages |
| Readout calibration after each DNPU layer | Digital `BatchNorm2d` |
| Activation after each encoder stage | Digital ReLU |
| Bottleneck | Raw DNPU output or optional digital linear projection |
| Decoder | Digital `Linear` + `ConvTranspose2d` network |
| Training | Global digital backpropagation with Adam |
| Loss and metrics | Digital |
| Hardware execution | Not yet implemented; current experiments use a surrogate |

The phrase *physical encoder* in this repository refers to a neural-network module constructed from surrogate DNPU layers. It does not imply that the complete experiment is already running on a fabricated array.

## Latent modes

Two bottleneck definitions are supported.

### Raw latent

```text
--latent-mode raw
```

The final DNPU feature map is flattened directly. No trainable digital compression is inserted between the physical encoder and the latent representation.

For the main `16,1` architecture:

```text
1 x 8 x 8 -> 64 latent values
```

### Linear latent

```text
--latent-mode linear --latent-dim 64
```

The final physical feature map is flattened and passed through a trainable digital linear layer. This mode allows the requested latent dimension to differ from the raw number of DNPU readouts, but the latent is no longer purely physical.

## Indicative results

The following values summarize representative runs with 5,000 grayscale CIFAR-10 training images and short training schedules. They are not formal benchmarks.

| Experiment | Indicative result |
|---|---:|
| Trainable `16,1` DNPU encoder, raw latent 64 | test MAE about **0.077–0.08** |
| DNPU controls frozen | test MAE about **0.087** |
| BatchNorm affine parameters frozen | test MAE about **0.110** |
| Complete encoder frozen | test MAE about **0.19** |
| Random encoder + linear classification probe | **15–17%** accuracy |
| Trained encoder + linear classification probe | **28–29%** accuracy |
| Trained encoder + small MLP probe | **34–35%** accuracy |
| Oracle latent optimization with fixed random decoder | MAE saturates near **0.176** |

The encoder ablations show that both the DNPU controls and digital readout calibration matter. The probe experiments show that reconstruction training introduces class-relevant structure into the latent space. The gap between the linear and MLP probes indicates that this information is not fully linearly organized.

The fixed-random-decoder experiment is intentionally severe. Directly optimizing one latent vector per image still saturates near MAE 0.176, so poor reconstruction in that setting cannot be attributed only to the encoder.

## Repository layout

### Shared CIFAR package

```text
dnpu_ae/
├── __init__.py
├── processor.py        # BrainSpy Processor construction
├── cifar_data.py       # grayscale CIFAR-10 subsets and DataLoaders
├── cifar_models.py     # autoencoders and fixed-decoder baselines
├── reconstruction.py  # losses, metrics, and reconstruction grids
├── model_utils.py      # parameter counting, freezing, reset helpers
└── checkpoints.py      # reconstruction of encoders from checkpoints
```

### CIFAR experiment entry points

- `train_cifar_dnpu_stack_autoencoder.py`  
  Main configurable DNPUConv autoencoder.

- `train_cifar_dnpuconv_autoencoder.py`  
  Earlier hybrid/two-stage CIFAR implementation retained for checkpoint compatibility.

- `train_cifar_latent_probe.py`  
  Linear and MLP classification probes trained on frozen latent representations.

- `train_cifar_fixed_decoder.py`  
  Fixed-decoder hierarchy and oracle-\(z\) sanity check.

### Earlier Bars & Stripes prototype

- `data.py`
- `dnpu_layer.py`
- `models.py`
- `visualize.py`
- `train_hybrid.py`
- `train_generalization.py`
- `train_frozen_encoder.py`

### Diagnostics

Files named `check_*.py`, `inspect_*.py`, and `test_*.py` contain focused API and surrogate checks from development.

## Installation

The code was developed with Python 3.10.

### Conda environment

```bash
git clone https://github.com/BagrovAndrey/dnpu_autoencoder.git
cd dnpu_autoencoder
git checkout dnpu-conv-cifar

conda env create -f environment.yml
conda activate dnpu-ae
```

The cluster environment used for the reported runs is recorded in `requirements_cluster.txt`. Its central versions include:

```text
Python       3.10
PyTorch      1.13.1
torchvision  0.14.1
BrainSpy     1.0.2
NumPy        1.26.4
```

BrainSpy and the PyTorch stack are version-sensitive. Reproducing the recorded environment is safer than installing the newest available packages.

## Required surrogate checkpoint

The file

```text
surrogate_model.pt
```

must be placed in the repository root before launching DNPU experiments. It is intentionally excluded from Git.

The processor is constructed in `dnpu_ae/processor.py` as a BrainSpy simulation processor using:

- `ckpt["info"]`;
- `ckpt["model_state_dict"]`;
- `plateau_length = 1`;
- `slope_length = 0`.

The current code expects this exact filename and relative location.

## CIFAR-10 data

CIFAR-10 is loaded through `torchvision` into:

```text
data_cifar/
```

The transform is:

```text
RGB image -> one grayscale channel -> tensor in [0,1]
```

Before the first DNPU layer, the model shifts the image to the voltage-like interval:

```text
[-0.5, 0.5]
```

Training and test subsets are deterministic prefixes of the corresponding CIFAR-10 splits. DataLoaders use `num_workers=0`, which is intentional for compatibility with the cluster filesystem.

## Quick start

All commands below are intended to be run from the repository root.

### Main raw-64 autoencoder

```bash
python train_cifar_dnpu_stack_autoencoder.py \
  --data-dir data_cifar \
  --results-dir results_cifar_dnpu_stack_16_1_raw64_l1 \
  --subset-size 5000 \
  --batch-size 32 \
  --epochs 10 \
  --encoder-type dnpu \
  --dnpu-channels 16,1 \
  --latent-mode raw \
  --loss l1 \
  --device cpu
```

Expected architecture:

```text
1 x 32 x 32 -> 16 x 16 x 16 -> 1 x 8 x 8 -> raw latent 64
```

The script saves:

```text
results directory/
├── cifar_recon_epoch0001.png
├── cifar_recon_epoch0010.png
└── cifar_dnpu_stack_autoencoder.pt
```

### Deeper DNPU stack with the same latent dimension

```bash
python train_cifar_dnpu_stack_autoencoder.py \
  --data-dir data_cifar \
  --results-dir results_cifar_dnpu_stack_8_4_4_raw64_l1 \
  --subset-size 5000 \
  --batch-size 32 \
  --epochs 10 \
  --encoder-type dnpu \
  --dnpu-channels 8,4,4 \
  --latent-mode raw \
  --loss l1 \
  --device cpu
```

This gives:

```text
1 x 32 x 32 -> 8 x 16 x 16 -> 4 x 8 x 8 -> 4 x 4 x 4
```

and therefore again a 64-dimensional raw latent.

## Encoder ablations

### Freeze DNPU control voltages

```bash
python train_cifar_dnpu_stack_autoencoder.py \
  --data-dir data_cifar \
  --results-dir results_cifar_dnpu16_1_freeze_dnpu \
  --subset-size 5000 \
  --batch-size 32 \
  --epochs 10 \
  --encoder-type dnpu \
  --dnpu-channels 16,1 \
  --latent-mode raw \
  --loss l1 \
  --freeze-dnpu \
  --device cpu
```

The digital BatchNorm calibration and decoder remain trainable.

### Freeze BatchNorm affine parameters

```bash
python train_cifar_dnpu_stack_autoencoder.py \
  --data-dir data_cifar \
  --results-dir results_cifar_dnpu16_1_freeze_bn \
  --subset-size 5000 \
  --batch-size 32 \
  --epochs 10 \
  --encoder-type dnpu \
  --dnpu-channels 16,1 \
  --latent-mode raw \
  --loss l1 \
  --freeze-bn \
  --device cpu
```

This freezes the trainable BatchNorm scale and offset parameters. It does not replace the existing train/eval handling of BatchNorm running statistics.

### Freeze the complete encoder

```bash
python train_cifar_dnpu_stack_autoencoder.py \
  --data-dir data_cifar \
  --results-dir results_cifar_dnpu16_1_freeze_encoder \
  --subset-size 5000 \
  --batch-size 32 \
  --epochs 10 \
  --encoder-type dnpu \
  --dnpu-channels 16,1 \
  --latent-mode raw \
  --loss l1 \
  --freeze-encoder \
  --device cpu
```

Only the digital decoder remains trainable.

## Latent classification probes

The probe script reconstructs the encoder from an autoencoder checkpoint, freezes it, caches the latent vectors, and trains a ten-class classifier on top.

### Linear probe

```bash
python train_cifar_latent_probe.py \
  --checkpoint results_cifar_dnpu_stack_16_1_raw64_l1/cifar_dnpu_stack_autoencoder.pt \
  --head linear \
  --subset-size 5000 \
  --test-size 5000 \
  --batch-size 64 \
  --epochs 50 \
  --results-dir results_probe_dnpu16_1_raw64_linear \
  --device cpu
```

### MLP probe

```bash
python train_cifar_latent_probe.py \
  --checkpoint results_cifar_dnpu_stack_16_1_raw64_l1/cifar_dnpu_stack_autoencoder.pt \
  --head mlp \
  --hidden-dim 128 \
  --subset-size 5000 \
  --test-size 5000 \
  --batch-size 64 \
  --epochs 50 \
  --results-dir results_probe_dnpu16_1_raw64_mlp \
  --device cpu
```

### Random-encoder control

Add:

```text
--random-init
```

to either probe command. The model architecture is reconstructed from checkpoint metadata, but the encoder weights are not loaded.

## Fixed random decoder experiments

The fixed-decoder experiments separate encoder limitations from decoder expressivity.

Available modes:

```text
frozen_random
train_dnpu_encoder
train_digital_encoder
oracle_z
```

The most direct decoder-capacity test is `oracle_z`, which removes the encoder and optimizes a separate latent vector for each image while keeping the random decoder fixed:

```bash
python train_cifar_fixed_decoder.py \
  --mode oracle_z \
  --data-dir data_cifar \
  --results-dir results_fixed_decoder_raw64_oracle \
  --test-size 64 \
  --batch-size 16 \
  --latent-dim 64 \
  --loss l1 \
  --oracle-steps 10000 \
  --oracle-lr 1e-2 \
  --oracle-batches 1 \
  --oracle-log-every 500 \
  --device cpu
```

In current tests, the optimized latent saturates near MAE 0.176. This indicates that the frozen random decoder itself is too restrictive to serve as the main reconstruction benchmark.

## Losses and reported metrics

Training supports:

```text
bce
mse
l1
bce_l1
```

The decoder returns logits. A sigmoid is applied inside the reconstruction loss where required and for all image-space metrics.

Every evaluation reports per-pixel:

- selected optimization loss;
- binary cross-entropy;
- mean squared error;
- mean absolute error.

## Reproducibility notes

- Default random seed: `0`.
- CIFAR subsets are deterministic prefixes.
- Training DataLoaders shuffle the selected training subset.
- Test DataLoaders do not shuffle.
- Evaluation in the autoencoder scripts is limited to 20 test batches.
- The surrogate checkpoint is loaded on CPU.
- Results depend on the surrogate calibration, PyTorch/BrainSpy versions, optimization duration, and subset size.

The reported runs use only about ten autoencoder epochs. The aim is to establish that the architecture works and that the latent is nontrivial, not to minimize CIFAR-10 reconstruction error exhaustively.

## Limitations

- The decoder is currently entirely digital.
- BatchNorm and ReLU are digital operations between physical layers.
- Training relies on global digital backpropagation.
- Experiments use a calibrated surrogate rather than a complete hardware array.
- CIFAR-10 is converted to grayscale.
- Results are based on a 5,000-image subset and short training schedules.
- The random-decoder hierarchy is a diagnostic, not a competitive model.

## Planned next step

The immediate engineering goal is a DNPU-based decoder. Because BrainSpy currently provides ordinary `DNPUConv2d` layers rather than a DNPU analogue of `ConvTranspose2d`, the planned decoder will perform spatial upsampling by inserting zeros into the feature maps and then applying ordinary DNPU convolutional layers.

That extension is not implemented in the current version of the repository.

## Status

Active research code. Interfaces, experiment organization, and scientific interpretation may change as the hardware-like decoder and more conceptually distinct network architectures are developed.
