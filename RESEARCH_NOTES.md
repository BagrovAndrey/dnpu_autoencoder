# Research questions and experimental interpretation

## Few trainable parameters versus expressivity

Discussion recorded on 2026-10-18, on branch `dnpu-conv-cifar`.

Good reconstruction with few trainable parameters would demonstrate useful
adaptation through a small number of adjustable controls. It would not, by
itself, establish exceptional expressivity or an advantage over digital models.

For the current hardware–hardware configuration (`dnpu_channels=16,1`,
`latent_mode=raw`, `decoder_type=dnpu_nearest_conv`,
`decoder_channels=16,1`), the verified counts are:

| Parameter group | Count | Trainable |
|---|---:|---|
| Encoder DNPU controls | 96 | Yes |
| Decoder DNPU controls | 96 | Yes |
| Encoder BatchNorm affine parameters | 34 | Yes |
| Decoder BatchNorm affine parameters | 34 | Yes |
| Shared surrogate parameters | 33,571 | No |
| Total trainable | 260 | Yes |

Here, “hardware” means surrogate-simulated DNPU operations. BatchNorm, ReLU,
routing, upsampling, and optimization remain digital. The 64-value raw latent
is taken after encoder BatchNorm and ReLU. BatchNorm running statistics adapt
during training but are not included in the trainable-parameter count.

Distinguish three quantities when interpreting results:

1. The number of adjustable parameters used for task learning.
2. The capability supplied by the fixed nonlinear response and architecture.
3. The computational and physical cost of implementing that response.

Frozen nonlinear structure can supply substantial capability. The surrogate's
33,571 weights are one numerical approximation to a calibrated device response;
their count is neither the device's parameter count nor a direct measure of its
expressivity. Assuming surrogate fidelity, those weights describe the physics
rather than provide independently adjustable task parameters. Nevertheless,
digital simulation evaluates that network, and a low trainable-parameter count
alone does not establish low digital compute or memory use.

### Hypothesis to test

A conventional digital CNN autoencoder with all but 260 parameters frozen might
achieve comparable reconstruction. This is an open hypothesis, not an established
result. Matching counts alone does not guarantee matching behavior: a DNPU
control changes a nonlinear response, whereas an unfrozen convolution weight
changes a coefficient. Parameter placement, initialization, and fixed structure
matter.

### Proposed comparisons

| Model | Purpose |
|---|---|
| Current DNPU model: 192 controls + 68 BatchNorm parameters | Reference |
| Frozen random digital convolutions with 192 distributed trainable weights + the same 68 BatchNorm parameters | Direct test of the parameter-budget hypothesis |
| Frozen nonlinear digital modules with three trainable control inputs per input/output-channel pair, plus matched normalization | Compare a similar adjustment mechanism with a different fixed response |
| Fully trainable small digital CNN | Ordinary digital reconstruction baseline |
| DNPU controls frozen, BatchNorm affine parameters trainable | Isolate the contribution of learning controls from digital calibration |

Match the latent dimension, spatial layout, normalization, data subsets, loss,
and training budget where possible. Document unavoidable differences. Repeat
across initialization seeds, specify how the 192 trainable digital weights are
distributed, and report the BatchNorm running-statistics policy. Separate task
training from any prior fitting or selection of the fixed response. Report
trainable and frozen parameter counts separately from runtime and memory cost.

### Claim boundaries and current evidence

- A possible supported claim is “useful reconstruction through a small number
  of physical controls,” once reconstruction is demonstrated.
- Superior expressivity or parameter efficiency requires matched digital
  comparisons; it does not follow from the number 260 alone.
- Hardware efficiency requires physical measurements, including surrounding
  digital operations; surrogate-only results do not establish it.
- The first local smoke run used 32 training images, 20 epochs (20 optimizer
  steps), L1 loss, and seed 0. Training MAE decreased from about 0.1896 to
  0.1874; final held-out MAE was about 0.2002. Reconstructions remained mostly
  gray with sparse artifacts, so successful small-data overfitting has not yet
  been demonstrated. Controls changed and surrogate weights stayed frozen.
- Local artifacts are under `results_smoke_hardware_hardware_32/`, including
  `training.log`, `metrics.json`, reconstruction grids, and a checkpoint. They
  are ignored by Git and may not be available in another checkout.

Related background: [Weight Agnostic Neural Networks](https://arxiv.org/abs/1906.04358)
illustrates the contribution of fixed architecture with little weight adjustment;
it does not establish the outcome of the proposed autoencoder comparisons.
