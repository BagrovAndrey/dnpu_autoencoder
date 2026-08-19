from pathlib import Path

import torch

from brainspy.processors.processor import Processor
from dnpu_ae.dnpu_conv import DNPUConv2d_DNPUChild


def make_processor():
    ckpt = torch.load(Path("surrogate_model.pt"), map_location="cpu")

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

    return processor


def main():
    torch.manual_seed(0)

    processor = make_processor()

    # 2x2 DNPU convolution:
    # one DNPU node, four data electrodes, three control electrodes.
    conv = DNPUConv2d_DNPUChild(
        processor=processor,
        data_input_indices=[[0, 1, 2, 3]],
        in_channels=1,
        out_channels=4,
        kernel_size=2,
        stride=2,
        padding=0,
        forward_pass_type="vec",
    )

    x = torch.rand(8, 1, 32, 32) - 0.5

    y = conv(x)

    print("x shape:", tuple(x.shape))
    print("y shape:", tuple(y.shape))
    print("y mean:", y.mean().item())
    print("y std:", y.std().item())
    print("y finite:", torch.isfinite(y).all().item())

    loss = y.pow(2).mean()
    loss.backward()

    grad_ok = True
    n_params = 0

    for name, param in conv.named_parameters():
        n_params += param.numel()
        if param.grad is None:
            print("no grad:", name, tuple(param.shape))
        else:
            finite = torch.isfinite(param.grad).all().item()
            print("grad:", name, tuple(param.shape), "finite:", finite)
            grad_ok = grad_ok and finite

    n_trainable = sum(p.numel() for p in conv.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in conv.parameters())

    print("n total params", n_total)
    print("n trainable params:", n_trainable)
    print("all grads finite:", grad_ok)


if __name__ == "__main__":
    main()
