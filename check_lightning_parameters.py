import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

from dnpu_ae.cifar_models import DNPUStackCIFARAutoencoder
from dnpu_ae.model_utils import count_parameter_breakdown
from dnpu_ae.processor import make_backend


class LitWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


CASES = [
    (
        "transpose baseline",
        dict(
            encoder_type="dnpu",
            dnpu_channels=[16, 1],
            latent_mode="raw",
            decoder_type="transpose",
            decoder_channels=[16, 1],
        ),
    ),
    (
        "DNPU nearest decoder",
        dict(
            encoder_type="dnpu",
            dnpu_channels=[16, 1],
            latent_mode="raw",
            decoder_type="dnpu_nearest_conv",
            decoder_channels=[16, 1],
        ),
    ),
    (
        "global Linear + DNPU decoder",
        dict(
            encoder_type="dnpu",
            dnpu_channels=[16, 1],
            latent_mode="raw",
            decoder_type="dnpu_nearest_conv_linear",
            decoder_channels=[16, 1],
        ),
    ),
]


def audit_case(name, config):
    backend = make_backend()
    model = DNPUStackCIFARAutoencoder(backend=backend, **config)

    lightning = ModelSummary(LitWrapper(model), max_depth=0)
    breakdown = count_parameter_breakdown(model)

    row = {
        "case": name,
        "total": breakdown["registered_parameters_total"],
        "trainable": breakdown["trainable_parameters_total"],
        "surrogate": breakdown["processor_surrogate_frozen"]["total"],
        "surrogate_train": breakdown["processor_surrogate_frozen"]["trainable"],
        "enc_ctrl": breakdown["encoder_dnpu_controls"]["total"],
        "enc_ctrl_train": breakdown["encoder_dnpu_controls"]["trainable"],
        "dec_ctrl": breakdown["decoder_dnpu_controls"]["total"],
        "dec_ctrl_train": breakdown["decoder_dnpu_controls"]["trainable"],
        "enc_bn": breakdown["encoder_batchnorm"]["trainable"],
        "dec_bn": breakdown["decoder_batchnorm"]["trainable"],
        "digital": breakdown["other_digital"]["trainable"],
    }

    assert lightning.total_parameters == row["total"]
    assert lightning.trainable_parameters == row["trainable"]
    assert row["surrogate_train"] == 0
    assert row["enc_ctrl_train"] == row["enc_ctrl"]
    assert row["dec_ctrl_train"] == row["dec_ctrl"]
    assert all(not p.requires_grad for p in backend.processor.parameters())

    return row


def print_table(rows):
    print(
        f'{"case":<30} {"total":>10} {"trainable":>10} {"surrogate":>10} '
        f'{"enc ctrl":>9} {"dec ctrl":>9} {"enc BN":>8} {"dec BN":>8} {"digital":>10}'
    )
    print("-" * 111)

    for r in rows:
        print(
            f'{r["case"]:<30} {r["total"]:>10d} {r["trainable"]:>10d} '
            f'{r["surrogate"]:>10d} {r["enc_ctrl"]:>9d} {r["dec_ctrl"]:>9d} '
            f'{r["enc_bn"]:>8d} {r["dec_bn"]:>8d} {r["digital"]:>10d}'
        )


def main():
    rows = [audit_case(name, config) for name, config in CASES]
    print_table(rows)

    print("\nChecks:")
    print("  Lightning total == local total          OK")
    print("  Lightning trainable == local trainable  OK")
    print("  surrogate trainable == 0                OK")
    print("  all DNPU control_voltages trainable     OK")


if __name__ == "__main__":
    main()
