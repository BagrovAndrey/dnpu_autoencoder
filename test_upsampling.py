import pytest
import torch

from dnpu_ae.upsampling import (
    DigitalZeroConvDecoder,
    _pad_kernel2_same_size,
    parse_decoder_stage_specs,
    zero_insert_upsample_2d,
)


def test_zero_insert_shape():
    x = torch.randn(2, 3, 4, 5)
    y = zero_insert_upsample_2d(x, scale_factor=2)
    assert y.shape == (2, 3, 8, 10)


def test_zero_insert_values_at_even_coordinates():
    x = torch.randn(2, 3, 4, 5)
    y = zero_insert_upsample_2d(x, scale_factor=2)
    assert torch.equal(y[..., ::2, ::2], x)


def test_zero_inserted_positions_are_zero():
    x = torch.randn(2, 3, 4, 5)
    y = zero_insert_upsample_2d(x, scale_factor=2)

    odd_row_mask = torch.ones_like(y, dtype=torch.bool)
    odd_row_mask[..., ::2, :] = False
    odd_col_mask = torch.ones_like(y, dtype=torch.bool)
    odd_col_mask[..., :, ::2] = False

    assert torch.count_nonzero(y[odd_row_mask]) == 0
    assert torch.count_nonzero(y[odd_col_mask]) == 0


def test_zero_insert_gradients_propagate():
    x = torch.randn(2, 3, 4, 5, requires_grad=True)
    y = zero_insert_upsample_2d(x, scale_factor=2)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("pad_mode", ["bottom_right", "top_left"])
def test_kernel2_same_size_padding_gives_expected_spatial_shape(pad_mode):
    x = torch.randn(2, 3, 8, 8)
    y = _pad_kernel2_same_size(x, pad_mode)
    assert y.shape == (2, 3, 9, 9)


def test_digital_two_stage_decoder_maps_1x8x8_to_1x32x32():
    decoder = DigitalZeroConvDecoder(
        raw_channels=1,
        raw_spatial_size=8,
        decoder_channels=[16, 1],
    )
    x = torch.randn(4, 1, 8, 8)
    y = decoder(x)
    assert y.shape == (4, 1, 32, 32)


def test_digital_two_stage_mixing_decoder_maps_1x8x8_to_1x32x32():
    decoder = DigitalZeroConvDecoder(
        raw_channels=1,
        raw_spatial_size=8,
        decoder_channels=[16, 1],
        use_mixing=True,
    )
    x = torch.randn(4, 1, 8, 8)
    y = decoder(x)
    assert y.shape == (4, 1, 32, 32)


def test_mixing_decoder_has_extra_convs_per_stage():
    decoder = DigitalZeroConvDecoder(
        raw_channels=1,
        raw_spatial_size=8,
        decoder_channels=[16, 1],
        use_mixing=True,
    )
    assert len(decoder.layers) == 2
    assert len(decoder.mixing_layers) == 2
    assert len(decoder.norm_layers) == 2
    assert len(decoder.mixing_norm_layers) == 1


@pytest.mark.parametrize(
    "raw_spatial_size,decoder_channels,error_text",
    [
        (6, [16, 1], "must divide target size"),
        (10, [16, 1], "must divide target size"),
        (8, [1], "requires 2 factor-two upsampling stages"),
        (8, [16, 2], "last decoder channel must be 1"),
        (4, [16, 8], "requires 3 factor-two upsampling stages"),
    ],
)
def test_invalid_decoder_configurations_raise_informative_errors(
    raw_spatial_size,
    decoder_channels,
    error_text,
):
    with pytest.raises(ValueError, match=error_text):
        parse_decoder_stage_specs(
            raw_spatial_size=raw_spatial_size,
            raw_channels=1,
            decoder_channels=decoder_channels,
        )
