import numpy as np
import pytest
import torch

from rvc.lib.predictors.fcn.decoder import FCNDecoder
from rvc.lib.predictors.fcn.model import FCNModel
from rvc.lib.predictors.fcn.preprocess import sliding_norm


@pytest.mark.parametrize("length,frames", [(993, 1), (1000, 1), (1001, 2), (1601, 77)])
def test_valid_network_geometry(length, frames):
    model = FCNModel()
    torch.set_num_threads(2)
    with torch.inference_mode():
        result = model(torch.zeros(1, 1, length))
    assert result.shape == (1, frames, 486)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("kind", ["random", "silence", "dc", "tiny"])
def test_normalization_matches_original_float32_reductions(kind):
    x = np.random.default_rng(9).normal(size=1600).astype(np.float32)
    if kind == "silence":
        x[:] = 0
    elif kind == "dc":
        x[:] = 0.125
    elif kind == "tiny":
        x *= 1e-12
    padded = np.pad(x, 497, mode="wrap")
    frames = np.lib.stride_tricks.as_strided(
        padded, shape=(994, len(x)), strides=(4, 4)
    ).T
    std = frames.std(axis=1)
    std[std == 0] = np.finfo(np.float32).eps
    expected = (x - frames.mean(axis=1)) / std
    np.testing.assert_array_equal(sliding_norm(x, block_samples=37), expected)


def test_decoder_edges_and_degenerate_activation():
    activation = torch.zeros(4, 486)
    activation[0, 0] = 1
    activation[1, -1] = 1
    activation[2, 200:209] = torch.arange(1, 10)
    cents, hz, confidence = FCNDecoder()(activation)
    assert hz[0].item() == pytest.approx(30)
    assert hz[1].item() == pytest.approx(1000)
    assert hz[3].item() == 0
    mapping = np.linspace(1200 * np.log2(3), 1200 * np.log2(100), 486)
    assert cents[2].item() == pytest.approx(
        np.average(mapping[204:213], weights=activation[2, 204:213].numpy())
    )
    assert confidence.tolist() == [1, 1, 9, 0]
