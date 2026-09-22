import numpy as np
import pytest
import torch

from rvc.lib.predictors.fcn.adapter import DEFAULT_WEIGHT, FCNPredictor


@pytest.fixture(scope="module")
def predictor():
    if not DEFAULT_WEIGHT.is_file():
        pytest.skip("Local original-weight conversion is required for this test")
    torch.set_num_threads(2)
    return FCNPredictor(block_frames=32)


def test_local_asset_required(tmp_path):
    with pytest.raises(FileNotFoundError, match="convert_fcn993"):
        FCNPredictor(weight_path=tmp_path / "missing.pt")


def test_grid_is_native_decimation_and_block_context_is_real(predictor):
    audio = np.sin(np.arange(1601) * 2 * np.pi * 220 / 16000).astype(np.float32)
    cents, _hz, _confidence, activation = predictor.native(
        audio, return_activation=True
    )
    predictor.block_frames = 256
    try:
        whole = predictor.native(audio, return_activation=True)
        np.testing.assert_allclose(activation, whole[3], atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(cents, whole[0], atol=0.1, rtol=0)
        track = predictor.extract_track(audio)
        assert len(track.pitch_hz) == 10
        np.testing.assert_allclose(track.pitch_hz, whole[1][::10][:10], atol=0, rtol=0)
        np.testing.assert_array_equal(track.timestamps, np.arange(10) * 0.01)
        assert track.pitch_hz.dtype == np.float32
    finally:
        predictor.block_frames = 32


def test_input_contract_and_silence_is_not_gated(predictor):
    assert not len(predictor.get_f0(np.zeros(0)))
    assert not len(predictor.get_f0(np.zeros(159)))
    with pytest.raises(ValueError, match="explicitly pad"):
        predictor.get_f0(np.zeros(160), p_len=2)
    with pytest.raises(ValueError, match="finite mono"):
        predictor.get_f0(np.array([np.nan]))
    with pytest.raises(ValueError, match="finite mono"):
        predictor.get_f0(np.zeros((2, 160)))
    assert np.all(predictor.get_f0(np.zeros(160)) > 0)
