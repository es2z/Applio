import numpy as np
import pytest
import resampy
import torch

from rvc.lib.predictors.fcn.adapter import (
    DEFAULT_WEIGHT,
    FCNPredictor,
    FCNRVCAdapter,
    tensor_grid,
)
from rvc.lib.predictors.fcn.preprocess import FCNPreprocessor, FCNTensorPreprocessor
from rvc.lib.predictors.fcn.profiles import FCNProfile
from rvc.lib.predictors.fcn.streaming import FCNStream

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def predictor():
    if not DEFAULT_WEIGHT.exists():
        pytest.skip("Convert FCN weights first")
    return FCNPredictor("cuda")


@pytest.mark.parametrize("kind", ["sine", "noise", "dc", "silence", "tiny", "impulse"])
def test_cuda_preprocessing_reference(kind):
    n = 8001
    wave = np.sin(np.arange(n) * (2 * np.pi * 220 / 16000)).astype(np.float32)
    if kind == "noise":
        wave = np.random.default_rng(993).normal(size=n).astype(np.float32)
    elif kind == "dc":
        wave[:] = 0.125
    elif kind == "silence":
        wave[:] = 0
    elif kind == "tiny":
        wave *= 1e-12
    elif kind == "impulse":
        wave[:] = 0
        wave[0] = 1
    prep = FCNTensorPreprocessor("cuda")
    with torch.backends.cudnn.flags(allow_tf32=False):
        audio = torch.from_numpy(wave).cuda()
        actual = prep(audio).cpu().numpy()
        resampled = prep.resample(audio).cpu().numpy()
    expected = FCNPreprocessor()(wave)
    np.testing.assert_allclose(
        resampled, resampy.resample(wave, 16000, 8000), atol=2e-6, rtol=1e-5
    )
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("median", [0, 3, 5, 9])
def test_tensor_adapter_matches_numpy(median):
    rng = np.random.default_rng(28)
    cents = rng.uniform(3000, 6000, 101)
    confidence = rng.uniform(0, 1, 101).astype(np.float32)
    audio = np.ones(1601, np.float32)
    profile = FCNProfile(
        method="fcn-993-rvc",
        enter_threshold=0.6,
        exit_threshold=0.4,
        median_frames=median,
    )
    expected = FCNRVCAdapter(profile)(cents, confidence, 10, audio)
    hz, voiced, conf = tensor_grid(
        torch.tensor(cents, device="cuda"),
        torch.tensor(confidence, device="cuda"),
        torch.tensor(audio, device="cuda"),
        torch.arange(10, device="cuda") * 10,
        profile,
    )
    np.testing.assert_allclose(hz.cpu(), expected.pitch_hz, atol=1e-4)
    np.testing.assert_array_equal(voiced.cpu(), expected.voiced)
    np.testing.assert_allclose(conf.cpu(), expected.confidence, atol=1e-7)


@pytest.mark.parametrize("method", ["fcn-993", "fcn-993-rvc"])
def test_stream_irregular_chunks_and_reset(predictor, method):
    profile = (
        FCNProfile()
        if method == "fcn-993"
        else FCNProfile(
            method=method, enter_threshold=0.6, exit_threshold=0.4, median_frames=5
        )
    )
    model = predictor.with_profile(method, profile)
    audio = (np.sin(np.arange(16001) * 2 * np.pi * 220 / 16000) * 0.1).astype(
        np.float32
    )
    audio[6000:9000] = 0
    expected = model.extract_track(audio)
    stream = FCNStream(model)
    pieces, position = [], 0
    for size in [1, 15, 79, 160, 7, 3001, 1700, 99, 2000, 19, 4017, 4903]:
        pieces.append(stream.push(audio[position : position + size]))
        position += size
    pieces.append(stream.push(audio[position:]))
    pieces.append(stream.flush())
    hz = torch.cat([p.pitch_hz for p in pieces]).cpu().numpy()
    indices = torch.cat([p.frame_index for p in pieces]).cpu().numpy()
    voiced = torch.cat([p.voiced for p in pieces]).cpu().numpy()
    np.testing.assert_array_equal(indices, np.arange(len(audio) // 160))
    offset = 1 if method == "fcn-993" else 0
    np.testing.assert_array_equal(voiced[offset:], expected.voiced[offset:])
    valid = (hz > 0) & expected.voiced
    if method == "fcn-993":
        valid[0] = False  # baseline's documented noncausal wrap startup exception
    assert np.max(np.abs(1200 * np.log2(hz[valid] / expected.pitch_hz[valid]))) < 0.1
    assert len(stream.buffer) <= stream.context_samples + stream.holdback_samples + 160
    with pytest.raises(RuntimeError):
        stream.push(audio[:160])
    stream.reset()
    assert stream.sample_count == stream.next_frame == 0


def test_negative_stride_audio_from_highpass_filter(predictor):
    wave = np.sin(np.arange(1600) * 2 * np.pi * 220 / 16000).astype(np.float64)[::-1]
    expected = predictor.get_f0(wave.copy())
    np.testing.assert_array_equal(predictor.get_f0(wave), expected)
