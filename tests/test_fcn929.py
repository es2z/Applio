"""FCN-929 / FCN-929-RVC, and pins that adding them left FCN-993 unchanged."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rvc.lib.predictors.f0_methods import FCN_METHODS, fcn_variant
from rvc.lib.predictors.fcn.adapter import (
    DEFAULT_WEIGHT,
    FCNPredictor,
    FCNRVCAdapter,
    default_weight,
    grid_geometry,
)
from rvc.lib.predictors.fcn.model import ARCHITECTURES, FCNModel
from rvc.lib.predictors.fcn.preprocess import FCNPreprocessor, sliding_norm
from rvc.lib.predictors.fcn.profiles import FCNProfile, resolve_profile

WEIGHT_929 = default_weight("fcn-929")
needs_929 = pytest.mark.skipif(
    not WEIGHT_929.is_file(), reason="Local FCN_929 conversion is required"
)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# SHA-256 of the converted fcn-993.pt, and the fingerprints FCN-993 had before
# FCN-929 existed. Extraction reuse compares these, so they must never move.
WEIGHT_993_SHA256 = "4ebcd1d8bd16cd4b3db784ec3514a5abbaa89aacfc8cc94693b0c69997f193a9"
FINGERPRINTS_993 = {
    "fcn-993": "c67f4b41b31cf42aeafc991cbe350165668bd8a0039d7a35acecab07f38d9ae7",
    "fcn-993-rvc": "d734c4ee4b65d4dcbcf856df64a328ba00580f9a038143ec7e202d5eefe2b7cb",
}


def test_methods_and_variants():
    assert FCN_METHODS == ("fcn-993", "fcn-993-rvc", "fcn-929", "fcn-929-rvc")
    assert fcn_variant("fcn-929-rvc") == ("fcn-929", True)
    assert fcn_variant("fcn-993") == ("fcn-993", False)
    with pytest.raises(ValueError):
        fcn_variant("fcn-1953")
    assert grid_geometry("fcn-993") == (10, 16)
    assert grid_geometry("fcn-929-rvc") == (20, 8)


@pytest.mark.parametrize(
    "length,frames", [(929, 1), (932, 1), (933, 2), (1001, 19), (1601, 169)]
)
def test_929_network_geometry(length, frames):
    """Two 2x pools: stride 4 at 8 kHz, receptive field 929 (upstream core.py)."""
    model = FCNModel("fcn-929")
    torch.set_num_threads(2)
    with torch.inference_mode():
        result = model(torch.zeros(1, 1, length))
    assert result.shape == (1, frames, 486)
    assert frames == (length - 929) // 4 + 1
    assert [model.get_submodule(f"conv{i}").kernel_size[0] for i in range(1, 6)] == [
        32, 64, 64, 64, 64,
    ]
    assert not hasattr(model, "conv6")


def test_993_state_dict_keys_unchanged():
    keys = list(FCNModel().state_dict())
    assert keys[0] == "conv1.weight" and "conv6.weight" in keys and "bn6.running_var" in keys
    assert ARCHITECTURES["fcn-993"]["receptive_field"] == 993


@pytest.mark.parametrize("boundary", ["wrap", "constant"])
def test_929_normalization_window_is_930(boundary):
    """Upstream sliding_norm rounds the odd 929 up to a 930-sample window."""
    x = np.random.default_rng(929).normal(size=1300).astype(np.float32)
    padded = np.pad(x, 465, mode=boundary)
    frames = np.lib.stride_tricks.as_strided(
        padded, shape=(930, len(x)), strides=(4, 4)
    ).T
    std = frames.std(axis=1)
    std[std == 0] = np.finfo(np.float32).eps
    expected = (x - frames.mean(axis=1)) / std
    np.testing.assert_array_equal(
        sliding_norm(x, boundary, block_samples=37, window=930), expected
    )
    assert len(FCNPreprocessor(boundary, 929)(np.zeros(1600, np.float32))) == 800 + 928


def test_929_rvc_default_profile():
    profile = resolve_profile("fcn-929-rvc")
    assert (profile.enter_threshold, profile.exit_threshold, profile.median_frames) == (
        0.6,
        0.35,
        9,
    )
    assert not profile.calibrated and not profile.compile_model
    assert resolve_profile("fcn-929").enter_threshold is None


def test_median_frames_are_native_frames_per_architecture():
    FCNProfile(method="fcn-929-rvc", enter_threshold=0.5, exit_threshold=0.4, median_frames=17)
    with pytest.raises(ValueError, match="median_frames"):
        FCNProfile(method="fcn-929-rvc", enter_threshold=0.5, exit_threshold=0.4, median_frames=3)
    with pytest.raises(ValueError, match="median_frames"):
        FCNProfile(method="fcn-993-rvc", enter_threshold=0.5, exit_threshold=0.4, median_frames=17)
    with pytest.raises(ValueError, match="Baseline"):
        FCNProfile(method="fcn-929", enter_threshold=0.5, exit_threshold=0.4)
    with pytest.raises(ValueError, match="does not match"):
        resolve_profile("fcn-929-rvc", resolve_profile("fcn-993-rvc"))


def test_993_fingerprints_are_unchanged():
    for method, fingerprint in FINGERPRINTS_993.items():
        assert resolve_profile(method).fingerprint(WEIGHT_993_SHA256) == fingerprint


def test_929_fingerprint_differs_from_993():
    """Same profile values, different network: never reuse one's pitch for the other."""
    a = FCNProfile(method="fcn-993-rvc", enter_threshold=0.5, exit_threshold=0.4, median_frames=5)
    b = FCNProfile(method="fcn-929-rvc", enter_threshold=0.5, exit_threshold=0.4, median_frames=5)
    assert a.fingerprint("w") != b.fingerprint("w")


def test_929_rvc_adapter_uses_twenty_native_frames_per_hop():
    """The 10 ms window is -5..+4.5 ms either way: 20 half-millisecond frames."""
    profile = FCNProfile(method="fcn-929-rvc", enter_threshold=0.7, exit_threshold=0.4)
    confidence = np.repeat([0.8, 0.5, 0.1, 0.5, 0.8], 20).astype(np.float32)
    cents = np.full(100, 1200 * np.log2(22))
    track = FCNRVCAdapter(profile)(cents, confidence, 5, np.ones(800))
    assert track.voiced.tolist() == [True, True, False, False, False]
    np.testing.assert_allclose(track.pitch_hz[:2], 220, rtol=1e-6)


@needs_929
def test_929_weight_and_manifest():
    predictor = FCNPredictor("cpu", "fcn-929")
    assert predictor.weight_path == WEIGHT_929
    assert predictor.metadata()["architecture"] == "fcn-929"
    assert predictor.metadata()["normalization"] == "930-population-wrap"
    with pytest.raises(ValueError, match="architecture"):
        FCNPredictor("cpu", "fcn-929", weight_path=DEFAULT_WEIGHT)


@needs_929
def test_with_profile_never_crosses_architectures():
    predictor = FCNPredictor("cpu", "fcn-929")
    assert predictor.with_profile("fcn-929-rvc").model is predictor.model
    with pytest.raises(ValueError, match="fcn-993 network"):
        predictor.with_profile("fcn-993-rvc")


@needs_929
@pytest.mark.parametrize("method", ["fcn-929", "fcn-929-rvc"])
def test_929_grid_is_native_decimation(method):
    audio = (0.2 * np.sin(np.arange(3200) * 2 * np.pi * 220 / 16000)).astype(np.float32)
    predictor = FCNPredictor("cpu", method, block_frames=64)
    cents, hz, confidence = predictor.native(audio)
    # 464 samples of padding on each side of the 8 kHz signal: one frame per 0.5 ms.
    assert len(cents) == (len(audio) // 2 - 1) // 4 + 1
    track = predictor.extract_track(audio)
    assert len(track.pitch_hz) == 20
    if method == "fcn-929":
        np.testing.assert_array_equal(track.pitch_hz, hz[::20][:20])
    voiced = track.pitch_hz[track.voiced]
    assert voiced.size and np.all(np.abs(1200 * np.log2(voiced / 220)) < 30)


@needs_929
@needs_cuda
@pytest.mark.parametrize("method", ["fcn-929", "fcn-929-rvc"])
def test_929_cuda_matches_cpu_and_streaming(method):
    from rvc.lib.predictors.fcn.streaming import FCNStream

    rng = np.random.default_rng(929)
    t = np.arange(24000) / 16000
    audio = (0.3 * np.sin(2 * np.pi * 150 * t * (1 + 0.2 * t))).astype(np.float32)
    audio += 0.01 * rng.normal(size=len(t)).astype(np.float32)
    audio[:3200] = 0
    gpu = FCNPredictor("cuda", method)
    cpu = FCNPredictor("cpu", method).extract_track(audio)
    track = gpu.extract_track(audio)
    np.testing.assert_array_equal(track.voiced, cpu.voiced)
    both = track.voiced
    assert np.max(np.abs(1200 * np.log2(track.pitch_hz[both] / cpu.pitch_hz[both]))) < 0.1
    stream = FCNStream(gpu)
    assert stream.holdback_samples == (2240 if method.endswith("-rvc") else 2080)
    pieces = [stream.push(audio[i : i + 1999]) for i in range(0, len(audio), 1999)]
    streamed = torch.cat([p.pitch_hz for p in pieces + [stream.flush()]]).cpu().numpy()
    assert len(streamed) == len(track.pitch_hz)
    # The baseline's start is zero-extended in streaming instead of wrapped.
    np.testing.assert_allclose(streamed[40:], track.pitch_hz[40:], atol=1e-3, rtol=0)


@pytest.mark.skipif(not DEFAULT_WEIGHT.is_file(), reason="Local FCN_993 conversion is required")
@needs_cuda
def test_993_holdback_unchanged():
    from rvc.lib.predictors.fcn.streaming import FCNStream

    for method in ("fcn-993", "fcn-993-rvc"):
        assert FCNStream(FCNPredictor("cuda", method)).holdback_samples == 2240


@needs_929
@needs_cuda
def test_929_training_matches_inference_and_switch_reloads_weights():
    from rvc.infer.pipeline import Pipeline
    from rvc.train.extract.extract import FeatureInput

    audio = (0.1 * np.sin(np.arange(4800) * 2 * np.pi * 220 / 16000)).astype(np.float32)
    config = SimpleNamespace(x_pad=1, x_query=1, x_center=2, x_max=3, device="cuda")
    pipeline = Pipeline(40000, config)
    first = pipeline.configure_fcn("fcn-929-rvc")
    _, hz = pipeline.get_f0(audio, 30, "fcn-929-rvc")
    np.testing.assert_array_equal(hz, FeatureInput("fcn-929-rvc", "cuda").compute_f0(audio))
    assert pipeline.configure_fcn("fcn-929").model is first.model
    if DEFAULT_WEIGHT.is_file():
        other = pipeline.configure_fcn("fcn-993-rvc")
        assert other.model is not first.model and other.architecture_id == "fcn-993"
        assert pipeline.configure_fcn("fcn-929-rvc").architecture_id == "fcn-929"


@needs_929
def test_929_extraction_spec_forces_reextraction_from_993():
    from rvc.train.extract.fcn_metadata import can_reuse, extraction_spec

    spec = extraction_spec("fcn-929-rvc")
    assert spec["architecture"] == "fcn-929"
    assert spec["normalization"] == "930-population-zero"
    previous = {"complete": True, "input_signature": "s", "specification": spec}
    assert can_reuse(previous, extraction_spec("fcn-929-rvc"), "s")
    if DEFAULT_WEIGHT.is_file():
        assert not can_reuse(previous, extraction_spec("fcn-993-rvc"), "s")


@needs_929
@needs_cuda
def test_929_realtime_session_shares_absolute_grid():
    import torchaudio.transforms as tat

    from rvc.realtime.fcn_session import FCNRealtimeSession

    predictor = FCNPredictor("cuda", "fcn-929-rvc")
    resampler = tat.Resample(48000, 16000).cuda()
    session = FCNRealtimeSession(predictor, resampler, 8000)
    audio = 0.1 * torch.sin(torch.arange(48001, device="cuda") * (2 * torch.pi * 220 / 48000))
    reference_audio = resampler(audio)
    reference_pitch = predictor.extract_track(reference_audio).pitch_hz
    pos = 0
    for size in [31, 200, 8000, 1547, 20000, 18223]:
        window, pitch = session.push(audio[pos : pos + size])
        pos += size
        end = session.window_end
        count = min(8000, end)
        if count:
            torch.testing.assert_close(
                window[-count:], reference_audio[end - count : end], atol=1e-6, rtol=1e-5
            )
            frames = count // 160
            expected = reference_pitch[end // 160 - frames : end // 160]
            actual = pitch[-frames:].cpu().numpy()
            voiced = (expected > 0) & (actual > 0)
            np.testing.assert_array_equal(actual > 0, expected > 0)
            assert np.max(np.abs(1200 * np.log2(actual[voiced] / expected[voiced]))) < 0.1
    assert session.holdback_ms == pytest.approx(2240 / 16 + session.capture.lookahead / 48)
