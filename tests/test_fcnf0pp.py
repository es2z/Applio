import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rvc.lib.predictors.fcnf0pp.profiles import (
    FCNF0PPProfile,
    default_profile,
    resolve_profile,
)
from rvc.lib.predictors.fcnf0pp.weights import DEFAULT_WEIGHT, manifest_path

SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def profile(method="fcnf0++", **changes):
    values = default_profile(method).to_dict()
    values.update(changes)
    return values


def harmonic(frequency, seconds=2.0, harmonics=5):
    """A tone with a known pitch; frequency may be a per-sample array."""
    samples = int(SR * seconds)
    frequency = np.broadcast_to(np.asarray(frequency, dtype=np.float64), (samples,))
    phase = 2 * np.pi * np.cumsum(frequency) / SR
    return (sum(np.sin(k * phase) / k for k in range(1, harmonics + 1)) * 0.3).astype(
        np.float32
    )


def glide(low, high, seconds=4.0, harmonics=1):
    t = np.arange(int(SR * seconds)) / SR
    return harmonic(low * (high / low) ** (t / seconds), seconds, harmonics)


def glide_offset_ms(f0, start, end, seconds=4.0):
    """Median of (when the reported pitch actually sounded) - (frame index * 10 ms)."""
    f0 = np.asarray(f0, dtype=np.float64)
    index = np.arange(len(f0))
    keep = (index >= 50) & (index <= len(f0) - 50) & (f0 > 0)
    sounded = seconds * np.log(f0[keep] / start) / np.log(end / start)
    return float(np.median(sounded - index[keep] * 0.01) * 1000)


def timing_ms(predictor, low, high):
    """Timing offset with the model's small constant pitch bias cancelled.

    A constant cents error reads as an early offset on a rising glide and a late one
    on a falling glide, so the mean of the two is timing alone.
    """
    up = glide_offset_ms(predictor.get_f0(glide(low, high)), low, high)
    down = glide_offset_ms(predictor.get_f0(glide(high, low)), high, low)
    return (up + down) / 2


# --- profiles: no model needed ---------------------------------------------------


def test_bundled_profiles():
    baseline, rvc = default_profile("fcnf0++"), default_profile("fcnf0++-rvc")
    assert baseline.periodicity_threshold is None
    assert rvc.periodicity_threshold == 0.035
    for bundled in (baseline, rvc):
        assert bundled.decoder == "viterbi"
        assert bundled.center == "zero"
        assert (bundled.coarse_min, bundled.coarse_max) == (50.0, 1680.0)


def test_profile_validation():
    with pytest.raises(ValueError, match="ungated baseline"):
        FCNF0PPProfile(method="fcnf0++", periodicity_threshold=0.1)
    with pytest.raises(ValueError, match="needs periodicity_threshold"):
        FCNF0PPProfile(method="fcnf0++-rvc")
    with pytest.raises(ValueError, match="decoder"):
        FCNF0PPProfile(decoder="pyin")
    with pytest.raises(ValueError, match="center"):
        FCNF0PPProfile(center="half-window")  # +64 ms and too few frames for p_len
    with pytest.raises(ValueError, match="50–1680"):
        FCNF0PPProfile(coarse_max=2000.0)
    with pytest.raises(ValueError, match="periodicity_threshold"):
        FCNF0PPProfile(method="fcnf0++-rvc", periodicity_threshold=1.5)


def test_profile_resolution_order(tmp_path):
    explicit = profile("fcnf0++-rvc", periodicity_threshold=0.2)
    checkpoint = profile("fcnf0++-rvc", periodicity_threshold=0.3)
    assert resolve_profile("fcnf0++-rvc").periodicity_threshold == 0.035
    assert resolve_profile("fcnf0++-rvc", None, checkpoint).periodicity_threshold == 0.3
    assert resolve_profile("fcnf0++-rvc", "  ", checkpoint).periodicity_threshold == 0.3
    assert (
        resolve_profile("fcnf0++-rvc", json.dumps(explicit), checkpoint).periodicity_threshold
        == 0.2
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(explicit), encoding="utf-8")
    assert resolve_profile("fcnf0++-rvc", str(path)).periodicity_threshold == 0.2
    # A checkpoint trained with another method never leaks its settings in.
    fcn_checkpoint = {"method": "fcn-993-rvc", "enter_threshold": 0.5}
    assert resolve_profile("fcnf0++-rvc", None, fcn_checkpoint) == default_profile("fcnf0++-rvc")
    with pytest.raises(ValueError, match="is selected"):
        resolve_profile("fcnf0++", json.dumps(explicit))


def test_extraction_spec_changes_with_every_setting():
    if not DEFAULT_WEIGHT.is_file():
        pytest.skip("FCNF0++ weights are required")
    from rvc.train.extract.fcn_metadata import can_reuse, extraction_spec

    base = extraction_spec("fcnf0++-rvc")
    assert base["grid"] == {"sample_rate": 16000, "origin": 0, "hop": 160}
    assert base["interp_unvoiced_at"] is None
    for change in (
        {"periodicity_threshold": 0.2},
        {"decoder": "argmax"},
        {"center": "half-hop"},
        {"coarse_max": 1100.0},
    ):
        other = extraction_spec("fcnf0++-rvc", profile("fcnf0++-rvc", **change))
        assert other != base and other["fingerprint"] != base["fingerprint"]
    # Without a completed record, profile methods always extract.
    assert not can_reuse(None, base, "signature")
    record = {"complete": True, "specification": base, "input_signature": "signature"}
    assert can_reuse(record, base, "signature")


# --- the network -------------------------------------------------------------------


@pytest.fixture(scope="module")
def penn():
    return pytest.importorskip("penn")


def make(method="fcnf0++", device=DEVICE, **changes):
    if not DEFAULT_WEIGHT.is_file():
        pytest.skip("FCNF0++ weights are required (tools/strip_fcnf0pp.py)")
    from rvc.lib.predictors.fcnf0pp import FCNF0PPPredictor

    return FCNF0PPPredictor(device, method, profile(method, **changes))


def test_weights_are_stripped_and_checksum_pinned(penn):
    from rvc.lib.predictors.fcnf0pp.weights import (
        PARAMETER_COUNT,
        load_state_dict,
        sha256,
    )

    manifest = json.loads(manifest_path(DEFAULT_WEIGHT).read_text(encoding="utf-8"))
    assert manifest["weight_sha256"] == sha256(DEFAULT_WEIGHT)
    state, digest = load_state_dict()
    assert digest == manifest["weight_sha256"]
    assert sum(value.numel() for value in state.values()) == PARAMETER_COUNT
    model = penn.Model()
    model.load_state_dict(state, strict=True)


def test_missing_weight_is_reported(tmp_path, penn):
    from rvc.lib.predictors.fcnf0pp import FCNF0PPPredictor

    with pytest.raises(FileNotFoundError, match="strip_fcnf0pp"):
        FCNF0PPPredictor("cpu", weight_path=tmp_path / "missing.pt")


@pytest.mark.parametrize("decoder", ["viterbi", "argmax"])
def test_bit_identical_to_penn_from_audio(penn, decoder):
    """Everything numeric is penn's; only the caching around it is ours."""
    audio = np.concatenate(
        [glide(120, 300, 1.0, 5), np.zeros(1600, np.float32), harmonic(220, 0.5)]
    )
    predictor = make(device="cpu", decoder=decoder)
    track = predictor.extract_track(audio)
    pitch, periodicity = penn.from_audio(
        torch.from_numpy(audio)[None],
        SR,
        0.01,
        50.0,
        1680.0,
        DEFAULT_WEIGHT,
        2048,
        "zero",
        decoder,
        None,
        None,
    )
    p_len = len(audio) // 160
    assert pitch.shape[-1] == p_len + 1  # "zero" framing gives one extra frame
    np.testing.assert_array_equal(track.raw_pitch_hz, pitch[0, :p_len].numpy())
    np.testing.assert_array_equal(track.periodicity, periodicity[0, :p_len].numpy())


@pytest.mark.parametrize("samples", [16000, 16001, 24000, 100000])
def test_frame_count_is_the_rvc_grid(samples):
    predictor = make()
    audio = harmonic(200, samples / SR)
    track = predictor.extract_track(audio)
    assert len(track.pitch_hz) == samples // 160
    np.testing.assert_array_equal(track.timestamps, np.arange(samples // 160) * 0.01)
    assert len(predictor.get_f0(audio, 10)) == 10
    with pytest.raises(ValueError, match="10 ms grid"):
        predictor.get_f0(audio, samples // 160 + 1)


def test_short_input_is_refused():
    with pytest.raises(ValueError, match="reflect padding"):
        make().get_f0(np.zeros(1024, np.float32))


def test_zero_center_puts_the_window_on_the_rvc_grid():
    """A pure tone glide measures where each window sits: frame i at t = i * 10 ms.

    With harmonics in the speech range the reported pitch runs ~11 ms late even with
    "zero"; that lag is the model's own and is not corrected (docs/fcnf0pp.md).
    """
    zero = timing_ms(make(decoder="argmax"), 300, 800)
    half_hop = timing_ms(make(decoder="argmax", center="half-hop"), 300, 800)
    assert abs(zero) < 2.0
    # half-hop centres frame i at i * 10 ms + 5 ms, so its pitch reads 5 ms early.
    assert half_hop - zero == pytest.approx(5.0, abs=1.5)


def test_periodicity_is_penn_entropy_and_independent_of_the_decoder(penn):
    audio = np.concatenate([np.zeros(8000, np.float32), harmonic(180, 1.0)])
    viterbi = make(decoder="viterbi").extract_track(audio)
    argmax = make(decoder="argmax").extract_track(audio)
    np.testing.assert_array_equal(viterbi.periodicity, argmax.periodicity)
    assert np.isfinite(viterbi.periodicity).all()
    assert ((viterbi.periodicity >= 0) & (viterbi.periodicity <= 1)).all()
    # Digital silence gives a flat posterior over the allowed bins, whose entropy
    # floor is 1 - log(K) / log(PITCH_BINS) for the 50-1680 Hz range.
    low = int(penn.convert.frequency_to_bins(torch.tensor(50.0)))
    high = int(penn.convert.frequency_to_bins(torch.tensor(1680.0), torch.ceil))
    floor = 1 - math.log(high - low) / math.log(penn.PITCH_BINS)
    assert np.median(viterbi.periodicity[5:40]) == pytest.approx(floor, abs=2e-3)
    assert np.median(viterbi.periodicity[70:140]) > 0.3


def test_rvc_gate_is_only_the_periodicity_threshold():
    audio = np.concatenate([np.zeros(8000, np.float32), harmonic(180, 1.0)])
    baseline = make("fcnf0++").extract_track(audio)
    rvc = make("fcnf0++-rvc").extract_track(audio)
    # The baseline never marks a frame unvoiced: no gate, no interpolation.
    assert baseline.voiced.all() and (baseline.pitch_hz > 0).all()
    np.testing.assert_array_equal(rvc.raw_pitch_hz, baseline.raw_pitch_hz)
    np.testing.assert_array_equal(rvc.voiced, rvc.periodicity > 0.035)
    np.testing.assert_array_equal(rvc.pitch_hz[rvc.voiced], baseline.pitch_hz[rvc.voiced])
    assert (rvc.pitch_hz[~rvc.voiced] == 0).all()
    # Silence is unvoiced and is not filled in with an interpolated pitch.
    assert not rvc.voiced[5:40].any()
    assert rvc.voiced[70:140].all()


def test_offline_pipeline_keeps_voicing_through_pitch_shift():
    from rvc.infer.pipeline import Pipeline
    from rvc.lib.predictors.f0_quantization import quantize_f0

    make()  # skip early when weights or penn are missing
    config = SimpleNamespace(x_pad=1, x_query=6, x_center=38, x_max=41, device=DEVICE)
    pipeline = Pipeline(48000, config)
    audio = np.concatenate([np.zeros(8000, np.float32), harmonic(180, 1.0)])
    p_len = len(audio) // 160
    coarse, f0 = pipeline.get_f0(audio, p_len, "fcnf0++-rvc", pitch=12)
    track = pipeline.fcnf0pp_predictor.extract_track(audio)
    assert len(f0) == p_len
    np.testing.assert_array_equal(f0 > 0, track.voiced)
    np.testing.assert_allclose(f0[track.voiced], track.pitch_hz[track.voiced] * 2, rtol=1e-6)
    np.testing.assert_array_equal(coarse, quantize_f0(f0, 50.0, 1680.0))


def test_realtime_pipeline_uses_the_training_quantization():
    from rvc.lib.predictors.f0_quantization import quantize_f0
    from rvc.realtime.pipeline import Realtime_Pipeline

    make()
    vc = SimpleNamespace(
        use_f0=1,
        version="v2",
        tgt_sr=48000,
        config=SimpleNamespace(device=DEVICE),
        cpt={},
        infer_audio=lambda *args: None,
    )
    pipeline = Realtime_Pipeline(vc, f0_method="fcnf0++-rvc")
    audio = torch.from_numpy(np.concatenate([np.zeros(8000, np.float32), harmonic(900, 1.0)]))
    pitch, pitchf = pipeline.get_f0(audio)
    coarse, f0 = pitch[0].cpu().numpy(), pitchf[0].cpu().numpy()
    assert len(f0) == len(audio) // 160
    np.testing.assert_array_equal(coarse, quantize_f0(f0, 50.0, 1680.0))
    # The formula the other methods use in realtime mixes Hz bounds into mel values,
    # which puts 900 Hz about 30 bins lower than training does.
    mel = 1127.0 * np.log(1.0 + f0 / 700.0)
    legacy = np.rint(np.clip((mel - 50.0) * 254 / (1680.0 - 50.0) + 1, 1, 255))
    assert coarse[f0 > 0].max() > legacy[f0 > 0].max() + 20
    assert (coarse[f0 == 0] == 1).all()
