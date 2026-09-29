"""rmvpe-filled: RMVPE with its unvoiced frames filled from the voiced ones."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rvc.lib.predictors.f0_gap_fill import fill_unvoiced_gaps
from rvc.lib.predictors.f0_methods import GAP_FILLED_METHODS, gap_filled_base

RMVPE_WEIGHT = Path("rvc/models/predictors/rmvpe.pt")
needs_rmvpe = pytest.mark.skipif(
    not RMVPE_WEIGHT.is_file() or not torch.cuda.is_available(),
    reason="RMVPE weights and CUDA required",
)


def test_gaps_are_filled_log_linearly_and_edges_held():
    f0 = np.array([0, 0, 200, 0, 0, 800, 0], np.float32)
    filled = fill_unvoiced_gaps(f0)
    assert filled.dtype == np.float32 and np.all(filled > 0)
    np.testing.assert_allclose(filled[:3], 200, rtol=1e-6)  # held before the first voiced frame
    np.testing.assert_allclose(filled[5:], 800, rtol=1e-6)  # held after the last one
    # 200 -> 800 is two octaves over three frames: one octave per 1.5 frames, in log.
    np.testing.assert_allclose(filled[3:5], 200 * 2 ** (np.array([1, 2]) * 2 / 3), rtol=1e-5)


def test_voiced_frames_are_untouched():
    f0 = np.array([110.0, 0.0, 130.0, 140.0, 0.0, 0.0, 150.0])
    filled = fill_unvoiced_gaps(f0)
    np.testing.assert_array_equal(filled[f0 > 0], f0[f0 > 0])


@pytest.mark.parametrize("f0", [np.zeros(5), np.array([0, 0, 220.0, 0]), np.zeros(0)])
def test_fewer_than_two_voiced_frames_is_left_alone(f0):
    np.testing.assert_array_equal(fill_unvoiced_gaps(f0), f0)


def test_registration():
    assert GAP_FILLED_METHODS == {
        "rmvpe-filled": "rmvpe",
        "hpa-rmvpe-76000-aligned-filled": "hpa-rmvpe-76000-aligned",
        "hpa-rmvpe-112000-aligned-filled": "hpa-rmvpe-112000-aligned",
    }
    assert gap_filled_base("rmvpe-filled") == "rmvpe"
    assert gap_filled_base("hpa-rmvpe-112000-aligned-filled") == "hpa-rmvpe-112000-aligned"
    assert gap_filled_base("rmvpe") == "rmvpe"
    assert gap_filled_base("fcpe") == "fcpe"


def test_extraction_never_reuses_unrecorded_or_rmvpe_pitch():
    from rvc.train.extract.fcn_metadata import can_reuse, extraction_spec

    spec = extraction_spec("rmvpe-filled")
    assert spec == {"method": "rmvpe-filled"}
    # Pitch files with no record were extracted by something else.
    assert not can_reuse(None, spec, "s")
    rmvpe_record = {"complete": True, "specification": extraction_spec("rmvpe"), "input_signature": "s"}
    assert not can_reuse(rmvpe_record, spec, "s")
    assert can_reuse({"complete": True, "specification": spec, "input_signature": "s"}, spec, "s")


@needs_rmvpe
def test_training_and_conversion_fill_the_same_rmvpe_pitch():
    from rvc.infer.pipeline import Pipeline
    from rvc.train.extract.extract import FeatureInput

    t = np.arange(16000 * 2) / 16000
    audio = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    audio[8000:12000] = np.random.default_rng(0).normal(0, 0.05, 4000).astype(np.float32)  # a "consonant"
    plain = FeatureInput("rmvpe", "cuda").compute_f0(audio)
    filled = FeatureInput("rmvpe-filled", "cuda").compute_f0(audio)
    assert (plain <= 0).any(), "the noise burst should read unvoiced to RMVPE"
    # Two separate RMVPE runs on CUDA agree to ~1e-7, not bit for bit.
    np.testing.assert_allclose(filled, fill_unvoiced_gaps(plain), rtol=1e-5)
    assert np.all(filled > 0)

    pipeline = Pipeline(40000, SimpleNamespace(x_pad=1, x_query=1, x_center=2, x_max=3, device="cuda"))
    n = len(audio) // 160
    _, rmvpe_hz = pipeline.get_f0(audio, n, "rmvpe")
    coarse, filled_hz = pipeline.get_f0(audio, n, "rmvpe-filled")
    np.testing.assert_allclose(filled_hz, fill_unvoiced_gaps(np.asarray(rmvpe_hz)), rtol=1e-5)
    assert np.all(coarse > 1)


HPA_WEIGHT = Path("rvc/models/predictors/hpa-rmvpe-112000.pt")


def test_hpa_filled_spec_carries_the_base_weight():
    if not HPA_WEIGHT.is_file():
        pytest.skip("HPA-RMVPE 112000 weights required")
    from rvc.train.extract.fcn_metadata import can_reuse, extraction_spec

    base = extraction_spec("hpa-rmvpe-112000-aligned")
    spec = extraction_spec("hpa-rmvpe-112000-aligned-filled")
    assert spec["method"] == "hpa-rmvpe-112000-aligned-filled"
    assert spec["weight_sha256"] == base["weight_sha256"]
    assert not can_reuse(None, spec, "s")
    assert not can_reuse({"complete": True, "specification": base, "input_signature": "s"}, spec, "s")


@pytest.mark.skipif(not HPA_WEIGHT.is_file() or not torch.cuda.is_available(), reason="HPA-RMVPE weights and CUDA required")
def test_hpa_filled_is_the_aligned_pitch_filled():
    from rvc.infer.pipeline import Pipeline
    from rvc.train.extract.extract import FeatureInput

    t = np.arange(16000 * 2) / 16000
    audio = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    audio[8000:12000] = np.random.default_rng(0).normal(0, 0.05, 4000).astype(np.float32)
    plain = FeatureInput("hpa-rmvpe-112000-aligned", "cuda").compute_f0(audio)
    filled = FeatureInput("hpa-rmvpe-112000-aligned-filled", "cuda").compute_f0(audio)
    assert (plain <= 0).any()
    np.testing.assert_allclose(filled, fill_unvoiced_gaps(plain), rtol=1e-5)
    assert np.all(filled > 0)

    pipeline = Pipeline(40000, SimpleNamespace(x_pad=1, x_query=1, x_center=2, x_max=3, device="cuda"))
    n = len(audio) // 160
    _, base_hz = pipeline.get_f0(audio, n, "hpa-rmvpe-112000-aligned")
    _, filled_hz = pipeline.get_f0(audio, n, "hpa-rmvpe-112000-aligned-filled")
    np.testing.assert_allclose(filled_hz, fill_unvoiced_gaps(np.asarray(base_hz)), rtol=1e-5)
