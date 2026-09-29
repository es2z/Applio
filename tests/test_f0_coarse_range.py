"""The per-model coarse F0 range: chosen at extraction, read back everywhere else."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rvc.lib.predictors.f0_quantization import (
    COARSE_MAXIMA,
    DEFAULT_COARSE_MAX,
    align_profile_coarse,
    coarse_bin_centers,
    default_coarse_max,
    extraction_coarse_maxima,
    quantize_f0,
    recorded_coarse_max,
    remap_pitch_embedding,
    validate_extraction_coarse_max,
)


def legacy_quantize(f0, maximum):
    """The formula training extraction and offline conversion used before."""
    mel_min, mel_max = 1127 * np.log(1 + np.array([50.0, maximum]) / 700)
    mel = 1127.0 * np.log(1.0 + f0 / 700.0)
    return np.rint(np.clip((mel - mel_min) * 254 / (mel_max - mel_min) + 1, 1, 255)).astype(int)


@pytest.mark.parametrize("maximum", COARSE_MAXIMA)
def test_one_quantizer_matches_the_training_formula(maximum):
    f0 = np.concatenate(([0.0], np.geomspace(30, 2500, 5000)))
    np.testing.assert_array_equal(quantize_f0(f0, 50.0, maximum), legacy_quantize(f0, maximum))


def test_offline_inference_used_the_wrong_range():
    """Why this exists: 200 Hz was trained on bin 41 and converted on bin 54."""
    assert quantize_f0(200.0, 50, 1680) == 41
    assert quantize_f0(200.0, 50, 1100) == 54


def test_recorded_range_precedence():
    assert recorded_coarse_max({}) == DEFAULT_COARSE_MAX == 1680.0
    assert recorded_coarse_max(None) == 1680.0
    fcn = {"f0_extraction": {"coarse": {"maximum": 1100}}}
    assert recorded_coarse_max(fcn) == 1100.0
    assert recorded_coarse_max({**fcn, "f0_coarse_max": 1000.0}) == 1000.0
    with pytest.raises(ValueError):
        recorded_coarse_max({"f0_coarse_max": 900})


def test_750_and_1000_are_fcn_only_choices():
    assert default_coarse_max("fcn-929-rvc") == 1000.0
    assert default_coarse_max("rmvpe") == default_coarse_max("fcnf0++") == 1680.0
    assert extraction_coarse_maxima("fcn-993") == (750.0, 1000.0, 1100.0, 1680.0)
    assert extraction_coarse_maxima("rmvpe") == extraction_coarse_maxima("fcnf0++") == (1100.0, 1680.0)
    assert validate_extraction_coarse_max("750", "fcn-993-rvc") == 750.0
    for value in (750, 1000):
        with pytest.raises(ValueError, match="only offered for FCN"):
            validate_extraction_coarse_max(value, "rmvpe")
    # A model extracted at 750 Hz is converted at 750 Hz whatever the method.
    assert recorded_coarse_max({"f0_coarse_max": 750}) == 750.0


def test_profiles_follow_the_model():
    from rvc.lib.predictors.fcn.profiles import resolve_profile

    profile = resolve_profile("fcn-929-rvc")
    assert align_profile_coarse(profile, 1680) is profile
    aligned = align_profile_coarse(profile, 750)
    assert aligned.coarse_max == 750.0 and aligned.enter_threshold == profile.enter_threshold


def test_spec_records_the_range_only_when_it_is_not_the_default():
    from rvc.train.extract.fcn_metadata import can_reuse, extraction_spec

    assert extraction_spec("rmvpe") == {"method": "rmvpe"}
    assert extraction_spec("rmvpe", None, 1100) == {"method": "rmvpe", "coarse_max": 1100.0}
    # Unrecorded pitch files were quantized at 1680 Hz: reusable there, nowhere else.
    assert can_reuse(None, extraction_spec("rmvpe"), "s")
    assert not can_reuse(None, extraction_spec("rmvpe", None, 1100), "s")
    record = {"complete": True, "specification": {"method": "rmvpe"}, "input_signature": "s"}
    assert not can_reuse(record, extraction_spec("rmvpe", None, 1100), "s")


def test_pipeline_quantizes_over_the_models_range():
    from rvc.infer.pipeline import Autotune, Pipeline

    config = SimpleNamespace(x_pad=1, x_query=1, x_center=2, x_max=3, device="cpu")
    assert Pipeline(40000, config).f0_max == 1680.0
    pipeline = Pipeline(40000, config, coarse_max=750)
    assert pipeline.f0_max == 750.0
    from rvc.lib.predictors.fcn.profiles import FCNProfile

    pipeline.fcn_predictor = SimpleNamespace(
        profile=FCNProfile(method="fcn-993"),
        get_f0=lambda x, n: np.array([0, 200, 700, 900], np.float32),
    )
    pipeline.autotune = Autotune()
    coarse, hz = pipeline.get_f0(np.zeros(640), 4, "fcn-993")
    np.testing.assert_array_equal(coarse, quantize_f0(hz, 50, 750))
    assert coarse[-1] == 255


def test_remap_keeps_every_pitch_on_its_trained_row():
    weight = torch.randn(256, 8)
    assert torch.equal(remap_pitch_embedding(weight, 1680, 1680), weight)
    remapped = remap_pitch_embedding(weight, 1680, 1000)
    centers = coarse_bin_centers(1000)
    for b in (1, 40, 128, 255):
        assert torch.equal(remapped[b], weight[int(quantize_f0(centers[b], 50, 1680))])
    # Converted pitches land where they were trained: within one source bin.
    for hz in (110.0, 220.0, 440.0, 880.0):
        now = remapped[int(quantize_f0(hz, 50, 1000))]
        neighbours = [weight[int(quantize_f0(hz, 50, 1680)) + d] for d in (-1, 0, 1)]
        assert any(torch.equal(now, n) for n in neighbours), hz


def test_resume_guard_treats_an_unstamped_checkpoint_as_1680():
    from rvc.train.utils import describe_architecture_mismatch

    current = {"vocoder": None, "f0_coarse_max": 1680.0}
    assert describe_architecture_mismatch({}, current) is None
    reason = describe_architecture_mismatch({}, {**current, "f0_coarse_max": 1000.0})
    assert "F0 coarse range 1680 Hz -> 1000 Hz" in reason
    assert describe_architecture_mismatch({"f0_coarse_max": 1000.0}, {**current, "f0_coarse_max": 1000.0}) is None


def test_reset_reindexes_the_pitch_embedding():
    from rvc.train.reset_run import _retarget_coarse

    weight = torch.randn(256, 4)
    checkpoint = {"model": {"enc_p.emb_pitch.weight": weight.clone()}}
    assert _retarget_coarse(checkpoint, 1000.0) is not None
    assert checkpoint["f0_coarse_max"] == 1000.0
    assert torch.equal(
        checkpoint["model"]["enc_p.emb_pitch.weight"], remap_pitch_embedding(weight, 1680, 1000)
    )
    before = checkpoint["model"]["enc_p.emb_pitch.weight"].clone()
    assert _retarget_coarse(checkpoint, 1000.0) is None
    assert torch.equal(checkpoint["model"]["enc_p.emb_pitch.weight"], before)


def test_warm_start_reindexes_only_a_stamped_pretrain():
    from rvc.train.warm_start import PITCH_EMBEDDING_KEY, Transfer, _plan_pitch_embedding

    weight = torch.randn(256, 4)

    def plan(checkpoint):
        transfer = Transfer("G", {PITCH_EMBEDDING_KEY: weight}, {PITCH_EMBEDDING_KEY: weight})
        transfer.inherit(PITCH_EMBEDDING_KEY, PITCH_EMBEDDING_KEY)
        _plan_pitch_embedding(transfer, checkpoint, {"f0_coarse_max": 1000.0})
        return transfer

    stamped = plan({"f0_coarse_max": 1680.0})
    assert torch.equal(stamped.loaded[PITCH_EMBEDDING_KEY], remap_pitch_embedding(weight, 1680, 1000))
    assert stamped.notes
    stock = plan({})
    assert torch.equal(stock.loaded[PITCH_EMBEDDING_KEY], weight) and not stock.notes


def test_blender_refuses_different_ranges(tmp_path):
    from rvc.train.process.model_blender import model_blender

    for name, coarse in (("a", 1680.0), ("b", 1000.0)):
        torch.save({"sr": "48k", "config": [], "f0": 1, "version": "v2",
                    "weight": {}, "f0_coarse_max": coarse}, tmp_path / f"{name}.pth")
    message = model_blender("x", str(tmp_path / "a.pth"), str(tmp_path / "b.pth"), 0.5)
    assert "coarse" in message
