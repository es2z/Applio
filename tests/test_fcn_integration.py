import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rvc.lib.predictors.f0_quantization import quantize_f0
from rvc.lib.predictors.fcn.profiles import FCNProfile, resolve_profile
from rvc.train.extract.fcn_metadata import (
    can_reuse,
    input_signature,
    validate_pitch_files,
    write_metadata,
)


def test_profile_precedence_and_blank_ui_value(tmp_path):
    checkpoint = FCNProfile(coarse_max=1100).to_dict()
    assert resolve_profile("fcn-993", "", checkpoint).coarse_max == 1100
    assert resolve_profile("fcn-993", FCNProfile(), checkpoint).coarse_max == 1680
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(checkpoint))
    assert resolve_profile("fcn-993", path) == resolve_profile(
        "fcn-993", json.dumps(checkpoint)
    )


def test_metadata_failure_and_method_changes_never_reuse(tmp_path):
    specification = {"method": "fcn-993", "fingerprint": "a"}
    previous = {
        "specification": specification,
        "input_signature": "x",
        "complete": True,
    }
    assert can_reuse(previous, specification, "x")
    assert not can_reuse(previous, specification, "y")
    assert not can_reuse(previous, {"method": "rmvpe"}, "x")
    assert not can_reuse(None, specification, "x")
    previous["complete"] = False
    assert not can_reuse(previous, specification, "x")
    path = tmp_path / "model_info.json"
    write_metadata(path, {"pitch_extraction_run": previous})
    assert json.loads(path.read_text())["pitch_extraction_run"]["complete"] is False


def test_source_changed_in_place_invalidates_signature(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"first")
    files = [[str(path), "", "", ""]]
    signature = input_signature(files)
    path.write_bytes(b"other")
    assert signature != input_signature(files)


def test_grid_pair_validation(tmp_path):
    import soundfile as sf

    source, coarse, hz = (tmp_path / name for name in ("a.wav", "coarse.npy", "hz.npy"))
    sf.write(source, np.zeros(1600), 16000)
    np.save(hz, np.full(10, 220, np.float32))
    np.save(coarse, quantize_f0(np.full(10, 220, np.float32)))
    files = [[source, coarse, hz, ""]]
    validate_pitch_files(files)
    np.save(hz, np.full(11, 220))
    with pytest.raises(ValueError):
        validate_pitch_files(files)


def test_inference_uv_survives_autotune_and_shift(monkeypatch):
    from rvc.infer.pipeline import Autotune, Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.fcn_predictor = SimpleNamespace(
        profile=FCNProfile(
            method="fcn-993-rvc", enter_threshold=0.5, exit_threshold=0.4
        ),
        get_f0=lambda x, n: np.array([0, 220, 900], np.float32),
    )
    pipeline.autotune = Autotune()
    coarse, hz = pipeline.get_f0(np.zeros(480), 3, "fcn-993-rvc", pitch=12)
    np.testing.assert_array_equal(hz, [0, 440, 1800])
    assert coarse[-1] == 255
    coarse, hz = pipeline.get_f0(
        np.zeros(480), 3, "fcn-993-rvc", f0_autotune=True, f0_autotune_strength=0.5
    )
    assert hz[0] == 0 and coarse[0] == 1


def test_extract_subprocess_failure_is_propagated(monkeypatch):
    import subprocess

    import core

    def fail(*args, **kwargs):
        assert kwargs["check"]
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(core.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        core.run_extract_script("test", "fcn-993", 1, "0", 40000, "contentvec")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_training_inference_match_and_cache_shares_weights():
    from rvc.infer.pipeline import Pipeline
    from rvc.lib.predictors.fcn.adapter import DEFAULT_WEIGHT
    from rvc.train.extract.extract import FeatureInput

    if not DEFAULT_WEIGHT.exists():
        pytest.skip("Convert weights first")
    audio = (0.1 * np.sin(np.arange(4800) * 2 * np.pi * 220 / 16000)).astype(np.float32)
    feature = FeatureInput("fcn-993", "cuda")
    config = SimpleNamespace(x_pad=1, x_query=1, x_center=2, x_max=3, device="cuda")
    pipeline = Pipeline(40000, config)
    first = pipeline.configure_fcn("fcn-993")
    coarse, hz = pipeline.get_f0(audio, 30, "fcn-993")
    expected = feature.compute_f0(audio)
    np.testing.assert_allclose(hz, expected, atol=0, rtol=0)
    np.testing.assert_array_equal(coarse, feature.coarse_f0(expected))
    second = pipeline.configure_fcn(
        "fcn-993-rvc",
        FCNProfile(method="fcn-993-rvc", enter_threshold=0.6, exit_threshold=0.4),
    )
    assert first.model is second.model


def test_export_checkpoint_preserves_pitch_metadata(tmp_path):
    from rvc.train.process.extract_model import extract_model

    config = json.loads(
        (Path(__file__).resolve().parents[1] / "rvc/configs/48000.json").read_text()
    )
    hps = SimpleNamespace(
        data=SimpleNamespace(**config["data"]), model=SimpleNamespace(**config["model"])
    )
    metadata = {
        "method": "fcn-993",
        "profile": FCNProfile().to_dict(),
        "coarse": {"minimum": 50, "maximum": 1680},
    }
    (tmp_path / "model_info.json").write_text(
        json.dumps(
            {"pitch_extraction_run": {"complete": True}, "f0_extraction": metadata}
        )
    )
    destination = tmp_path / "model.pth"
    extract_model(
        {"test_weight": torch.ones(2)},
        48000,
        "test",
        str(destination),
        1,
        1,
        hps,
        None,
        "HiFi-GAN",
    )
    saved = torch.load(destination, weights_only=True)
    assert saved["f0_extraction"] == metadata


def test_utility_csv_uses_seconds_and_hz(tmp_path, monkeypatch):
    from tabs.extra.sections import f0_extractor as utility

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utility.librosa, "get_samplerate", lambda path: 44100)
    track = {
        "timestamps": np.array([0, 0.01]),
        "pitch_hz": np.array([220, 0]),
        "confidence": np.array([0.9, 0.1]),
        "voiced": np.array([True, False]),
    }
    monkeypatch.setattr(
        utility,
        "F0Extractor",
        lambda *args, **kwargs: SimpleNamespace(extract_track=lambda: track),
    )
    utility.plt.switch_backend("Agg")
    _image, csv = utility.extract_f0_curve("unused.wav", "fcn-993")
    rows = Path(csv).read_text().splitlines()
    assert rows[0] == "seconds,pitch_hz,confidence,voiced"
    assert rows[1].startswith("0.0,220,")
    assert rows[2].startswith("0.01,0,")


def test_training_checks_completion_and_uses_current_pretrain_api(
    tmp_path, monkeypatch
):
    import core
    from rvc.lib.tools import pretrained_selector
    from rvc.train.extract import preparing_files

    monkeypatch.setattr(core, "logs_path", str(tmp_path))
    monkeypatch.setattr(
        preparing_files, "apply_train_settings", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        preparing_files,
        "apply_generator_lr_boost_settings",
        lambda *args, **kwargs: None,
    )
    calls = []

    def select(vocoder, sample_rate):
        calls.append((vocoder, sample_rate))
        return "G.pth", "D.pth"

    monkeypatch.setattr(pretrained_selector, "pretrained_selector", select)
    monkeypatch.setattr(
        core.subprocess, "run", lambda command: SimpleNamespace(returncode=1)
    )
    args = ("test", 1, True, True, 1, 48000, 1, "0", False, 0, True, False)
    assert "Training failed" in core.run_train_script(*args)
    assert calls == [("HiFi-GAN", 48000)]
    directory = tmp_path / "test"
    directory.mkdir()
    (directory / "model_info.json").write_text(
        json.dumps({"pitch_extraction_run": {"complete": False}})
    )
    with pytest.raises(ValueError, match="incomplete"):
        core.run_train_script(*args)
