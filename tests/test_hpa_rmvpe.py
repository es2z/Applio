import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from rvc.lib.predictors.f0_methods import (
    HPA_RMVPE_LAG_FRAMES,
    HPA_RMVPE_METHODS,
    HPA_RMVPE_UI_METHODS,
    hpa_rmvpe_variant,
)
from rvc.lib.predictors.hpa_rmvpe import weights

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "logs" / "reference" / "reference.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def installed(variant="76000"):
    if not weights.is_installed(variant):
        pytest.skip(f"HPA-RMVPE {variant} weights are required (tools/strip_hpa_rmvpe.py {variant})")


@pytest.fixture(scope="module")
def speech():
    import librosa

    return librosa.load(REFERENCE, sr=16000, mono=True)[0]


@pytest.fixture(scope="module")
def predictors():
    from rvc.lib.predictors.hpa_rmvpe import HPARMVPEPredictor

    for variant in weights.VARIANTS:
        installed(variant)
    return {variant: HPARMVPEPredictor(variant, DEVICE) for variant in weights.VARIANTS}


@pytest.fixture(scope="module")
def rmvpe():
    from rvc.lib.predictors.RMVPE import RMVPE0Predictor

    return RMVPE0Predictor(str(ROOT / "rvc" / "models" / "predictors" / "rmvpe.pt"), device=DEVICE)


def lag_ms(est, ref):
    """How late est reports ref, in ms, searched in 1 ms steps with ref interpolated in
    log frequency over its voiced frames."""
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    index = np.arange(n)
    log_ref = np.where(ref > 0, np.log2(np.maximum(ref, 1e-9)), np.nan)
    best = None
    for lag in np.arange(-3.0, 4.01, 0.1):
        position = index - lag
        low = np.floor(position).astype(int)
        weight = position - low
        ok = (low >= 0) & (low + 1 < n) & (est > 0)
        at = np.full(n, np.nan)
        at[ok] = (1 - weight[ok]) * log_ref[low[ok]] + weight[ok] * log_ref[low[ok] + 1]
        keep = ok & np.isfinite(at)
        error = np.median(np.abs(1200 * (np.log2(est[keep]) - at[keep])))
        if best is None or error < best[1]:
            best = (lag * 10, error)
    return best[0]


def test_methods_parse_and_every_list_offers_them():
    assert [value for _, value in HPA_RMVPE_UI_METHODS] == list(HPA_RMVPE_METHODS)
    assert [hpa_rmvpe_variant(method) for method in HPA_RMVPE_METHODS] == [
        ("76000", False), ("76000", True), ("112000", False), ("112000", True),
    ]
    with pytest.raises(ValueError):
        hpa_rmvpe_variant("rmvpe")
    # Every CLI --f0_method (infer, batch_infer, tts, extract) and every F0 radio.
    assert len(re.findall(r"\*HPA_RMVPE_METHODS,", (ROOT / "core.py").read_text(encoding="utf-8"))) == 4
    for tab, radios in (
        ("tabs/inference/inference.py", 2), ("tabs/tts/tts.py", 1), ("tabs/train/train.py", 1),
        ("tabs/realtime/realtime.py", 1), ("tabs/extra/sections/f0_extractor.py", 1),
    ):
        source = (ROOT / tab).read_text(encoding="utf-8")
        assert source.count("*HPA_RMVPE_UI_METHODS,") == radios, tab


def test_strip_refuses_anything_but_the_published_checkpoint(tmp_path):
    source = tmp_path / "model_76000.pt"
    torch.save({"model": {}}, source)
    with pytest.raises(ValueError, match="not the published"):
        weights.strip_checkpoint("76000", source, tmp_path / "out.pt")
    assert not (tmp_path / "out.pt").exists()
    with pytest.raises(ValueError, match="Unknown HPA-RMVPE variant"):
        weights.strip_checkpoint("80000", source, tmp_path / "out.pt")


def test_truncated_download_is_refused_and_cleaned_up(tmp_path):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b"PK" * 1000  # a dropped connection: far short of 203 MB

    with patch.object(weights, "PREDICTORS_DIR", tmp_path), patch("requests.get", return_value=Response()):
        with pytest.raises(IOError, match="incomplete"):
            weights.download("76000", progress=False)
    assert list(tmp_path.iterdir()) == []


def test_weight_is_stripped_and_checked_against_its_manifest(tmp_path):
    installed()
    manifest = json.loads(weights.manifest_path(weights.weight_path("76000")).read_text())
    assert manifest["source_sha256"] == weights.VARIANTS["76000"]["source_sha256"]
    assert manifest["source_revision"] == weights.SOURCE_REVISION
    assert manifest["parameters"] == weights.PARAMETER_COUNT
    assert manifest["iteration"] == 76000

    copy = tmp_path / "hpa-rmvpe-76000.pt"
    copy.write_bytes(weights.weight_path("76000").read_bytes())
    tampered = dict(manifest, weight_sha256="0" * 64)
    weights.manifest_path(copy).write_text(json.dumps(tampered))
    with patch.object(weights, "PREDICTORS_DIR", tmp_path):
        with pytest.raises(ValueError, match="differs from its manifest"):
            weights.load_state_dict("76000")


def test_network_loads_strictly_and_both_checkpoints_differ(predictors):
    first, second = (predictors[v].eager_model.state_dict() for v in ("76000", "112000"))
    assert first.keys() == second.keys()
    assert sum(t.numel() for t in first.values()) == weights.PARAMETER_COUNT
    assert any(not torch.equal(first[k], second[k]) for k in first)


# Under ~0.2 s mel2hidden's reflect padding refuses the input, for RMVPE exactly the same.
@pytest.mark.parametrize("samples", [3201, 16000, 16000 * 3 + 123, 48017])
def test_frame_count_is_rmvpes_grid(predictors, samples):
    audio = np.random.default_rng(samples).standard_normal(samples).astype(np.float32) * 0.1
    for predictor in predictors.values():
        assert len(predictor.get_f0(audio)) == samples // 160 + 1
        assert len(predictor.get_f0(audio, aligned=True)) == samples // 160 + 1


def test_speech_lag_and_its_compensation(predictors, rmvpe, speech):
    """Upstream's pitch runs ~20 ms late on speech; -aligned lands it on RMVPE's grid.

    Measured on this recording: 76000 +24 ms, 112000 +21 ms plain; +4 / -1 ms aligned,
    against RMVPE's own -2 ms from CREPE.
    """
    reference = rmvpe.infer_from_audio(speech, thred=0.03)
    for variant, predictor in predictors.items():
        plain = lag_ms(predictor.get_f0(speech), reference)
        aligned_f0 = predictor.get_f0(speech, aligned=True)
        aligned = lag_ms(aligned_f0, reference)
        assert 15 <= plain <= 28, (variant, plain)
        assert abs(aligned) <= 6, (variant, aligned)
        voiced = (aligned_f0 > 0) & (reference > 0)
        cents = np.abs(1200 * np.log2(aligned_f0[voiced] / reference[voiced]))
        # A wrong architecture or weight lands far from RMVPE; this one is ~10 cents.
        assert np.median(cents) < 20, (variant, np.median(cents))
    assert HPA_RMVPE_LAG_FRAMES == 2


def test_extraction_spec_pins_the_weight_and_never_trusts_unrecorded_files():
    from rvc.train.extract.fcn_metadata import can_reuse, extraction_spec

    installed()
    specification = extraction_spec("hpa-rmvpe-76000-aligned")
    assert specification == {
        "method": "hpa-rmvpe-76000-aligned",
        "weight_sha256": weights.weight_sha256("76000"),
    }
    assert not can_reuse(None, specification, "signature")
    assert can_reuse(None, extraction_spec("rmvpe"), "signature")


def test_offline_predictor_is_cached_until_the_compile_setting_changes():
    from rvc.lib.predictors.hpa_rmvpe import predictor as module

    built = []

    class Fake:
        def __init__(self, variant, device, compile_profile=None):
            enabled, mode = module.compile_settings()
            self.compile_enabled, self.compile_mode = enabled, mode if enabled else None
            built.append((variant, device, compile_profile))

    setting = [(False, "default")]
    with patch.object(module, "HPARMVPEPredictor", Fake), patch.object(
        module, "compile_settings", lambda: setting[0]
    ), patch.dict(module._offline_cache, clear=True):
        first = module.get_offline_predictor("76000", "cuda:0")
        assert module.get_offline_predictor("76000", "cuda:0") is first
        setting[0] = (False, "max-autotune")  # mode alone is irrelevant while off
        assert module.get_offline_predictor("76000", "cuda:0") is first
        setting[0] = (True, "default")
        second = module.get_offline_predictor("76000", "cuda:0")
        assert second is not first
        module.get_offline_predictor("112000", "cuda:0")
    assert built == [("76000", "cuda:0", "offline")] * 2 + [("112000", "cuda:0", "offline")]


def test_compile_session_reports_and_closes_an_f0_path():
    from rvc.realtime.compile_session import CompileSession, CompileSettings

    f0 = SimpleNamespace(compiled=object(), status=lambda: "HPA-RMVPE: compiled")
    session = CompileSession(CompileSettings(), lambda x: x, lambda x: x, "cpu", f0=f0)
    assert session.enabled
    assert session.status() == "\nHPA-RMVPE: compiled"
    session.close()
    assert f0.compiled is None and not session.enabled
    plain = CompileSession(CompileSettings(), lambda x: x, lambda x: x, "cpu")
    assert not plain.enabled and plain.status() == ""


def _compiled(profile, mode):
    from torch.utils._triton import has_triton
    from rvc.lib.predictors.hpa_rmvpe import predictor as module

    if not torch.cuda.is_available() or not has_triton():
        pytest.skip("CUDA and Triton are required to compile")
    installed()
    with patch.object(module, "compile_settings", lambda: (True, mode)):
        predictor = module.HPARMVPEPredictor("76000", "cuda", compile_profile=profile)
    assert predictor.compile_path is not None and predictor.compile_path.compiled is not None
    return predictor, module.HPARMVPEPredictor("76000", "cuda")


def _agreement(compiled, eager, audio):
    a, b = compiled.get_f0(audio), eager.get_f0(audio)
    voiced = (a > 0) & (b > 0)
    cents = np.abs(1200 * np.log2(a[voiced] / b[voiced]))
    return np.percentile(cents, 99), np.mean((a > 0) != (b > 0))


def test_offline_compile_matches_eager_at_several_lengths(speech):
    compiled, eager = _compiled("offline", "default")
    for samples in (16000 * 3 + 123, 16000 * 7 + 4567):
        p99, flips = _agreement(compiled, eager, speech[:samples])
        assert p99 < 1.0 and flips < 0.005, (samples, p99, flips)
    assert compiled.compile_path.state == "Using compiled inference"


def test_realtime_compile_matches_eager_on_fixed_windows(speech):
    compiled, eager = _compiled("realtime", "reduce-overhead")
    starts = np.random.default_rng(0).integers(0, len(speech) - 24000, 20)
    for start in starts:
        p99, flips = _agreement(compiled, eager, speech[start : start + 24000])
        assert p99 < 1.0 and flips < 0.01, (start, p99, flips)
    assert compiled.compile_path.state == "Using compiled inference"
