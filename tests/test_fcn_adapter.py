import numpy as np
import pytest

from rvc.lib.predictors.fcn.adapter import FCNRVCAdapter, weighted_median
from rvc.lib.predictors.fcn.profiles import FCNProfile, resolve_profile


def test_no_uncalibrated_default_threshold():
    with pytest.raises(ValueError, match="explicit"):
        resolve_profile("fcn-993-rvc")
    with pytest.raises(ValueError):
        FCNProfile(enter_threshold=0.1)
    with pytest.raises(ValueError):
        FCNProfile(method="fcn-993-rvc", enter_threshold=0.2, exit_threshold=0.4)


def test_fingerprint_includes_weight_and_quantization():
    default = FCNProfile()
    assert default.fingerprint("a") != default.fingerprint("b")
    assert default.fingerprint("a") != FCNProfile(coarse_max=1100).fingerprint("a")


def test_deterministic_weighted_median_tie():
    assert weighted_median(np.array([300, 100]), np.ones(2)) == 100


def test_hysteresis_does_not_fill_uv_or_hold_pitch():
    # Artificial test thresholds, not a shipping/calibrated profile.
    profile = FCNProfile(method="fcn-993-rvc", enter_threshold=0.7, exit_threshold=0.4)
    confidence = np.repeat([0.8, 0.5, 0.1, 0.5, 0.8], 10).astype(np.float32)
    cents = np.full(50, 1200 * np.log2(22))
    track = FCNRVCAdapter(profile)(cents, confidence, 5, np.ones(800))
    assert track.voiced.tolist() == [True, True, False, False, False]
    assert np.all(track.pitch_hz[~track.voiced] == 0)
    silence = FCNRVCAdapter(profile)(cents, confidence, 5, np.zeros(800))
    assert not silence.voiced.any()
