import numpy as np
import pytest

from tools.calibrate_fcn993 import calibrate, counts
from tools.fcn_viterbi_reference import parameters, viterbi


def test_viterbi_matches_categorical_oracle():
    hmm = pytest.importorskip("hmmlearn.hmm")
    activation = np.random.default_rng(993).random((25, 486)).astype(np.float32)
    prior, transition, emission = parameters()
    model = hmm.CategoricalHMM(n_components=486, init_params="")
    model.startprob_, model.transmat_, model.emissionprob_ = prior, transition, emission
    expected_score, _ = model.decode(activation.argmax(1)[:, None])
    actual, cents = viterbi(activation)
    # Uniform emissions have many equally optimal paths; compare the objective,
    # not implementation-specific floating-point tie breaking.
    score = (
        np.log(prior[actual[0]])
        + np.log(emission[actual, activation.argmax(1)]).sum()
        + np.log(transition[actual[:-1], actual[1:]]).sum()
    )
    assert score == pytest.approx(expected_score, abs=1e-10)
    assert np.isfinite(cents).all()


def test_viterbi_constant_observation_has_known_unique_path():
    activation = np.zeros((20, 486), np.float32)
    activation[:, 200] = 1
    path, cents = viterbi(activation)
    np.testing.assert_array_equal(path, np.full(20, 200))
    assert np.allclose(
        cents, np.linspace(1200 * np.log2(3), 1200 * np.log2(100), 486)[200]
    )


def test_calibration_hysteresis_resets_between_files():
    confidence = np.array([0.8, 0.5, 0.5, 0.8])
    support = np.ones(4)
    reference = np.array([True, True, False, True])
    resets = np.array([True, False, True, False])
    assert counts(confidence, support, reference, reference, resets, 0.7, 0.4) == (
        0,
        0,
        0,
    )


def test_calibration_rejects_overlapping_speakers(tmp_path):
    paths = []
    for split in ("calibration", "validation", "test"):
        path = tmp_path / (split + ".npz")
        np.savez(
            path,
            speaker="same",
            split=split,
            audio=np.ones(320),
            cents=np.ones(20),
            confidence=np.full(20, 0.8),
            voiced=np.array([True, False]),
            mangio_voiced=np.array([True, False]),
        )
        paths.append(path)
    with pytest.raises(ValueError, match="overlaps"):
        calibrate(paths, tmp_path / "profile.json")


def test_candidate_is_not_marked_listening_validated(tmp_path):
    paths = []
    for split in ("calibration", "validation", "test"):
        path = tmp_path / (split + ".npz")
        np.savez(
            path,
            speaker=split,
            split=split,
            audio=np.ones(320),
            cents=np.ones(20),
            confidence=np.r_[np.full(5, 0.9), np.full(15, 0.1)],
            voiced=np.array([True, False]),
            mangio_voiced=np.array([True, False]),
        )
        paths.append(path)
    output = tmp_path / "profile.json"
    calibrate(paths, output)
    import json

    assert json.loads(output.read_text())["calibrated"] is False
