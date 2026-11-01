"""Speaker perturbation for SPIN: Praat's "Change gender" followed by a random 10-band EQ.

Ported from the official SPIN implementation (https://github.com/vectominist/spin,
src/data/audio.py and AudioPretrainDataset in src/data/dataset.py; MIT,
docs/licenses/spin-LICENSE), which takes the filters and the Praat call from ContentVec
(auspicious3000/contentvec, contentvec/data/audio/audio_utils_1.py). The ranges are
SPIN's: formant shift ratio uniform in 1-1.4, F0 median ratio in 1-2 and F0 range ratio
in 1-1.5, each inverted with probability 1/2; EQ gains uniform in +-12 dB, Q
log-uniform in 2-5, centre frequencies log-spaced from 60 Hz to 7.6 kHz.

The two departures from the original, neither of which changes what the audio becomes:
the random generator is passed in (the training loop seeds it per batch, so a run is
reproducible), and the output is cut or zero padded back to the input length, because
the swapped prediction compares the two views frame by frame.
"""

import numpy as np
import parselmouth
from scipy.signal import sosfilt

Q_MIN, Q_MAX = 2, 5
EQ_CENTRES = np.exp(np.linspace(np.log(60), np.log(7600), 10))


def make_lowshelf(g, fc, Q, fs=44100):
    g = np.power(10, (g / 20))
    A = np.max([0.0, np.sqrt(g)])
    aminus1 = A - 1
    aplus1 = A + 1
    omega = (2 * np.pi * np.max([fc, 2.0])) / fs
    coso = np.cos(omega)
    beta = np.sin(omega) * np.sqrt(A) / Q
    aminus1TimesCoso = aminus1 * coso
    b0 = A * (aplus1 - aminus1TimesCoso + beta)
    b1 = A * 2 * (aminus1 - aplus1 * coso)
    b2 = A * (aplus1 - aminus1TimesCoso - beta)
    a0 = aplus1 + aminus1TimesCoso + beta
    a1 = -2 * (aminus1 + aplus1 * coso)
    a2 = aplus1 + aminus1TimesCoso - beta
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def make_highshelf(g, fc, Q, fs=44100):
    g = np.power(10, (g / 20))
    A = np.max([0.0, np.sqrt(g)])
    aminus1 = A - 1
    aplus1 = A + 1
    omega = (2 * np.pi * np.max([fc, 2.0])) / fs
    coso = np.cos(omega)
    beta = np.sin(omega) * np.sqrt(A) / Q
    aminus1TimesCoso = aminus1 * coso
    b0 = A * (aplus1 + aminus1TimesCoso + beta)
    b1 = A * -2 * (aminus1 + aplus1 * coso)
    b2 = A * (aplus1 + aminus1TimesCoso - beta)
    a0 = aplus1 - aminus1TimesCoso + beta
    a1 = 2 * (aminus1 - aplus1 * coso)
    a2 = aplus1 - aminus1TimesCoso - beta
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def make_peaking(g, fc, Q, fs=44100):
    g = np.power(10, (g / 20))
    A = np.max([0.0, np.sqrt(g)])
    omega = (2 * np.pi * np.max([fc, 2.0])) / fs
    alpha = np.sin(omega) / (Q * 2)
    c2 = -2 * np.cos(omega)
    alphaTimesA = alpha * A
    alphaOverA = alpha / A
    b0 = 1 + alphaTimesA
    b1 = c2
    b2 = 1 - alphaTimesA
    a0 = 1 + alphaOverA
    a1 = c2
    a2 = 1 - alphaOverA
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def params2sos(G, Fc, Q, fs):
    """One low shelf, eight peaking filters and one high shelf as second order sections."""
    sections = [make_lowshelf(G[0], Fc[0], Q[0], fs=fs)]
    sections += [make_peaking(G[i], Fc[i], Q[i], fs=fs) for i in range(1, 9)]
    sections.append(make_highshelf(G[9], Fc[9], Q[9], fs=fs))
    return np.concatenate(sections, axis=0)


def change_gender(x, fs, lo, hi, ratio_fs, ratio_ps, ratio_pr):
    s = parselmouth.Sound(x, sampling_frequency=fs)
    f0 = s.to_pitch_ac(pitch_floor=lo, pitch_ceiling=hi, time_step=0.8 / lo)
    f0_np = f0.selected_array["frequency"]
    voiced = f0_np[f0_np != 0]
    if len(voiced) == 0:
        raise RuntimeError("no voiced frame for Change gender")
    f0_med = np.median(voiced).item()
    ss = parselmouth.praat.call(
        [s, f0], "Change gender", ratio_fs, f0_med * ratio_ps, ratio_pr, 1.0
    )
    return ss.values.squeeze(0)


def _either_way(rng, high):
    ratio = rng.uniform(1, high)
    return ratio if rng.random() > 0.5 else 1 / ratio


def random_formant_f0(wav, sr, lo, hi, rng):
    ratio_fs = _either_way(rng, 1.4)
    ratio_ps = _either_way(rng, 2.0)
    ratio_pr = _either_way(rng, 1.5)
    return change_gender(wav, sr, lo, hi, ratio_fs, ratio_ps, ratio_pr)


def random_eq(wav, sr, rng):
    z = rng.uniform(0, 1, size=(10,))
    Q = Q_MIN * (Q_MAX / Q_MIN) ** z
    G = rng.uniform(-12, 12, size=(10,))
    return sosfilt(params2sos(G, EQ_CENTRES, Q, sr), wav)


def perturb_speaker(wav, sr, lo, hi, rng):
    """(perturbed float32 waveform of the same length, whether Praat failed).

    Like the original, a Praat failure keeps the formants and pitch as they are and still
    applies the EQ.
    """
    failed = False
    try:
        out = random_formant_f0(wav.astype(np.float64), sr, lo, hi, rng)
    except (RuntimeError, UserWarning, ValueError):
        out = np.array(wav, dtype=np.float64)
        failed = True
    out = random_eq(out, sr, rng)
    if len(out) >= len(wav):
        out = out[: len(wav)]
    else:
        out = np.pad(out, (0, len(wav) - len(out)))
    return np.clip(out, -1.0, 1.0).astype(np.float32), failed
