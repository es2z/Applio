"""Run the realtime path on a recording without audio devices, and score it.

Used by tools/realtime_stream_eval.py. Everything here is device independent and works
with any VoiceChanger, so the same code measures a real model on the GPU and an
identity stand-in in the tests.
"""

import time

import numpy as np
import torch

SAMPLE_RATE = 48000
HOP = 480  # 10 ms
N_FFT = 2048
N_MELS = 128


def stream(changer, audio, conversion_args=(), conversion_kwargs=None, synchronize=True):
    """Feed `audio` (48 kHz mono float32) block by block, as the audio callback does.

    Returns (output, per-block processing times in ms). The input is padded with
    silence to a whole number of blocks and the output trimmed back to the input
    length plus the padding, so its sample k is what was played k samples after the
    first block was handed in.
    """
    conversion_kwargs = dict(conversion_kwargs or {})
    block = changer.block_frame
    total = -(-len(audio) // block) * block
    padded = np.zeros(total, dtype=np.float32)
    padded[: len(audio)] = audio
    cuda = synchronize and torch.cuda.is_available()
    outputs, timings = [], []
    with torch.no_grad():
        for start in range(0, total, block):
            if cuda:
                torch.cuda.synchronize()
            began = time.perf_counter()
            out, _, _ = changer.on_request(
                padded[start : start + block], *conversion_args, **conversion_kwargs
            )
            if cuda:
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - began) * 1000.0)
            outputs.append(np.asarray(out, dtype=np.float32))
    return np.concatenate(outputs), np.asarray(timings)


def log_mel(audio, sample_rate=SAMPLE_RATE):
    import librosa

    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=0.0,
        fmax=sample_rate / 2,
        power=1.0,
    )
    return np.log(np.maximum(mel, 1e-5))


def seam_score(output, block, crossfade, sample_rate=SAMPLE_RATE, skip_blocks=2):
    """How much the spectrum jumps where blocks are joined, against everywhere else.

    The mean frame-to-frame change of the log-mel spectrum (in dB-like log units) in the
    frames that hold a crossfade, divided by the same in all other frames. 1.0 means the
    joins cannot be told apart from the rest; a join that clicks, flutters or swaps
    timbre pushes it up. Silent frames are left out of both. The first blocks are
    skipped because they still hold the warm-up's silence.
    """
    mel = log_mel(output, sample_rate)
    flux = np.abs(np.diff(mel, axis=1)).mean(axis=0)  # one value per frame step
    energy = mel.max(axis=0)[1:]
    active = energy > np.log(1e-3)
    frames = np.arange(flux.shape[0])
    positions = (frames + 1) * HOP
    in_block = positions % block
    seam = in_block < max(crossfade, HOP)
    valid = active & (positions >= skip_blocks * block)
    seam_frames = valid & seam
    other_frames = valid & ~seam
    if not seam_frames.any() or not other_frames.any():
        return float("nan")
    return float(flux[seam_frames].mean() / flux[other_frames].mean())


def align(reference, output, expected_delay, search=4800):
    """Lag (samples) that best lines `output` up with `reference`, searched around the
    expected delay by correlating the two log-mel energy envelopes."""
    ref_env = log_mel(reference).mean(axis=0)
    out_env = log_mel(output).mean(axis=0)
    best, best_score = int(expected_delay), -np.inf
    low = int(expected_delay) - search
    high = int(expected_delay) + search
    for lag in range(low - low % HOP, high + 1, HOP):
        shift = lag // HOP
        if shift >= 0:
            a, b = ref_env[: len(ref_env) - shift], out_env[shift:]
        else:
            a, b = ref_env[-shift:], out_env[: len(out_env) + shift]
        n = min(len(a), len(b))
        if n < 10:
            continue
        a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
        score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if score > best_score:
            best, best_score = lag, score
    return best


def mel_distance(reference, output, lag):
    """Mean |log-mel difference| over the frames where the reference is not silent,
    with `output` shifted back by `lag` samples."""
    if lag > 0:
        output = output[lag:]
    elif lag < 0:
        reference = reference[-lag:]
    n = min(len(reference), len(output))
    a, b = log_mel(reference[:n]), log_mel(output[:n])
    frames = min(a.shape[1], b.shape[1])
    a, b = a[:, :frames], b[:, :frames]
    active = a.max(axis=0) > np.log(1e-3)
    if not active.any():
        return float("nan")
    return float(np.abs(a[:, active] - b[:, active]).mean())


def summarise(timings):
    steady = timings[3:] if len(timings) > 6 else timings
    return {
        "p50_ms": float(np.percentile(steady, 50)),
        "p95_ms": float(np.percentile(steady, 95)),
        "max_ms": float(np.max(steady)),
    }
