"""Trim the leading and trailing silence of every WAV in a folder, keeping a little.

    env\\python.exe tools\\trim_edge_silence.py <input dir> <output dir> [--keep 0.1]

Corpus recordings often carry a second or more of silence before and after each
utterance. Preprocess "Automatic" cuts most of it, but leaves up to 0.5 s at each cut, so
measured on the ITA corpus readings this leaves 21% silent frames and 3.0% of 0.36 s
training crops almost entirely silent; trimming the edges to 0.1 s first takes that to
15% and 1.1%. Pauses inside an utterance are left alone - the slicer splits the long ones
itself - and nothing else about the audio changes.

Done with soundfile rather than ffmpeg on purpose: it is only a cut, so the kept samples
are copied bit for bit at the file's own sample rate and bit depth, and every output is
verified against its source. The threshold follows each file's level - 40 dB below its
loudest 10 ms frame, and never above -50 dBFS - so a quiet speaker's soft onsets and
endings are not mistaken for silence. The folder structure (speaker folders 0/, 1/, ...)
is mirrored.

Levels are measured on a 100 Hz high-passed copy, and only measured there. The ITA
recordings carry -70 to -60 dBFS of rumble before and after speech, all of it under
300 Hz (spectral centroid 20-130 Hz); measured on the full band, the relative threshold
took that for sound and kept a median 0.86 s of lead-in where speech starts at 1.35 s.
The output itself is never filtered.

--stereo louder writes a stereo file as its louder channel alone, for binaural sources.
Preprocess folds stereo to mono by averaging the channels, and a binaural recording's ears
hear the same voice up to ~0.8 ms apart: averaging them is a comb filter, notches at
roughly 670 Hz, 2 kHz, 3.3 kHz and up. Measured on such a folder, 34% of the files got
notches deeper than -10 dB (p10 -22.8 dB), and a file whose far ear is 15 dB down loses
6 dB of level to the average. The nearer ear has neither problem, and in 94-98% of those
files the same ear stays nearer throughout. The default, keep, leaves channels alone.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from scipy import signal

DETECTION_HIGH_PASS_HZ = 100
FRAME_SECONDS = 0.01
RELATIVE_THRESHOLD_DB = 40.0
CEILING_THRESHOLD_DB = -50.0
FADE_SECONDS = 0.005
# soundfile subtypes whose samples int32 holds exactly; anything else is read as float64.
INTEGER_SUBTYPES = {"PCM_16", "PCM_24", "PCM_32"}


def frame_db(mono, sample_rate):
    hop = max(1, int(sample_rate * FRAME_SECONDS))
    count = len(mono) // hop
    if count == 0:
        return np.array([-200.0]), hop
    frames = mono[: count * hop].reshape(count, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20 * np.log10(rms + 1e-12), hop


def detection_signal(audio, sample_rate):
    """Every channel as float in [-1, 1], 100 Hz high-passed: what levels are read from."""
    channels = audio.astype(np.float64)
    if np.issubdtype(audio.dtype, np.integer):
        channels /= 2**31
    high_pass = signal.butter(
        4, DETECTION_HIGH_PASS_HZ, btype="high", fs=sample_rate, output="sos"
    )
    return signal.sosfiltfilt(high_pass, channels, axis=0)


def louder_channel(audio, sample_rate):
    """Index of the channel with more energy over the file's speech."""
    channels = detection_signal(audio, sample_rate)
    hop = max(1, int(sample_rate * FRAME_SECONDS))
    count = len(channels) // hop
    if count == 0:
        return 0
    energy = np.mean(channels[: count * hop].reshape(count, hop, -1) ** 2, axis=1)
    level = energy.max(axis=1)
    speech = level >= level.max() * 10 ** (-RELATIVE_THRESHOLD_DB / 10)
    return int(np.argmax(energy[speech].sum(axis=0)))


def trim_bounds(audio, sample_rate, keep_seconds):
    """(start, end, threshold) in samples, or None when nothing is above the threshold."""
    mono = detection_signal(audio, sample_rate).mean(axis=1)
    db, hop = frame_db(mono, sample_rate)
    threshold = min(CEILING_THRESHOLD_DB, db.max() - RELATIVE_THRESHOLD_DB)
    loud = np.nonzero(db >= threshold)[0]
    if len(loud) == 0:
        return None
    keep = int(keep_seconds * sample_rate)
    start = max(0, loud[0] * hop - keep)
    end = min(len(audio), (loud[-1] + 1) * hop + keep)
    return start, end, threshold


def fade_edges(audio, sample_rate, at_start, at_end):
    """A 5 ms fade where a cut landed, so a cut into low noise cannot click."""
    length = min(int(FADE_SECONDS * sample_rate), len(audio) // 2)
    if length == 0:
        return audio
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, length)))[:, None]
    out = audio.astype(np.float64)
    if at_start:
        out[:length] *= ramp
    if at_end:
        out[-length:] *= ramp[::-1]
    if np.issubdtype(audio.dtype, np.integer):
        return np.round(out).astype(audio.dtype)
    return out.astype(audio.dtype)


def process(job):
    source, target, keep, stereo = job
    info = sf.info(source)
    dtype = "int32" if info.subtype in INTEGER_SUBTYPES else "float64"
    audio, sample_rate = sf.read(source, dtype=dtype, always_2d=True)
    channel = None
    if stereo == "louder" and audio.shape[1] == 2:
        channel = louder_channel(audio, sample_rate)
        audio = audio[:, [channel]]
    bounds = trim_bounds(audio, sample_rate, keep)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if bounds is None:
        sf.write(target, audio, sample_rate, subtype=info.subtype)
        return source, len(audio) / sample_rate, len(audio) / sample_rate, None, channel, "no sound above threshold; copied unchanged"
    start, end, threshold = bounds
    trimmed = fade_edges(audio[start:end], sample_rate, start > 0, end < len(audio))
    sf.write(target, trimmed, sample_rate, subtype=info.subtype)

    # Verify: same rate and format, and every sample between the fades is the source's.
    check, check_rate = sf.read(target, dtype=dtype, always_2d=True)
    fade = min(int(FADE_SECONDS * sample_rate), len(trimmed) // 2)
    inner = slice(fade, len(trimmed) - fade)
    problem = None
    if check_rate != sample_rate or sf.info(target).subtype != info.subtype:
        problem = "format changed"
    elif len(check) != end - start or not np.array_equal(
        check[inner], audio[start:end][inner]
    ):
        problem = "samples differ from the source"
    return source, len(audio) / sample_rate, (end - start) / sample_rate, threshold, channel, problem


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--keep", type=float, default=0.1, help="silence kept at each end, s")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--stereo",
        choices=["keep", "louder"],
        default="keep",
        help="louder: write stereo files as their louder channel (binaural sources)",
    )
    args = parser.parse_args()

    source_root = os.path.abspath(args.input)
    target_root = os.path.abspath(args.output)
    if target_root == source_root or target_root.startswith(source_root + os.sep):
        sys.exit("The output folder must not be the input folder or inside it.")
    jobs = []
    for root, _, names in os.walk(source_root):
        for name in sorted(names):
            if name.lower().endswith(".wav"):
                source = os.path.join(root, name)
                target = os.path.join(target_root, os.path.relpath(source, source_root))
                jobs.append((source, target, args.keep, args.stereo))
    if not jobs:
        sys.exit(f"No .wav files under {source_root}")

    before = after = 0.0
    thresholds, problems = [], []
    folders = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for done, (source, seconds_in, seconds_out, threshold, channel, problem) in enumerate(
            pool.map(process, jobs), 1
        ):
            before += seconds_in
            after += seconds_out
            folder = os.path.relpath(os.path.dirname(source), source_root)
            stats = folders.setdefault(folder, [0, 0.0, 0.0, 0, 0])
            stats[0] += 1
            stats[1] += seconds_in
            stats[2] += seconds_out
            if channel is not None:
                stats[3 + channel] += 1
            if threshold is not None:
                thresholds.append(threshold)
            if problem:
                problems.append((source, problem))
            if done % 500 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} files")

    for folder, (count, seconds_in, seconds_out, left, right) in sorted(folders.items()):
        picked = f", stereo -> left {left} / right {right}" if left or right else ""
        print(
            f"  {folder}: {count} files, {seconds_in / 60:.1f} -> {seconds_out / 60:.1f} min"
            + picked
        )
    print(
        f"{len(jobs)} files: {before / 60:.1f} min -> {after / 60:.1f} min "
        f"({1 - after / before:.1%} removed), kept {args.keep:g} s of silence at each end"
    )
    if thresholds:
        print(
            f"threshold per file: median {np.median(thresholds):.1f} dBFS, "
            f"lowest {min(thresholds):.1f} dBFS"
        )
    for source, problem in problems:
        print(f"  {problem}: {source}")
    if any(problem != "no sound above threshold; copied unchanged" for _, problem in problems):
        sys.exit(1)


if __name__ == "__main__":
    main()
