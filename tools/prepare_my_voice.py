"""Cut long gated recordings of one speaker into 5-15 s pieces, holding out each file's end.

    env314\\Scripts\\python.exe tools\\prepare_my_voice.py <input dir> <output dir>
        [--heldout 1.wav 4.wav ...] [--heldout-seconds 30]

Written for recordings made through a live voice chain that ends in a noise gate and a
limiter (Clear -> RNNoise -> gate -> limiter): the gate leaves digital silence between
utterances, so that is where the recordings are cut, never inside a gate-open stretch.
Neighbouring stretches are joined, gaps included, into pieces of at most --max-seconds,
because SPIN drops utterances under 2.5 s and the gated stretches have a median of 3-4 s.

Only the louder channel is kept. The recordings this was written for are stereo with the
voice in the left channel and digital silence in the right; averaging the two, as
preprocess does, would halve the level.

The last --heldout-seconds (or a little more, to land on a gate gap) of every file named
by --heldout go whole into <output>/heldout/ and are never in <output>/train/: they are
the clips the trained models are later judged on. manifest.json records every piece's
source and span. Samples are copied, not processed, at the source rate and subtype.
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

FRAME_SECONDS = 0.01
GAP_SECONDS = 0.3
PAD_SECONDS = 0.1


def louder_channel(audio):
    if audio.ndim == 1:
        return audio, 0
    energy = (audio.astype(np.float64) ** 2).sum(axis=0)
    channel = int(np.argmax(energy))
    return audio[:, channel], channel


def gate_open_stretches(mono, sample_rate):
    """(start, end) sample spans of audio that is not digital silence, gaps < GAP joined."""
    hop = int(sample_rate * FRAME_SECONDS)
    count = len(mono) // hop
    active = np.abs(mono[: count * hop].reshape(count, hop)).max(axis=1) > 0
    edges = np.diff(np.concatenate([[0], active.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    gap = int(GAP_SECONDS / FRAME_SECONDS)
    spans = []
    for s, e in zip(starts, ends):
        if spans and s - spans[-1][1] < gap:
            spans[-1][1] = e
        else:
            spans.append([s, e])
    return [(s * hop, e * hop) for s, e in spans]


def group(spans, sample_rate, max_seconds):
    """Join consecutive spans while the joined piece stays within max_seconds."""
    limit = int(max_seconds * sample_rate)
    pieces = []
    for s, e in spans:
        if pieces and e - pieces[-1][0] <= limit:
            pieces[-1][1] = e
        else:
            pieces.append([s, e])
    return [tuple(p) for p in pieces]


def padded(start, end, total, sample_rate):
    pad = int(PAD_SECONDS * sample_rate)
    return max(0, start - pad), min(total, end + pad)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--heldout", nargs="*", default=[])
    parser.add_argument("--heldout-seconds", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=15.0)
    args = parser.parse_args()

    if os.path.exists(args.output):
        sys.exit(f"{args.output} already exists; refusing to mix runs")
    train_dir = os.path.join(args.output, "train")
    heldout_dir = os.path.join(args.output, "heldout")
    os.makedirs(train_dir)
    os.makedirs(heldout_dir)

    names = sorted(
        (n for n in os.listdir(args.input) if n.lower().endswith(".wav")),
        key=lambda n: (len(n), n),
    )
    missing = set(args.heldout) - set(names)
    if missing:
        sys.exit(f"--heldout names not in {args.input}: {sorted(missing)}")

    manifest = []
    totals = {"train": 0.0, "heldout": 0.0}
    voiced = {"train": 0.0, "heldout": 0.0}
    for name in names:
        path = os.path.join(args.input, name)
        info = sf.info(path)
        audio, sr = sf.read(path, dtype="int32" if info.subtype.startswith("PCM") else "float64")
        mono, channel = louder_channel(audio)
        spans = gate_open_stretches(mono, sr)
        stem = os.path.splitext(name)[0]

        split = len(mono)
        if name in args.heldout:
            # The latest gate gap that leaves at least heldout-seconds after it.
            need = int(args.heldout_seconds * sr)
            last_end = spans[-1][1]
            for s, _ in reversed(spans):
                if last_end - s >= need:
                    split = s
                    break
            else:
                sys.exit(f"{name}: shorter than --heldout-seconds")
            held = [(s, e) for s, e in spans if s >= split]
            a, b = padded(held[0][0], held[-1][1], len(mono), sr)
            a = max(a, split - int(PAD_SECONDS * sr))
            out = os.path.join(heldout_dir, f"{stem}_heldout.wav")
            sf.write(out, mono[a:b], sr, subtype=info.subtype)
            manifest.append({"set": "heldout", "file": os.path.relpath(out, args.output),
                             "source": path, "channel": channel,
                             "start_s": a / sr, "end_s": b / sr})
            totals["heldout"] += (b - a) / sr
            voiced["heldout"] += sum(e - s for s, e in held) / sr
            spans = [(s, e) for s, e in spans if e <= split]

        previous_end = 0
        for i, (s, e) in enumerate(group(spans, sr, args.max_seconds)):
            a, b = padded(s, e, len(mono), sr)
            a = max(a, previous_end)
            b = min(b, split - int(PAD_SECONDS * sr)) if split < len(mono) else b
            previous_end = b
            out = os.path.join(train_dir, f"{stem}_{i:03d}.wav")
            sf.write(out, mono[a:b], sr, subtype=info.subtype)
            manifest.append({"set": "train", "file": os.path.relpath(out, args.output),
                             "source": path, "channel": channel,
                             "start_s": a / sr, "end_s": b / sr})
            totals["train"] += (b - a) / sr
            voiced["train"] += sum(min(e2, b) - max(s2, a) for s2, e2 in spans
                                   if e2 > a and s2 < b) / sr

    with open(os.path.join(args.output, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    for key in ("train", "heldout"):
        count = sum(1 for m in manifest if m["set"] == key)
        print(f"{key}: {count} files, {totals[key] / 60:.2f} min, gate open {voiced[key] / 60:.2f} min")
    lengths = [m["end_s"] - m["start_s"] for m in manifest if m["set"] == "train"]
    print("train piece seconds: min %.1f median %.1f max %.1f, under 2.5 s: %d" % (
        min(lengths), float(np.median(lengths)), max(lengths), sum(x < 2.5 for x in lengths)))


if __name__ == "__main__":
    main()
