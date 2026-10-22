"""Voiced ratio an APEX-GAN model's noise envelope was normalised against in training.

    env\\python.exe -X utf8 tools\\apex_noise_voiced_ratio.py <model name or logs folder>

APEX-GAN divides its noise envelope by the envelope's mean over the training batch
(rvc/lib/algorithm/generators/apex_gan.py). That mean depends on how much of the batch
was voiced: voiced frames carry 0.003 of noise, unvoiced ones 0.033. In realtime the
"fixed" APEX-GAN noise envelope option (rvc/realtime/options.py) replaces the window's
own mean with the mean a training batch would have had, which needs this ratio.

A training batch is one random segment_size window from each clip, so the ratio is the
expected voiced fraction of such a window, averaged over the filelist's lines - mute
lines included, because training draws them too. Clips outside the bucket sampler's
length boundaries are not excluded, so treat the result as accurate to a percent or so.
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def window_voiced_fraction(f0, frames):
    """Mean voiced fraction over every `frames`-long window of `f0` (whole clip if shorter)."""
    voiced = (np.asarray(f0) > 0).astype(np.float64)
    if voiced.size == 0:
        return 0.0
    if voiced.size <= frames:
        return float(voiced.mean())
    sums = np.convolve(voiced, np.ones(frames), mode="valid")
    return float(sums.mean() / frames)


def model_folder(name):
    if os.path.isdir(name):
        return name
    return os.path.join(REPO, "logs", name)


def segment_frames(folder):
    try:
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        return config["train"]["segment_size"] // config["data"]["hop_length"]
    except (OSError, KeyError, ValueError):
        return 36  # 17280 / 480, the stock 48k config


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="model name under logs/, or a folder")
    args = parser.parse_args()

    folder = model_folder(args.model)
    filelist = os.path.join(folder, "filelist.txt")
    frames = segment_frames(folder)
    rows = []
    if os.path.exists(filelist):
        with open(filelist, encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("|")
                if len(parts) >= 4:
                    rows.append((parts[3], "mute" in os.path.basename(parts[0])))
    else:
        f0_dir = os.path.join(folder, "f0_voiced")
        if not os.path.isdir(f0_dir):
            sys.exit(f"No filelist.txt or f0_voiced/ in {folder}")
        rows = [(os.path.join(f0_dir, name), False) for name in sorted(os.listdir(f0_dir))]

    ratios, speech = [], []
    for path, is_mute in rows:
        if not os.path.exists(path):
            continue
        ratio = window_voiced_fraction(np.load(path), frames)
        ratios.append(ratio)
        if not is_mute:
            speech.append(ratio)
    if not ratios:
        sys.exit("No F0 files found")

    print(f"{folder}: {len(ratios)} lines, segment {frames} frames")
    print(f"  voiced ratio as trained (mute lines included): {np.mean(ratios):.3f}")
    if len(speech) != len(ratios):
        print(f"  speech clips only:                             {np.mean(speech):.3f}")
    print("Use the first number for 'Voiced ratio of the training data'.")


if __name__ == "__main__":
    main()
