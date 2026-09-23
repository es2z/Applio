"""Measure FCNF0++ periodicity on real audio before choosing a voicing threshold.

PENN's entropy periodicity has a floor set by the F0 range: masked-out bins leave a
flat posterior over K allowed bins, whose periodicity is 1 - log(K) / log(1440). A
threshold just above that floor chatters; one in the empty valley between the floor
and the voiced cluster does not. This prints both, and how voicing behaves at a set of
thresholds, so the choice is read off the audio rather than guessed.

    env\\python.exe tools/measure_fcnf0pp_periodicity.py input.wav --csv out.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

THRESHOLDS = (0.065, 0.1, 0.1625, 0.25, 0.4)


def voicing_stats(voiced):
    voiced = np.asarray(voiced, dtype=bool)
    transitions = int((voiced[1:] != voiced[:-1]).sum())
    holes = int((~voiced[1:-1] & voiced[:-2] & voiced[2:]).sum())
    spikes = int((voiced[1:-1] & ~voiced[:-2] & ~voiced[2:]).sum())
    return {
        "voiced_ratio": float(voiced.mean()) if len(voiced) else 0.0,
        "transitions": transitions,
        "one_frame_holes": holes,
        "one_frame_spikes": spikes,
    }


def measure(audio, device, decoder="viterbi", coarse_max=1680.0):
    import penn

    from rvc.lib.predictors.fcnf0pp import FCNF0PPPredictor

    predictor = FCNF0PPPredictor(
        device,
        "fcnf0++",
        {
            "method": "fcnf0++",
            "version": 1,
            "decoder": decoder,
            "periodicity_threshold": None,
            "center": "zero",
            "coarse_min": 50.0,
            "coarse_max": coarse_max,
            "calibrated": False,
        },
    )
    track = predictor.extract_track(audio)
    low = int(penn.convert.frequency_to_bins(torch.tensor(50.0)))
    high = int(penn.convert.frequency_to_bins(torch.tensor(coarse_max), torch.ceil))
    floor = 1 - math.log(high - low) / math.log(penn.PITCH_BINS)
    periodicity = track.periodicity
    report = {
        "frames": len(periodicity),
        "f0_range_hz": [50.0, coarse_max],
        "unvoiced_floor": floor,
        "quantiles": {
            f"{q:g}": float(np.quantile(periodicity, q / 100))
            for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
        "histogram": {
            "edges": np.linspace(0, 1, 21).round(3).tolist(),
            "counts": np.histogram(periodicity, bins=np.linspace(0, 1, 21))[0].tolist(),
        },
        "thresholds": {
            f"{t:g}": voicing_stats(periodicity > t) for t in THRESHOLDS
        },
    }
    return report, track


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decoder", default="viterbi", choices=("viterbi", "argmax"))
    parser.add_argument("--coarse_max", type=float, default=1680.0, choices=(1100.0, 1680.0))
    parser.add_argument("--csv", help="Per-frame seconds, periodicity, raw pitch, RMS dBFS")
    parser.add_argument("--json", help="Write the report here as well")
    args = parser.parse_args()

    from rvc.lib.utils import load_audio_16k

    audio = load_audio_16k(args.audio).astype(np.float32)
    report, track = measure(audio, args.device, args.decoder, args.coarse_max)
    report["audio"] = str(args.audio)
    report["decoder"] = args.decoder

    print(f"{report['frames']} frames, F0 range 50-{args.coarse_max:g} Hz, decoder {args.decoder}")
    print(f"unvoiced floor (flat posterior) = {report['unvoiced_floor']:.4f}")
    print("quantiles: " + ", ".join(f"{k}%={v:.4f}" for k, v in report["quantiles"].items()))
    counts, edges = report["histogram"]["counts"], report["histogram"]["edges"]
    width = max(counts) or 1
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        print(f"  {lo:4.2f}-{hi:4.2f} {count:7d} {'#' * round(40 * count / width)}")
    print("threshold  voiced  transitions  1-frame holes  1-frame spikes")
    for threshold, stats in report["thresholds"].items():
        print(
            f"  {threshold:>7s}  {stats['voiced_ratio']:6.1%}  {stats['transitions']:11d}"
            f"  {stats['one_frame_holes']:13d}  {stats['one_frame_spikes']:14d}"
        )

    if args.csv:
        frame = audio[: len(track.periodicity) * 160].reshape(-1, 160)
        rms = 20 * np.log10(np.sqrt((frame.astype(np.float64) ** 2).mean(1)) + 1e-10)
        with open(args.csv, "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["seconds", "periodicity", "raw_pitch_hz", "rms_dbfs"])
            for row in zip(track.timestamps, track.periodicity, track.raw_pitch_hz, rms):
                writer.writerow([f"{row[0]:.2f}", f"{row[1]:.5f}", f"{row[2]:.2f}", f"{row[3]:.1f}"])
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
