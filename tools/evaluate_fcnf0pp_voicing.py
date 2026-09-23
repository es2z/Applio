"""Evaluate FCNF0++ periodicity thresholds against the FCN-f0 manual annotations.

The same four upstream examples (DTB 4/5/11/28), the same 10 ms grid mapping and the
same frame counts as tools/evaluate_fcn_rvc_presets.py, which chose FCN-993-RVC's
Balanced v1, so the numbers are directly comparable. The threshold is chosen PENN's way
(penn/evaluate: maximise voiced F1 with voiced = periodicity > threshold), and the
balanced-error optimum is reported next to it.

Four examples are a development check, not a speaker-disjoint calibration corpus.
No downloads: pass the path of a local FCN-f0 checkout.

    env\\python.exe tools/evaluate_fcnf0pp_voicing.py <FCN-f0 checkout> --output report.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_fcn_rvc_presets import read_reference

EXAMPLES = ("4", "5", "11", "28")
THRESHOLDS = np.round(np.arange(0.0, 0.6001, 0.0025), 4)
REPORTED = (0.065, 0.1, 0.1625)


def reference_grid(audio, reference_path):
    """Nearest annotation within 5 ms of every 10 ms frame, as the FCN evaluation does."""
    times, ref = read_reference(reference_path)
    grid = np.arange(len(audio) // 160) / 100
    right = np.searchsorted(times, grid).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    index = np.where(abs(times[left] - grid) <= abs(times[right] - grid), left, right)
    usable = abs(times[index] - grid) <= 0.005001
    return ref[index] > 0, ref[index], usable


def scores(periodicity, truth, usable, threshold):
    predicted = periodicity > threshold
    use = usable
    tp = int((predicted & truth & use).sum())
    fp = int((predicted & ~truth & use).sum())
    fn = int((~predicted & truth & use).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    voiced, unvoiced = int((truth & use).sum()), int((~truth & use).sum())
    return {
        "false_voiced": fp,
        "false_unvoiced": fn,
        "unvoiced_frames": unvoiced,
        "voiced_frames": voiced,
        "f1": f1,
        "balanced_error": 0.5 * (fp / unvoiced + fn / voiced),
        "transitions": int((predicted[1:] != predicted[:-1]).sum()),
    }


def evaluate(root, output, decoder, device, lag_compensation_ms=0.0):
    from rvc.lib.predictors.fcnf0pp import FCNF0PPPredictor

    root = Path(root)
    method = "fcnf0++-aligned" if lag_compensation_ms else "fcnf0++"
    predictor = FCNF0PPPredictor(device, method, {
        "method": method, "version": 1, "decoder": decoder,
        "periodicity_threshold": None, "center": "zero",
        "coarse_min": 50.0, "coarse_max": 1680.0, "calibrated": False,
        "lag_compensation_ms": float(lag_compensation_ms),
    })
    periodicity, truth, usable, cents, files = [], [], [], [], []
    for name in EXAMPLES:
        audio_path = root / "examples/DTB/manual/audio/16kHz" / f"{name}.orig-16kHz.wav"
        reference_path = root / "examples/DTB/manual/f0_corrected" / f"{name}.f0_corrected.sdif"
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000 or audio.ndim != 1:
            raise ValueError("Expected upstream mono 16 kHz example")
        track = predictor.extract_track(audio)
        voiced, ref_hz, use = reference_grid(audio, reference_path)
        periodicity.append(track.periodicity)
        truth.append(voiced)
        usable.append(use)
        joint = use & voiced
        cents.append(np.abs(1200 * np.log2(track.raw_pitch_hz[joint] / ref_hz[joint])))
        files.append({
            "name": name,
            "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        })
    periodicity, truth, usable = map(np.concatenate, (periodicity, truth, usable))
    cents = np.concatenate(cents)

    table = {f"{t:g}": scores(periodicity, truth, usable, t) for t in THRESHOLDS}
    f1_best = max(table.items(), key=lambda item: (item[1]["f1"], -float(item[0])))
    balanced_best = min(table.items(), key=lambda item: (item[1]["balanced_error"], float(item[0])))
    report = {
        "scope": "four upstream manually corrected development examples; not a speaker-disjoint or independent held-out evaluation",
        "weight_sha256": predictor.weight_sha256,
        "decoder": decoder,
        "lag_compensation_ms": float(lag_compensation_ms),
        "f0_range_hz": [50.0, 1680.0],
        "files": files,
        "pitch_on_reference_voiced_frames": {
            "median_abs_cents": float(np.median(cents)),
            "p90_abs_cents": float(np.percentile(cents, 90)),
            "over_600_cents": int((cents > 600).sum()),
        },
        "penn_f1_optimum": {"threshold": float(f1_best[0]), **f1_best[1]},
        "balanced_error_optimum": {"threshold": float(balanced_best[0]), **balanced_best[1]},
        "reported": {f"{t:g}": table[f"{t:g}"] for t in REPORTED},
        "table": table,
    }
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def line(label, stats):
        return (
            f"  {label:28s} UV->V {stats['false_voiced']:4d}/{stats['unvoiced_frames']}"
            f" ({stats['false_voiced'] / stats['unvoiced_frames']:5.1%})"
            f"  V->UV {stats['false_unvoiced']:4d}/{stats['voiced_frames']}"
            f" ({stats['false_unvoiced'] / stats['voiced_frames']:5.1%})"
            f"  F1 {stats['f1']:.4f}  transitions {stats['transitions']}"
        )

    print(f"FCNF0++ ({decoder}, lag compensation {lag_compensation_ms:g} ms), 50-1680 Hz, {int(usable.sum())} usable frames")
    print(line(f"PENN F1 optimum {f1_best[0]}", f1_best[1]))
    print(line(f"balanced optimum {balanced_best[0]}", balanced_best[1]))
    for t in REPORTED:
        print(line(f"threshold {t:g}", table[f"{t:g}"]))
    pitch = report["pitch_on_reference_voiced_frames"]
    print(f"  pitch on reference-voiced frames: median {pitch['median_abs_cents']:.1f} c,"
          f" p90 {pitch['p90_abs_cents']:.1f} c, >600 c {pitch['over_600_cents']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decoder", default="viterbi", choices=("viterbi", "argmax"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lag_compensation_ms", type=float, default=0.0,
                        help="Evaluate the -aligned framing with this compensation")
    args = parser.parse_args()
    evaluate(args.upstream, args.output, args.decoder, args.device, args.lag_compensation_ms)


if __name__ == "__main__":
    main()
