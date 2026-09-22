"""Search candidate RVC profiles using speaker-disjoint labeled NPZ tracks.

Each NPZ: speaker (scalar string), split (calibration/validation/test), audio
(mono 16 kHz), cents/confidence (native 1 ms), voiced and mangio_voiced (10 ms
ground truth and comparator), optional weak_voiced (10 ms reference subset).
The output is experimental until listening evaluation is completed separately.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rvc.lib.predictors.fcn.profiles import FCNProfile


@njit
def counts(confidence, support, reference, weak, resets, enter, exit_threshold):
    state = False
    fp = fn = weak_fn = 0
    for i in range(len(confidence)):
        if resets[i]:
            state = False
        threshold = exit_threshold if state else enter
        state = (
            support[i] > 0
            and support[i] >= exit_threshold
            and confidence[i] >= threshold
        )
        fp += int(state and not reference[i])
        fn += int(not state and reference[i])
        weak_fn += int(not state and weak[i])
    return fp, fn, weak_fn


def prepare(tracks, window):
    confidence, support, reference, weak, resets, comparator = [], [], [], [], [], []
    radius = window // 2
    for track in tracks:
        raw = track["confidence"]
        smooth = raw.copy()
        if radius:
            smooth = np.array(
                [
                    np.median(raw[max(0, i - radius) : i + radius + 1])
                    for i in range(len(raw))
                ]
            )
        n = len(track["voiced"])
        for k in range(n):
            lo, hi = max(0, 10 * k - 5), min(len(raw), 10 * k + 5)
            valid = np.isfinite(track["cents"][lo:hi])
            audible = np.any(track["audio"][max(0, k * 160 - 80) : k * 160 + 80])
            weights = np.minimum(raw[lo:hi], smooth[lo:hi])
            support.append(
                float(weights[valid].max()) if valid.any() and audible else 0
            )
            confidence.append(float(np.median(smooth[lo:hi])))
        reference.extend(track["voiced"])
        weak.extend(track.get("weak_voiced", track["voiced"]))
        comparator.extend(track["mangio_voiced"])
        resets.extend([True] + [False] * (n - 1))
    return tuple(
        np.asarray(value)
        for value in (confidence, support, reference, weak, resets, comparator)
    )


def calibrate(paths, output):
    splits = {name: [] for name in ("calibration", "validation", "test")}
    speakers = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            track = {key: data[key] for key in data.files}
        speaker, split = str(track["speaker"].item()), str(track["split"].item())
        if split not in splits:
            raise ValueError(f"Unknown split: {path}")
        if speaker in speakers and speakers[speaker] != split:
            raise ValueError(
                f"Speaker {speaker} overlaps {speakers[speaker]} and {split}"
            )
        speakers[speaker] = split
        n = len(track["voiced"])
        if (
            n == 0
            or len(track["audio"]) // 160 != n
            or len(track["mangio_voiced"]) != n
        ):
            raise ValueError(f"Invalid 10 ms alignment: {path}")
        if (
            track["cents"].shape != track["confidence"].shape
            or len(track["confidence"]) < 10 * (n - 1) + 1
        ):
            raise ValueError(f"Invalid native track: {path}")
        if not np.isfinite(track["confidence"]).all() or np.any(
            (track["confidence"] < 0) | (track["confidence"] > 1)
        ):
            raise ValueError(f"Invalid confidence: {path}")
        if track["voiced"].dtype != bool or track["mangio_voiced"].dtype != bool:
            raise ValueError("Voicing arrays must be Boolean")
        if "weak_voiced" in track and (
            track["weak_voiced"].shape != (n,)
            or np.any(track["weak_voiced"] & ~track["voiced"])
        ):
            raise ValueError("weak_voiced must be a reference-voiced subset")
        splits[split].append(track)
    if not all(splits.values()):
        raise ValueError(
            "Speaker-disjoint calibration, validation and test splits are all required"
        )
    best = None
    distributions = {}
    for window in (0, 3, 5, 9):
        cal = prepare(splits["calibration"], window)
        distributions[str(window)] = {
            name: np.quantile(
                cal[0][cal[2] == voiced], [0.05, 0.25, 0.5, 0.75, 0.95]
            ).tolist()
            for name, voiced in (("voiced", True), ("unvoiced", False))
            if np.any(cal[2] == voiced)
        }
        conf, support, reference, weak, resets, mangio = prepare(
            splits["validation"], window
        )
        if not reference.any() or reference.all():
            raise ValueError("Validation requires voiced and unvoiced frames")
        max_fp = int(np.sum(mangio & ~reference))
        for enter_step in range(101):
            for exit_step in range(enter_step + 1):
                enter, exit_threshold = enter_step / 100, exit_step / 100
                fp, fn, weak_fn = counts(
                    conf, support, reference, weak, resets, enter, exit_threshold
                )
                if fp > max_fp:
                    continue
                score = (weak_fn, fn, enter_step - exit_step, window, fp, enter_step)
                if best is None or score < best[0]:
                    best = (
                        score,
                        FCNProfile(
                            method="fcn-993-rvc",
                            enter_threshold=enter,
                            exit_threshold=exit_threshold,
                            median_frames=window,
                        ),
                    )
    if best is None:
        raise ValueError(
            "No candidate satisfies the comparator's validation UV→V error bound"
        )
    profile = best[1]
    conf, support, reference, weak, resets, mangio = prepare(
        splits["test"], profile.median_frames
    )
    fp, fn, weak_fn = counts(
        conf,
        support,
        reference,
        weak,
        resets,
        profile.enter_threshold,
        profile.exit_threshold,
    )
    comparator_fp = int(np.sum(mangio & ~reference))
    report = {
        "candidate": profile.to_dict(),
        "validation_score": best[0],
        "test": {
            "uv_to_v": fp,
            "v_to_uv": fn,
            "weak_v_to_uv": weak_fn,
            "mangio_uv_to_v": comparator_fp,
            "uv_to_v_bound_passed": fp <= comparator_fp,
        },
        "calibration_aggregated_confidence_quantiles": distributions,
        "listening_evaluation": "pending",
        "speakers": speakers,
    }
    output = Path(output)
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    if fp > comparator_fp:
        raise ValueError(
            "Candidate failed the held-out test UV→V bound; report saved, no profile published"
        )
    output.write_text(json.dumps(profile.to_dict(), indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    calibrate(args.tracks, args.output)
