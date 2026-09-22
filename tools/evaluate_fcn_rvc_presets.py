"""Evaluate FCN RVC defaults against the upstream manual example annotations.

No downloads. Four examples are a development check, not a speaker-disjoint
calibration corpus. SDIF layout: https://sdif.sourceforge.net/standard/sdif-standard.html
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rvc.lib.predictors.fcn import FCNPredictor, FCNProfile
from rvc.lib.predictors.fcn.adapter import FCNRVCAdapter
from tools.calibrate_fcn993 import counts, prepare


def read_reference(path):
    data = Path(path).read_bytes()
    if data[:4] != b"SDIF":
        raise ValueError("Expected SDIF")
    offset = 8 + struct.unpack_from(">I", data, 4)[0]
    values = []
    while offset < len(data):
        signature, size = struct.unpack_from(">4sI", data, offset)
        end = offset + 8 + size
        if end > len(data) or size < 16:
            raise ValueError("Truncated SDIF frame")
        if signature == b"1FQ0":
            timestamp, _, matrices = struct.unpack_from(">dII", data, offset + 8)
            pos = offset + 24
            for _ in range(matrices):
                name, dtype, rows, cols = struct.unpack_from(">4sIII", data, pos)
                width = dtype & 255
                pos += 16
                nbytes = rows * cols * width
                if pos + nbytes > end:
                    raise ValueError("Truncated SDIF matrix")
                if name == b"1FQ0":
                    if dtype not in (4, 8) or rows != 1 or cols < 1:
                        raise ValueError("Unsupported pitch matrix")
                    frequency = np.frombuffer(
                        data, dtype=f">f{width}", count=rows * cols, offset=pos
                    )[0]
                    values.append((timestamp, frequency))
                pos += (nbytes + 7) // 8 * 8
        offset = end
    result = np.asarray(values)
    if not len(result) or np.any(np.diff(result[:, 0]) <= 0):
        raise ValueError("Expected increasing reference timestamps")
    return result[:, 0], result[:, 1]


def evaluate(root, output):
    from rvc.lib.predictors.f0 import MANGIO_CREPE

    root = Path(root)
    predictor = FCNPredictor(
        "cuda",
        "fcn-993-rvc",
        FCNProfile(
            method="fcn-993-rvc",
            enter_threshold=0.6,
            exit_threshold=0.4,
            median_frames=5,
        ),
    )
    tracks = []
    comparator = MANGIO_CREPE("cuda", decoder="viterbi")
    for name in ("4", "5", "11", "28"):
        audio_path = root / "examples/DTB/manual/audio/16kHz" / f"{name}.orig-16kHz.wav"
        reference_path = (
            root / "examples/DTB/manual/f0_corrected" / f"{name}.f0_corrected.sdif"
        )
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000 or audio.ndim != 1:
            raise ValueError("Expected upstream mono 16 kHz example")
        cents, _, confidence = predictor.native(audio)
        times, ref = read_reference(reference_path)
        grid = np.arange(len(audio) // 160) / 100
        right = np.searchsorted(times, grid).clip(0, len(times) - 1)
        left = (right - 1).clip(0)
        index = np.where(
            abs(times[left] - grid) <= abs(times[right] - grid), left, right
        )
        usable = abs(times[index] - grid) <= 0.005001
        with patch(
            "rvc.lib.predictors.f0.get_torch_compile_settings",
            return_value=(False, "default"),
        ):
            mangio = comparator.get_f0(audio, 50, 1100, len(grid), "full_speech")
        track = {
            "audio": audio,
            "cents": cents,
            "confidence": confidence,
            "voiced": ref[index] > 0,
            "mangio_voiced": mangio > 0,
            "reference_hz": ref[index],
            "usable": usable,
            "name": name,
            "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        }
        tracks.append(track)
    # Explore on examples 4/5; all four examples inform the final preset review.
    conf, support, ref, weak, resets, _ = prepare(tracks[:2], 5)
    best = []
    for enter in range(1, 100):
        for exit_threshold in range(1, enter + 1):
            fp, fn, _ = counts(
                conf, support, ref, weak, resets, enter / 100, exit_threshold / 100
            )
            balanced_error = 0.5 * (fp / (~ref).sum() + fn / ref.sum())
            best.append(
                (
                    balanced_error,
                    enter - exit_threshold,
                    enter / 100,
                    exit_threshold / 100,
                )
            )
    best.sort()
    candidates = [
        ("candidate", best[0][2], best[0][3]),
        ("previous-smoke", 0.6, 0.4),
        ("balanced-050-040", 0.5, 0.4),
        ("balanced-030-020", 0.3, 0.2),
    ]
    reports = {}
    for label, enter, exit_threshold in candidates:
        profile = FCNProfile(
            method="fcn-993-rvc",
            enter_threshold=enter,
            exit_threshold=exit_threshold,
            median_frames=5,
        )
        results = []
        for track in tracks:
            result = FCNRVCAdapter(profile)(
                track["cents"],
                track["confidence"],
                len(track["voiced"]),
                track["audio"],
            )
            use, truth = track["usable"], track["voiced"]
            joint = use & truth & result.voiced
            cents_error = abs(
                1200 * np.log2(result.pitch_hz[joint] / track["reference_hz"][joint])
            )
            results.append(
                {
                    "example": track["name"],
                    "frames": int(use.sum()),
                    "voiced_frames": int((truth & use).sum()),
                    "unvoiced_frames": int((~truth & use).sum()),
                    "false_voiced": int((result.voiced & ~truth & use).sum()),
                    "false_unvoiced": int((~result.voiced & truth & use).sum()),
                    "mangio_false_voiced": int(
                        (track["mangio_voiced"] & ~truth & use).sum()
                    ),
                    "mangio_false_unvoiced": int(
                        (~track["mangio_voiced"] & truth & use).sum()
                    ),
                    "median_cents_on_joint_voiced": float(np.median(cents_error)),
                    "p90_cents_on_joint_voiced": float(np.percentile(cents_error, 90)),
                }
            )
        reports[label] = {"profile": profile.to_dict(), "examples": results}
    report = {
        "source_commit": "8a2b530af821319b6badca93c8a0ed1f14bfee3c",
        "weight_sha256": predictor.weight_sha256,
        "scope": "four upstream manually corrected development examples; exploratory search on 4/5; all four inform preset review; not a speaker-disjoint or independent held-out evaluation",
        "files": [
            {k: t[k] for k in ("name", "audio_sha256", "reference_sha256")}
            for t in tracks
        ],
        "selection_top5": best[:5],
        "candidates": reports,
    }
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.upstream, args.output)
