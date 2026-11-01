"""Sort a dataset's WAVs into good / bad / other (良 / 悪 / 他) by the rules used for temp3.

    env314\\Scripts\\python.exe tools\\judge_dataset.py <dataset dir> [--copy] [--csv out.csv]
        [--compare <dir with 良/悪/他>]

The rules (a file can trip several; any 悪 reason wins over 他):
- 音割れ (bad): a run of CLIP_RUN or more samples at CLIP_LEVEL or above, as in
  tools/inspect_dataset.py. A limiter's ceiling below CLIP_LEVEL never trips it.
- ほぼ無音 (bad): the peak is under SILENT_PEAK_DBFS.
- 定常ノイズの疑い (bad): a steady floor above 4 kHz closer than FLOOR_REL_DB to the
  speech level (see steady_floor).
- 帯域不足 (other): the long-term spectrum falls by CUT_DEPTH_DB or more across a
  frequency below CUT_MAX_HZ, the mark of an MP3 or a 16 kHz resample. Not bad audio,
  just not 48 kHz audio, so it is kept apart rather than thrown away.

The script that sorted temp3 was not kept, only its 判定結果.csv. These rules were
rebuilt from that CSV's reasons and recalibrated on temp3 itself (--compare reports the
agreement). The noise floor and cut-off measures are this script's own, so their numbers
are not the old CSV's stat_floor_sp / depth2.

Nothing is moved. Without --copy only the counts and the CSV are written; with --copy
every file is copied to <dataset>/<良|悪|他>/<its relative path>, so the originals are
untouched and the copies mirror the speaker folders. Folders named 良, 悪, 他 or 未 at
the top of the dataset are never read as input.
"""

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_dataset import CLIP_LEVEL, CLIP_RUN, db, frame_rms  # noqa: E402

VERDICTS = ("良", "悪", "他")
RESERVED = set(VERDICTS) | {"未"}
SILENT_PEAK_DBFS = -40.0
CUT_SEARCH_LOW_HZ = 8000
CUT_SEARCH_HIGH_HZ = 21000
CUT_MAX_HZ = 20000
CUT_DEPTH_DB = 25.0
FLOOR_BAND_HZ = 4000
FLOOR_FRAME = 0.02
FLOOR_STEADY_DB = 3.0
FLOOR_STEADY_SHARE = 0.2
FLOOR_REL_DB = -45.0
NO_FLOOR = -200.0


def cutoff(mono, sample_rate):
    """(frequency, depth in dB) of the lowest steep fall in the long-term spectrum.

    The lowest fall of CUT_DEPTH_DB or more, not the deepest one: under an MP3's 16 kHz
    cut the residue often falls again at the source's own anti-alias filter (~20.5 kHz),
    and that second fall can be the deeper of the two. Without such a fall it returns
    the deepest one found, which is then too shallow to count.
    """
    nyquist = sample_rate / 2
    if nyquist - 1500 <= CUT_SEARCH_LOW_HZ:
        return float(nyquist), float("inf")
    freqs, psd = signal.welch(mono, sample_rate, nperseg=min(4096, len(mono)))
    level = 10 * np.log10(psd + 1e-30)
    level = np.maximum(level, level.max() - 120)
    found = []
    for hz in np.arange(CUT_SEARCH_LOW_HZ, min(CUT_SEARCH_HIGH_HZ, nyquist - 1500), 50.0):
        below = level[(freqs >= hz - 1500) & (freqs < hz - 250)]
        above = level[(freqs >= hz + 250) & (freqs < hz + 1500)]
        if len(below) and len(above):
            found.append((float(hz), float(np.median(below) - np.median(above))))
    if not found:
        return float(nyquist), 0.0
    for i, (hz, depth) in enumerate(found):
        if depth >= CUT_DEPTH_DB:
            # Report the bottom of this fall: the deepest point within the next 1 kHz.
            return max((f for f in found[i:] if f[0] - hz <= 1000), key=lambda f: f[1])
    return max(found, key=lambda f: f[1])


def steady_floor(mono, sample_rate):
    """The steady high-band floor relative to the speech level, or NO_FLOOR.

    The floor is the 10th percentile of the >4 kHz frame levels, and counts only when at
    least FLOOR_STEADY_SHARE of the frames sit within FLOOR_STEADY_DB of it; it is
    compared with the 95th percentile of the full-band frame levels. Calibrated on temp3:
    this reproduces the old CSV's stat_floor_sp closely (median -34.2 against -34.9 dB
    over its noise files) and catches 33 of its 48 noise files with 16 others flagged
    among 4,393 it called good. Measuring the floor in the pauses only found none of
    them - they are short clips with no pause in them. Digital silence is skipped, never
    a floor.
    """
    sos = signal.butter(4, FLOOR_BAND_HZ, btype="high", fs=sample_rate, output="sos")
    high, _ = frame_rms(signal.sosfiltfilt(sos, mono), sample_rate, FLOOR_FRAME)
    full, _ = frame_rms(mono, sample_rate, FLOOR_FRAME)
    keep = full > 0
    high, full = db(high[keep]), db(full[keep])
    if len(full) < 10:
        return NO_FLOOR
    floor = float(np.percentile(high, 10))
    if np.mean(np.abs(high - floor) <= FLOOR_STEADY_DB) < FLOOR_STEADY_SHARE:
        return NO_FLOOR
    return floor - float(np.percentile(full, 95))


def measure(path):
    try:
        info = sf.info(path)
        audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    except Exception as error:  # an unreadable file is bad, not a crash
        return {"path": path, "error": str(error)}
    peaks = np.abs(audio).max(axis=1)
    clipped = peaks >= CLIP_LEVEL
    edges = np.diff(np.concatenate([[0], clipped.astype(np.int8), [0]]))
    runs = np.nonzero(edges == -1)[0] - np.nonzero(edges == 1)[0]
    mono = audio.mean(axis=1)
    result = {
        "path": path,
        "sr": info.samplerate,
        "dur": round(info.frames / info.samplerate, 3),
        "peak_dbfs": round(float(db(peaks.max())), 3),
        "clip_runs": int(np.sum(runs >= CLIP_RUN)),
    }
    if len(mono) >= 2048:
        cut_hz, depth = cutoff(mono, sample_rate)
        result["cut_hz"], result["cut_depth_db"] = round(cut_hz, 1), round(depth, 2)
        result["floor_rel_db"] = round(steady_floor(mono, sample_rate), 2)
    else:
        result["cut_hz"], result["cut_depth_db"], result["floor_rel_db"] = 0.0, 0.0, NO_FLOOR
    return result


def judge(m):
    if "error" in m:
        return "悪", f"読めない({m['error']})"
    bad, other = [], []
    if m["clip_runs"]:
        bad.append(f"音割れ({m['clip_runs']}箇所)")
    if m["peak_dbfs"] < SILENT_PEAK_DBFS:
        bad.append(f"ほぼ無音(最大{m['peak_dbfs']:.0f}dBFS)")
    if m["floor_rel_db"] > FLOOR_REL_DB:
        bad.append(f"定常ノイズの疑い(一定の高域の床が音声の{-m['floor_rel_db']:.0f}dB下)")
    if m["cut_hz"] < CUT_MAX_HZ and m["cut_depth_db"] >= CUT_DEPTH_DB:
        other.append(f"帯域不足({m['cut_hz'] / 1000:.1f}kHzで遮断)")
    if bad:
        return "悪", "; ".join(bad + other)
    if other:
        return "他", "; ".join(other)
    return "良", ""


def dataset_files(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath == root:
            dirnames[:] = [d for d in dirnames if d not in RESERVED]
        dirnames.sort()
        out += [os.path.join(dirpath, f) for f in sorted(filenames) if f.lower().endswith(".wav")]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--csv")
    parser.add_argument("--compare", action="store_true",
                        help="the dataset is already sorted into 良/悪/他: report agreement")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = os.path.abspath(args.dataset)
    if args.compare:
        paths = [p for v in VERDICTS for p in dataset_files(os.path.join(root, v))]
    else:
        paths = dataset_files(root)
    if args.copy:
        existing = [v for v in VERDICTS if os.path.exists(os.path.join(root, v))]
        if existing:
            sys.exit(f"{root} already has {existing}; refusing to mix runs")

    with ProcessPoolExecutor(args.workers) as pool:
        measured = list(pool.map(measure, paths, chunksize=16))

    rows, counts, agree = [], {v: 0 for v in VERDICTS}, {}
    for m in measured:
        verdict, reason = judge(m)
        rel = os.path.relpath(m["path"], root)
        speaker = rel.split(os.sep)[1 if args.compare else 0] if os.sep in rel else ""
        rows.append({"判定": verdict, "理由": reason, "元ファイル": rel, "話者": speaker,
                     **{k: m.get(k, "") for k in ("sr", "dur", "peak_dbfs", "floor_rel_db",
                                                  "cut_hz", "cut_depth_db", "clip_runs")}})
        counts[verdict] += 1
        if args.compare:
            before = rel.split(os.sep)[0]
            agree[(before, verdict)] = agree.get((before, verdict), 0) + 1

    if args.csv:
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} files:", ", ".join(f"{v} {counts[v]}" for v in VERDICTS))
    reasons = {}
    for r in rows:
        for part in filter(None, r["理由"].split("; ")):
            key = part.split("(")[0]
            reasons[key] = reasons.get(key, 0) + 1
    print("reasons:", reasons)
    if args.compare:
        print("before -> now:", {f"{a}->{b}": n for (a, b), n in sorted(agree.items())})
        same = sum(n for (a, b), n in agree.items() if a == b)
        print(f"agreement {same}/{len(rows)} = {same / len(rows):.4f}")

    if args.copy:
        for r in rows:
            src = os.path.join(root, r["元ファイル"])
            dst = os.path.join(root, r["判定"], r["元ファイル"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        print("copied", len(rows), "files into", ", ".join(os.path.join(root, v) for v in VERDICTS))


if __name__ == "__main__":
    main()
