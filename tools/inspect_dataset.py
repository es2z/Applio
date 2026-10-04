"""Inspect a training dataset folder before preprocessing, and say what needs doing.

    env\\python.exe -X utf8 tools\\inspect_dataset.py <dataset dir> [--simulate 100] [--pitch 60]

Reads only. Reports, per speaker folder:

- layout: what preprocess will make of the folder. Integer subfolders become speaker IDs,
  files directly in the root are speaker 0, and any other folder name - including a
  nested one - is skipped by preprocess without an error.
- formats, minutes, level (loudest 10 ms frame), silence at the ends, near-empty files,
  files too short to ever be trained on (the bucket sampler drops clips under 0.375 s at
  48 kHz), clipping (runs of full-scale samples; a single sample at 0 dBFS is just peak
  normalisation and is not counted), and how much content sits above 16 and 20 kHz.
- stereo: whether two channels are the same signal, or the same voice arriving at two
  ears a fraction of a millisecond apart (binaural). Preprocess averages the channels,
  which for the latter is a comb filter; tools/trim_edge_silence.py --stereo louder
  avoids it.
- --simulate N: runs preprocess "Automatic" (with "pre" normalisation) on N files per
  speaker and reports the silence left in the chunks it would train on.
- --pitch N: median F0 of N files per speaker, to catch folders that mix speakers.

Levels are read from a 100 Hz high-passed copy: recordings often carry rumble under
300 Hz that would otherwise pass for sound. See docs/dataset-prep-for-claude.md for how to
read the output and what to do about each finding.
"""

import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from scipy import signal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")  # what preprocess picks up
FRAME = 0.01
SHORTEST_TRAINED_SECONDS = 0.375
CLIP_LEVEL = 0.999
CLIP_RUN = 3
EMPTY_BAND_DB = -80.0


def high_passed(audio, sample_rate):
    sos = signal.butter(4, 100, btype="high", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0)


def frame_rms(x, sample_rate, frame=FRAME):
    hop = max(1, int(sample_rate * frame))
    count = len(x) // hop
    if count == 0:
        return np.zeros((1,) + x.shape[1:]), hop
    return np.sqrt(np.mean(x[: count * hop].reshape((count, hop) + x.shape[1:]) ** 2, axis=1)), hop


def db(x):
    return 20 * np.log10(np.asarray(x) + 1e-12)


def band_levels(mono, sample_rate):
    """Power in 16-20 kHz and above 20 kHz, in dB relative to 1-4 kHz. None if absent."""
    freqs, psd = signal.welch(mono, sample_rate, nperseg=4096)
    reference = psd[(freqs >= 1000) & (freqs < 4000)].sum() + 1e-30
    nyquist = min(sample_rate / 2, 24000)

    def band(low, high):
        if high <= low:
            return None
        return float(10 * np.log10(psd[(freqs >= low) & (freqs < high)].sum() / reference + 1e-30))

    return band(16000, min(20000, nyquist)), band(20000, nyquist)


def stereo_report(channels, sample_rate):
    left, right = channels[:, 0], channels[:, 1]
    if np.array_equal(left, right):
        return {"kind": "identical"}
    levels, hop = frame_rms(channels, sample_rate)
    loudest = levels.max(axis=1)
    speech = loudest >= loudest.max() * 10 ** (-30 / 20)
    mask = np.repeat(speech, hop)[: len(left)]
    if len(mask) < len(left):
        mask = np.concatenate([mask, np.zeros(len(left) - len(mask), bool)])
    L, R = left[mask], right[mask]
    if len(L) < sample_rate // 10:
        return {"kind": "too quiet to tell"}
    rms_l, rms_r = np.sqrt(np.mean(L**2)), np.sqrt(np.mean(R**2))
    span = min(len(L), sample_rate * 3)
    max_lag = int(sample_rate * 0.003)
    xc = signal.correlate(L[:span], R[:span], mode="full", method="fft")
    middle = len(xc) // 2
    window = xc[middle - max_lag : middle + max_lag + 1]
    lag = int(np.argmax(window)) - max_lag
    norm = np.sqrt(np.sum(L[:span] ** 2) * np.sum(R[:span] ** 2)) + 1e-12
    aligned = window.max() / norm
    zero_lag = xc[middle] / norm
    gain = min(rms_l, rms_r) / max(rms_l, rms_r)
    # 100 ms frames where the other ear is clearly louder than the file's louder ear
    coarse, _ = frame_rms(channels, sample_rate, 0.1)
    loud = coarse.max(axis=1) >= coarse.max() * 10 ** (-25 / 20)
    diff = db(coarse[loud, 0]) - db(coarse[loud, 1])
    louder = 1 if rms_l >= rms_r else -1
    switching = float(np.mean((np.sign(diff) != louder) & (np.abs(diff) > 3))) if len(diff) else 0.0
    if zero_lag > 0.99:
        kind = "dual mono"
    elif 0.05 <= abs(lag) / sample_rate * 1000 <= 1.2 and aligned > 0.5:
        kind = "binaural / spaced pair"
    else:
        kind = "other stereo"
    return {
        "kind": kind,
        "level_diff_db": float(db(rms_l) - db(rms_r)),
        "delay_ms": lag / sample_rate * 1000,
        "aligned_corr": float(aligned),
        "comb_notch_db": float(20 * np.log10((1 - gain) / (1 + gain) + 1e-9)),
        "ear_switching": switching,
    }


def inspect_file(path):
    try:
        return _inspect_file(path)
    except Exception as error:  # a file soundfile cannot read must not stop the report
        return {"path": path, "samplerate": 0, "channels": 0, "subtype": "unreadable",
                "seconds": 0.0, "error": str(error)}


def _inspect_file(path):
    info = sf.info(path)
    result = {
        "path": path,
        "samplerate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "seconds": info.frames / info.samplerate,
    }
    if not path.lower().endswith(".wav"):
        return result
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    peaks = np.abs(audio).max(axis=1)
    clipped = peaks >= CLIP_LEVEL
    edges = np.diff(np.concatenate([[0], clipped.astype(np.int8), [0]]))
    runs = np.nonzero(edges == -1)[0] - np.nonzero(edges == 1)[0]
    result["clipped_ms"] = float(runs[runs >= CLIP_RUN].sum() / sample_rate * 1000)
    filtered = high_passed(audio, sample_rate)
    mono = filtered.mean(axis=1)
    levels, hop = frame_rms(mono, sample_rate)
    level_db = db(levels)
    result["loudest_db"] = float(level_db.max())
    threshold = min(-50.0, level_db.max() - 40)
    sound = np.nonzero(level_db >= threshold)[0]
    if len(sound):
        result["leading_ms"] = int(sound[0] * FRAME * 1000)
        result["trailing_ms"] = int((len(level_db) - 1 - sound[-1]) * FRAME * 1000)
        result["silent_share"] = float(np.mean(level_db < threshold))
    raw_mono = audio.mean(axis=1)
    if len(raw_mono) >= 4096:
        result["band_16k_db"], result["band_20k_db"] = band_levels(raw_mono, sample_rate)
    if info.channels == 2:
        result["stereo"] = stereo_report(audio, sample_rate)
    return result


def layout(root):
    """Speaker folders as preprocess sees them, and what it will silently skip."""
    speakers, notes = defaultdict(list), []
    for dirpath, dirnames, filenames in os.walk(root):
        audio = [f for f in filenames if f.lower().endswith(AUDIO_EXTENSIONS)]
        other = [f for f in filenames if not f.lower().endswith(AUDIO_EXTENSIONS)]
        name = os.path.basename(dirpath)
        if dirpath == root:
            sid = 0
            if audio and dirnames:
                notes.append(
                    f"{len(audio)} audio files sit directly in the root next to speaker "
                    "folders: preprocess makes them speaker 0, mixed with folder 0"
                )
        elif name.isdigit():
            sid = int(name)
            if os.path.dirname(dirpath) != root:
                notes.append(f"nested folder {os.path.relpath(dirpath, root)}: becomes speaker {sid}")
        else:
            sid = None
            if audio:
                notes.append(
                    f"{len(audio)} audio files in {os.path.relpath(dirpath, root)}: the "
                    "folder name is not an integer, so preprocess skips them"
                )
        if other:
            exts = Counter(os.path.splitext(f)[1].lower() or "(none)" for f in other)
            notes.append(f"{os.path.relpath(dirpath, root)}: non-audio files {dict(exts)} (ignored)")
        if sid is not None:
            speakers[sid] += [os.path.join(dirpath, f) for f in sorted(audio)]
    ids = sorted(k for k, v in speakers.items() if v)
    if ids and ids != list(range(len(ids))):
        notes.append(
            f"speaker IDs {ids} are not 0..{len(ids) - 1}: model_info records {len(ids)} "
            "speakers, so training from scratch fails on the highest ID"
        )
    return {k: v for k, v in speakers.items() if v}, notes


def quantiles(values, points=(10, 50, 90)):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    return " / ".join(f"{np.percentile(values, p):.1f}" for p in points)


def simulate_preprocess(paths, count):
    """Silence left in the chunks preprocess "Automatic" + "pre" would write."""
    sys.path.insert(0, REPO)
    from rvc.lib.utils import load_audio
    from rvc.train.preprocess.preprocess import OVERLAP, PERCENTAGE, PreProcess
    from rvc.train.preprocess.slicer import Slicer

    sample_rate = 48000
    pp = PreProcess.__new__(PreProcess)
    pp.slicer = Slicer(sr=sample_rate, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
    b, a = signal.butter(N=5, Wn=48, btype="high", fs=sample_rate)
    silent = frames = crops = silent_crops = 0
    leading = []
    for path in random.sample(paths, min(count, len(paths))):
        audio = signal.lfilter(b, a, load_audio(path, sample_rate))
        audio = PreProcess._normalize_audio(pp, audio)
        if audio is None:
            continue
        peak_db = db(frame_rms(audio, sample_rate)[0]).max()
        for segment in pp.slicer.slice(audio):
            i = 0
            while True:
                start = int(sample_rate * (PERCENTAGE - OVERLAP) * i)
                i += 1
                last = len(segment[start:]) <= (PERCENTAGE + OVERLAP) * sample_rate
                chunk = segment[start:] if last else segment[start : start + int(PERCENTAGE * sample_rate)]
                if len(chunk) >= SHORTEST_TRAINED_SECONDS * sample_rate:
                    level = db(frame_rms(chunk, sample_rate)[0])
                    quiet = level < peak_db - 45
                    loud = np.nonzero(~quiet)[0]
                    leading.append((loud[0] if len(loud) else len(level)) * 10)
                    silent += quiet.sum()
                    frames += len(quiet)
                    if len(quiet) >= 36:
                        window = np.convolve(quiet.astype(float), np.ones(36), "valid") / 36
                        silent_crops += int((window >= 0.8).sum())
                        crops += len(window)
                if last:
                    break
    if not frames:
        return "nothing to simulate"
    return (
        f"silent frames {silent / frames:.1%}, 0.36 s crops >=80% silent "
        f"{silent_crops / max(crops, 1):.1%}, chunk leading silence p90 "
        f"{np.percentile(leading, 90):.0f} ms"
    )


def pitch_summary(paths, count):
    import librosa

    medians = []
    for path in random.sample(paths, min(count, len(paths))):
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        audio = librosa.resample(audio.mean(axis=1), orig_sr=sample_rate, target_sr=16000)
        f0, voiced, _ = librosa.pyin(audio, fmin=60, fmax=900, sr=16000, frame_length=1024)
        if voiced.any():
            medians.append(float(np.nanmedian(f0[voiced])))
    if not medians:
        return "no voiced files"
    p10, p50, p90 = np.percentile(medians, [10, 50, 90])
    return (
        f"per-file median F0 p10/p50/p90 {p10:.0f} / {p50:.0f} / {p90:.0f} Hz, "
        f"p10..p90 spans {12 * np.log2(p90 / p10):.1f} semitones"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--simulate", type=int, default=0, help="files per speaker to run preprocess on")
    parser.add_argument("--pitch", type=int, default=0, help="files per speaker to estimate F0 on")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--list", type=int, default=10, help="problem files to list per kind")
    args = parser.parse_args()
    random.seed(0)

    root = os.path.abspath(args.dataset)
    speakers, notes = layout(root)
    print(f"Dataset {root}")
    if not speakers:
        sys.exit("No audio files that preprocess would pick up.")
    print("Layout:")
    print(f"  speakers {sorted(speakers)} -> preprocess records speakers_id {len(speakers)}")
    for note in notes:
        print(f"  ! {note}")

    problems = defaultdict(list)
    for sid in sorted(speakers):
        with ThreadPoolExecutor(args.workers) as pool:
            rows = list(pool.map(inspect_file, speakers[sid]))
        minutes = sum(r["seconds"] for r in rows) / 60
        formats = Counter((r["samplerate"], r["channels"], r["subtype"]) for r in rows)
        print(f"\nSpeaker {sid}: {len(rows)} files, {minutes:.1f} min")
        print("  formats: " + ", ".join(f"{sr} Hz {ch}ch {st} x{n}" for (sr, ch, st), n in formats.most_common()))
        wav = [r for r in rows if "loudest_db" in r]
        print(f"  loudest 10 ms frame dBFS p10/p50/p90: {quantiles([r['loudest_db'] for r in wav])}")
        print(
            f"  silence ms at start p10/p50/p90: {quantiles([r.get('leading_ms') for r in wav])}, "
            f"at end: {quantiles([r.get('trailing_ms') for r in wav])}"
        )
        short = [r for r in rows if r["seconds"] < SHORTEST_TRAINED_SECONDS]
        empty = [r for r in wav if r["loudest_db"] < -40]
        clipped = [r for r in wav if r.get("clipped_ms", 0) > 0]
        no16 = [r for r in wav if r.get("band_16k_db") is not None and r["band_16k_db"] < EMPTY_BAND_DB]
        no20 = [r for r in wav if r.get("band_20k_db") is not None and r["band_20k_db"] < EMPTY_BAND_DB]
        print(
            f"  under {SHORTEST_TRAINED_SECONDS} s (never trained): {len(short)}, near-empty "
            f"(loudest frame < -40 dBFS): {len(empty)}, clipped: {len(clipped)}"
        )
        print(
            f"  content 16-20 kHz vs 1-4 kHz dB p10/p50/p90: {quantiles([r.get('band_16k_db') for r in wav])}; "
            f"above 20 kHz: {quantiles([r.get('band_20k_db') for r in wav])}; "
            f"empty above 16 kHz: {len(no16)}, above 20 kHz: {len(no20)}"
        )
        stereo = [r["stereo"] for r in wav if "stereo" in r]
        if stereo:
            kinds = Counter(s["kind"] for s in stereo)
            print(f"  stereo: {dict(kinds)}")
            spaced = [s for s in stereo if s["kind"] in ("binaural / spaced pair", "other stereo")]
            if spaced:
                print(
                    f"    delay ms p10/p50/p90 {quantiles([s['delay_ms'] for s in spaced])}, "
                    f"aligned corr p50 {np.median([s['aligned_corr'] for s in spaced]):.2f}, "
                    f"L-R level dB {quantiles([s['level_diff_db'] for s in spaced])}"
                )
                deep = np.mean([s["comb_notch_db"] < -10 for s in spaced])
                switch = np.mean([s["ear_switching"] > 0.2 for s in spaced])
                print(
                    f"    averaging to mono puts comb notches deeper than -10 dB in {deep:.0%} of them; "
                    f"the louder ear changes mid-file in {switch:.0%}"
                )
        unreadable = [r for r in rows if "error" in r]
        if unreadable:
            print(f"  unreadable: {len(unreadable)}")
        for kind, items in (("clipped", clipped), ("near-empty", empty), ("unreadable", unreadable)):
            for r in items:
                problems[kind].append((sid, r))
        if args.simulate:
            print(f"  after preprocess (Automatic, pre): {simulate_preprocess(speakers[sid], args.simulate)}")
        if args.pitch:
            print(f"  pitch: {pitch_summary(speakers[sid], args.pitch)}")

    for kind, items in problems.items():
        print(f"\n{kind} files ({len(items)}):")
        for sid, r in sorted(items, key=lambda x: -x[1].get("clipped_ms", 0))[: args.list]:
            if kind == "clipped":
                detail = f"{r['clipped_ms']:.1f} ms clipped"
            elif kind == "near-empty":
                detail = f"loudest {r['loudest_db']:.1f} dBFS"
            else:
                detail = r["error"]
            print(f"  [{sid}] {detail}, {r['seconds']:.2f} s  {os.path.relpath(r['path'], root)}")


if __name__ == "__main__":
    main()
