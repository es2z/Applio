"""Compare F0 methods on one recording, and optionally render each through a model.

Every method goes through rvc.infer.pipeline.Pipeline.get_f0, the same call offline
conversion makes, on the same 16 kHz audio and 10 ms grid. For each method this reports
voicing (ratio, transitions, one-frame holes and spikes) and, against a reference
method, the pitch difference on frames both call voiced and the time lag at which the
two contours agree best (positive = the method runs late).

With --pth, each method also converts the recording through core.py infer with pitch
shift 0, no index and otherwise default settings, so the outputs differ only in F0.

    env\\python.exe tools/compare_f0_methods.py input.wav --output compare.json ^
        --pth logs\\MODEL\\MODEL.pth --embedder_model kushinada-hubert-large --render_dir out
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_METHODS = ("fcnf0++", "fcnf0++-rvc", "fcn-993-rvc", "mangio-crepe-full-speech", "rmvpe")


def voicing(f0):
    voiced = np.asarray(f0) > 0
    return {
        "voiced_ratio": float(voiced.mean()),
        "transitions": int((voiced[1:] != voiced[:-1]).sum()),
        "one_frame_holes": int((~voiced[1:-1] & voiced[:-2] & voiced[2:]).sum()),
        "one_frame_spikes": int((voiced[1:-1] & ~voiced[:-2] & ~voiced[2:]).sum()),
    }


def agreement(f0, reference):
    both = (f0 > 0) & (reference > 0)
    cents = np.abs(1200 * np.log2(f0[both] / reference[both]))
    return {
        "common_voiced_frames": int(both.sum()),
        "voiced_only_here": int(((f0 > 0) & (reference <= 0)).sum()),
        "voiced_only_in_reference": int(((f0 <= 0) & (reference > 0)).sum()),
        "median_abs_cents": float(np.median(cents)) if len(cents) else None,
        "p95_abs_cents": float(np.percentile(cents, 95)) if len(cents) else None,
        "over_600_cents": int((cents > 600).sum()),
        "best_lag_ms": best_lag_ms(f0, reference),
    }


def best_lag_ms(f0, reference, span=30):
    """Shift f0 by fractions of a frame; the lag with the smallest mean cents error."""
    n = min(len(f0), len(reference))
    grid = np.arange(n, dtype=np.float64)
    best = None
    for lag in np.arange(-span, span + 1, 1.0):
        position = grid + lag / 10.0
        inside = (position >= 0) & (position <= n - 1)
        lo = np.floor(position[inside]).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        weight = position[inside] - lo
        a, b, ref = f0[lo], f0[hi], reference[:n][inside]
        ok = (a > 0) & (b > 0) & (ref > 0)
        if ok.sum() < 20:
            continue
        shifted = 2 ** ((1 - weight[ok]) * np.log2(a[ok]) + weight[ok] * np.log2(b[ok]))
        error = np.abs(1200 * np.log2(shifted / ref[ok]))
        error = error[error < 300]  # octave errors say nothing about timing
        if len(error) and (best is None or error.mean() < best[0]):
            best = (float(error.mean()), float(lag))
    return None if best is None else best[1]


def extract(audio, methods, device, fcn_profile=None):
    from rvc.infer.pipeline import Pipeline

    config = SimpleNamespace(x_pad=1, x_query=6, x_center=38, x_max=41, device=device)
    pipeline = Pipeline(48000, config)
    p_len = len(audio) // 160
    tracks = {}
    for method in methods:
        if fcn_profile and method.startswith("fcn"):
            configure = pipeline.configure_fcnf0pp if "f0++" in method else pipeline.configure_fcn
            configure(method, fcn_profile)
        _, f0 = pipeline.get_f0(audio, p_len, method, pitch=0)
        tracks[method] = np.asarray(f0, dtype=np.float64)[:p_len]
    return tracks


def render(audio_path, methods, pth, embedder, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for method in methods:
        target = out_dir / f"{Path(audio_path).stem}__{method}.wav"
        subprocess.run(
            [sys.executable, str(ROOT / "core.py"), "infer",
             "--input_path", str(audio_path), "--output_path", str(target),
             "--pth_path", str(pth), "--index_path", "", "--f0_method", method,
             "--embedder_model", embedder, "--pitch", "0"],
            check=True, cwd=ROOT,
        )
        outputs[method] = str(target)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--reference", default="rmvpe", help="Method the others are compared against")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pth", type=Path, help="Also convert through this model for listening")
    parser.add_argument("--embedder_model", default="contentvec")
    parser.add_argument("--render_dir", type=Path, default=Path("logs/f0_compare"))
    args = parser.parse_args()

    from rvc.lib.utils import load_audio_16k

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.reference not in methods:
        methods.append(args.reference)
    audio = load_audio_16k(args.audio).astype(np.float32)
    tracks = extract(audio, methods, args.device)
    reference = tracks[args.reference]
    report = {"audio": str(args.audio), "frames": len(reference), "reference": args.reference, "methods": {}}
    print(f"{len(reference)} frames; lag and cents are against {args.reference}")
    print(f"{'method':26s} voiced  trans  holes  spikes  med c   p95 c  >600c  lag ms")
    for method, f0 in tracks.items():
        entry = {"voicing": voicing(f0)}
        if method != args.reference:
            entry["against_reference"] = agreement(f0, reference)
        report["methods"][method] = entry
        v, a = entry["voicing"], entry.get("against_reference")
        tail = (
            f"{a['median_abs_cents']:6.1f}  {a['p95_abs_cents']:6.1f}  {a['over_600_cents']:5d}  {a['best_lag_ms']:+6.1f}"
            if a and a["median_abs_cents"] is not None and a["best_lag_ms"] is not None
            else "     -       -      -       -"
        )
        print(f"{method:26s} {v['voiced_ratio']:6.1%} {v['transitions']:6d} {v['one_frame_holes']:6d} {v['one_frame_spikes']:7d}  {tail}")
    if args.pth:
        report["renders"] = render(args.audio, methods, args.pth, args.embedder_model, args.render_dir)
        for method, path in report["renders"].items():
            print(f"rendered {method}: {path}")
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
