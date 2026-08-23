"""Continuous CUDA benchmark for Mangio-CREPE full and PENN FCNF0++."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch


METHODS = (
    "mangio-crepe-full-eager",
    "mangio-crepe-full-compiled",
    "fcnf0++",
    "fcnf0++-compiled",
    "fcnf0++-speech",
    "fcnf0++-speech-compiled",
)


def _synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(callable_):
    _synchronize()
    started = time.perf_counter()
    result = callable_()
    _synchronize()
    return (time.perf_counter() - started) * 1000, result


def _latency_stats(values):
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": max(values),
    }


def _rolling_chunks(audio, window_samples, advance_samples, iterations):
    required = window_samples + advance_samples * (iterations - 1)
    repeats = max(1, int(np.ceil(required / max(1, audio.shape[0]))))
    stream = np.tile(audio, repeats + 1)
    return [
        np.ascontiguousarray(
            stream[i * advance_samples : i * advance_samples + window_samples],
            dtype=np.float32,
        )
        for i in range(iterations)
    ]


def _make_predictor(method, device, signature):
    from rvc.lib.predictors.f0 import (
        FCNF0PP,
        FCNF0PP_SPEECH,
        MANGIO_CREPE,
    )
    from tabs.settings.sections.torch_compile import RealtimeCompileSettings

    compiled = method.endswith("-compiled")
    settings = RealtimeCompileSettings(
        crepe_enabled=compiled and method.startswith("mangio"),
        rvc_enabled=False,
        mode="reduce-overhead",
        fcnf0pp_enabled=compiled and method.startswith("fcnf0++"),
    )
    if method.startswith("mangio"):
        return MANGIO_CREPE(
            device,
            compile_settings=settings,
            compile_signature=signature,
        )
    predictor = FCNF0PP_SPEECH if "-speech" in method else FCNF0PP
    return predictor(
        device,
        compile_settings=settings,
        compile_signature=signature,
    )


def _extract(predictor, method, audio, p_len):
    if method.startswith("mangio"):
        return predictor.get_f0(audio, 50, 1100, p_len, "full")
    return predictor.get_f0(audio, 50, 1100, p_len)


def _model_load(predictor, method, audio, device):
    if method.startswith("mangio"):
        import torchcrepe

        elapsed, _ = _timed(
            lambda: torchcrepe.load.model(
                device, "full", compile_model=False
            )
        )
        return elapsed

    load_samples = min(audio.shape[0], 1600)
    if load_samples < 1600:
        load_audio = np.pad(audio, (0, 1600 - load_samples))
    else:
        load_audio = audio[:load_samples]
    elapsed, _ = _timed(
        lambda: _extract(predictor, method, load_audio, len(load_audio) // 160)
    )
    return elapsed


def _compile_status(predictor):
    return [
        session.status.__dict__
        for session in getattr(predictor, "_sessions", {}).values()
    ]


def run_worker(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = torch.device(args.device)
    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    full_p_len = audio.shape[0] // 160
    audio = audio[: full_p_len * 160]

    window_samples = round(args.window_ms * 16)
    window_samples -= window_samples % 160
    advance_samples = round(args.advance_ms * 16)
    chunks = _rolling_chunks(
        audio, window_samples, advance_samples, args.iterations
    )
    p_len = window_samples // 160

    torch.empty(1, device=device)
    _synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    predictor = _make_predictor(args.method, device, f"frames={p_len}")
    model_load_ms = _model_load(predictor, args.method, audio, device)
    first_inference_ms, first_f0 = _timed(
        lambda: _extract(predictor, args.method, chunks[0], p_len)
    )
    warmup_ms = []
    for chunk in chunks[:4]:
        elapsed, _ = _timed(
            lambda chunk=chunk: _extract(
                predictor, args.method, chunk, p_len
            )
        )
        warmup_ms.append(elapsed)

    predictor.finish_compile_warmup()
    continuous_ms = []
    output_frames = []
    for chunk in chunks:
        elapsed, f0 = _timed(
            lambda chunk=chunk: _extract(
                predictor, args.method, chunk, p_len
            )
        )
        continuous_ms.append(elapsed)
        output_frames.append(len(f0))

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    compile_status = _compile_status(predictor)

    # Quality extraction is intentionally after VRAM/timing capture. Restore an
    # eager PENN model to avoid compiling a second full-file shape.
    if args.method.startswith("fcnf0++") and args.method.endswith("-compiled"):
        session = predictor._sessions["fcnf0++"]
        if session._eager_model is not None:
            predictor.penn.infer.model = session._eager_model
        predictor.compile_settings = None
    elif args.method == "mangio-crepe-full-compiled":
        from tabs.settings.sections.torch_compile import RealtimeCompileSettings

        predictor.compile_settings = RealtimeCompileSettings()
        predictor._sessions = {}
    quality_f0 = _extract(predictor, args.method, audio, full_p_len)

    result = {
        "method": args.method,
        "device": str(device),
        "window_ms": window_samples / 16,
        "advance_ms": advance_samples / 16,
        "iterations": args.iterations,
        "model_load_ms": model_load_ms,
        "first_inference_ms": first_inference_ms,
        "warmup_ms": warmup_ms,
        "continuous": _latency_stats(continuous_ms),
        "peak_vram_allocated_mib": peak_allocated / 1024**2,
        "peak_vram_reserved_mib": peak_reserved / 1024**2,
        "peak_vram_delta_allocated_mib": (
            peak_allocated - baseline_allocated
        )
        / 1024**2,
        "peak_vram_delta_reserved_mib": (
            peak_reserved - baseline_reserved
        )
        / 1024**2,
        "output_frames": sorted(set(output_frames)),
        "first_output_finite": bool(np.isfinite(first_f0).all()),
        "compile_status": compile_status,
        "quality_f0": np.asarray(quality_f0, np.float32).tolist(),
    }
    print("F0_BENCHMARK_JSON=" + json.dumps(result, separators=(",", ":")))


def _quality(results):
    mangio = next(
        (item for item in results if item["method"].startswith("mangio")),
        None,
    )
    penn = next(
        (item for item in results if item["method"].startswith("fcnf0++")),
        None,
    )
    if mangio is None or penn is None:
        return None
    mangio_f0 = np.asarray(mangio["quality_f0"], np.float32)
    penn_f0 = np.asarray(penn["quality_f0"], np.float32)
    frames = min(len(mangio_f0), len(penn_f0))
    mangio_f0, penn_f0 = mangio_f0[:frames], penn_f0[:frames]
    common = (mangio_f0 > 0) & (penn_f0 > 0)
    cents = np.abs(1200 * np.log2(mangio_f0[common] / penn_f0[common]))
    mangio_voiced = mangio_f0[mangio_f0 > 0]
    penn_voiced = penn_f0[penn_f0 > 0]
    return {
        "frames": frames,
        "mangio_voiced": int(mangio_voiced.size),
        "fcnf0pp_voiced": int(penn_voiced.size),
        "common_voiced": int(common.sum()),
        "median_abs_cents": float(np.median(cents)) if cents.size else None,
        "p95_abs_cents": (
            float(np.percentile(cents, 95)) if cents.size else None
        ),
        "octave_errors_gt_600_cents": int((cents > 600).sum()),
        "mangio_finite": bool(np.isfinite(mangio_f0).all()),
        "fcnf0pp_finite": bool(np.isfinite(penn_f0).all()),
        "mangio_range_hz": (
            [float(mangio_voiced.min()), float(mangio_voiced.max())]
            if mangio_voiced.size
            else None
        ),
        "fcnf0pp_range_hz": (
            [float(penn_voiced.min()), float(penn_voiced.max())]
            if penn_voiced.size
            else None
        ),
    }


def run_parent(args):
    results = []
    for method in args.methods.split(","):
        method = method.strip()
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        command = [
            sys.executable,
            "-m",
            "rvc.lib.predictors.benchmark_f0",
            "--worker",
            "--audio",
            str(Path(args.audio).resolve()),
            "--method",
            method,
            "--device",
            args.device,
            "--iterations",
            str(args.iterations),
            "--window-ms",
            str(args.window_ms),
            "--advance-ms",
            str(args.advance_ms),
        ]
        worker_environment = os.environ.copy()
        if sys.platform == "win32":
            # Inductor's bundled templates are UTF-8. A fresh UTF-8-mode
            # worker avoids Windows' legacy cp932 default when reading them.
            worker_environment["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            env=worker_environment,
        )
        marker = next(
            line
            for line in reversed(completed.stdout.splitlines())
            if line.startswith("F0_BENCHMARK_JSON=")
        )
        results.append(json.loads(marker.split("=", 1)[1]))

    quality = _quality(results)
    for result in results:
        result.pop("quality_f0", None)
    report = {"results": results, "quality": quality}
    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument(
        "--methods",
        default="mangio-crepe-full-compiled,fcnf0++",
    )
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--window-ms", type=float, default=1070)
    parser.add_argument("--advance-ms", type=float, default=512)
    parser.add_argument("--output")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.method is None:
            parser.error("--worker requires --method")
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
