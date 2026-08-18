"""Offline benchmark for the fixed-Chunk realtime path and torch.compile profiles.

This intentionally opens no PortAudio devices, so it can be run while a game is
providing the representative GPU contention. Example:

    env\python.exe -m rvc.realtime.benchmark --model logs/model.pth \
        --f0 mangio-crepe-full --chunk 960 --modes default,reduce-overhead
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from rvc.realtime.callbacks import AudioCallbacks
from rvc.realtime.core import AUDIO_SAMPLE_RATE
from tabs.settings.sections.torch_compile import (
    TORCH_COMPILE_MODES,
    RealtimeCompileSettings,
)


def _percentile(values, percentile):
    return float(np.percentile(values, percentile)) if values else 0.0


def run_case(args, mode: str) -> dict:
    read_chunk_size = int(args.chunk * AUDIO_SAMPLE_RATE / 1000 / 128)
    block_frames = read_chunk_size * 128
    effective_chunk_ms = block_frames / AUDIO_SAMPLE_RATE * 1000
    settings = RealtimeCompileSettings(
        crepe_enabled=args.compile_crepe,
        rvc_enabled=args.compile_rvc,
        mode=mode,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    created = time.perf_counter()
    callbacks = AudioCallbacks(
        read_chunk_size=read_chunk_size,
        cross_fade_overlap_size=args.crossfade,
        extra_convert_size=args.extra,
        model_path=args.model,
        index_path=args.index or "",
        f0_method=args.f0,
        embedder_model=args.embedder,
        compile_settings=settings,
    )
    load_ms = (time.perf_counter() - created) * 1000
    warmup = callbacks.warmup()

    # A voiced signal exercises CREPE paths unlike an all-zero benchmark.
    phase = np.arange(block_frames, dtype=np.float32)
    sample = (0.1 * np.sin(2 * np.pi * 220 * phase / AUDIO_SAMPLE_RATE)).astype(
        np.float32
    )
    timings = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        output, _, _, _ = callbacks.change_voice(sample)
        if not np.all(np.isfinite(output)):
            raise RuntimeError("Benchmark produced non-finite audio")
        timings.append((time.perf_counter() - started) * 1000)

    statuses = [status.__dict__ for status in callbacks.compile_statuses()]
    return {
        "mode": mode,
        "compile_crepe": args.compile_crepe,
        "compile_rvc": args.compile_rvc,
        "processing": "fixed_chunk",
        "requested_chunk_ms": args.chunk,
        "effective_chunk_ms": effective_chunk_ms,
        "load_ms": load_ms,
        "warmup_ms": warmup,
        "p50_ms": _percentile(timings, 50),
        "p95_ms": _percentile(timings, 95),
        "p99_ms": _percentile(timings, 99),
        "mean_ms": statistics.fmean(timings),
        "realtime_ratio": statistics.fmean(timings) / effective_chunk_ms,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        ),
        "compile_status": statuses,
    }


def _markdown(results: list[dict]) -> str:
    rows = [
        "| mode | chunk ms | p50 | p95 | VRAM MiB | ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        rows.append(
            "| {mode} | {effective_chunk_ms:.1f} | "
            "{p50_ms:.1f} | {p95_ms:.1f} | "
            "{peak_vram_mb:.1f} | {realtime_ratio:.3f} |".format(**result)
        )
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", default="")
    parser.add_argument("--embedder", default="contentvec")
    parser.add_argument("--f0", default="mangio-crepe-full")
    parser.add_argument("--chunk", type=float, default=512)
    parser.add_argument("--crossfade", type=float, default=0.05)
    parser.add_argument("--extra", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--modes", default="default,reduce-overhead,max-autotune,max-autotune-no-cudagraphs")
    parser.add_argument("--compile-crepe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-rvc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", default="realtime-benchmark.json")
    args = parser.parse_args()

    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    unknown = set(modes) - set(TORCH_COMPILE_MODES)
    if unknown:
        parser.error(f"Unknown compile mode(s): {', '.join(sorted(unknown))}")
    results = [run_case(args, mode) for mode in modes]
    output = Path(args.output)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(results), encoding="utf-8")
    print(_markdown(results))


if __name__ == "__main__":
    main()
