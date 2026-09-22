"""CUDA FCN timing and streaming-clock checks; no microphone capture or downloads."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rvc.lib.predictors.fcn import FCNPredictor
from rvc.lib.predictors.fcn.streaming import FCNStream


def benchmark(
    method="fcn-993", profile=None, duration=30, chunk_samples=8192, output=None
):
    if duration <= 0 or chunk_samples <= 0:
        raise ValueError("Duration and chunk size must be positive")
    start = time.perf_counter()
    predictor = FCNPredictor("cuda", method, profile)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    stream = FCNStream(predictor)
    # Warm model kernels and preprocessing before measuring the steady state.
    warm = 0.1 * torch.sin(
        torch.arange(16000, device="cuda") * (2 * torch.pi * 220 / 16000)
    )
    stream.push(warm)
    stream.reset()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    timings = []
    samples = round(duration * 16000)
    frames = 0
    max_buffer = 0
    for position in range(0, samples, chunk_samples):
        size = min(chunk_samples, samples - position)
        clock = torch.arange(position, position + size, device="cuda")
        audio = 0.1 * torch.sin(clock.double() * (2 * torch.pi * 220 / 16000)).float()
        torch.cuda.synchronize()
        start = time.perf_counter()
        track = stream.push(audio)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
        if len(track.frame_index):
            assert track.frame_index[0].item() == frames
            frames += len(track.frame_index)
        max_buffer = max(max_buffer, len(stream.buffer))
    start = time.perf_counter()
    tail = stream.flush()
    torch.cuda.synchronize()
    flush_ms = (time.perf_counter() - start) * 1000
    if len(tail.frame_index):
        assert tail.frame_index[0].item() == frames
    frames += len(tail.frame_index)
    assert frames == samples // 160
    report = {
        "method": method,
        "profile": predictor.profile.to_dict(),
        "weight_sha256": predictor.weight_sha256,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "audio_seconds": duration,
        "chunk_samples": chunk_samples,
        "chunk_ms": chunk_samples / 16,
        "frames": frames,
        "clock_drift_frames": 0,
        "load_seconds": load_seconds,
        "holdback_ms": stream.holdback_samples / 16,
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "max_ms": max(timings),
        "deadline_misses": sum(t > chunk_samples / 16 for t in timings),
        "rtf": (sum(timings) + flush_ms) / (duration * 1000),
        "flush_ms": flush_ms,
        "max_buffer_samples": max_buffer,
        "allocated_delta_bytes": torch.cuda.memory_allocated() - baseline_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "scope": "synthetic F0 streaming only, processed as fast as possible; excludes RVC, audio devices and queues",
    }
    if output:
        Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("fcn-993", "fcn-993-rvc"), default="fcn-993"
    )
    parser.add_argument("--fcn_profile")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--chunk-samples", type=int, default=8192)
    parser.add_argument("--output")
    args = parser.parse_args()
    benchmark(
        args.method, args.fcn_profile, args.duration, args.chunk_samples, args.output
    )
