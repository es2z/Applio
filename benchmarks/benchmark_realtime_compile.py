"""Replay a template without opening audio devices; isolate each compile case.

Run with env/python.exe -X utf8 benchmarks/benchmark_realtime_compile.py --all.
Raw timings and execution logs go under benchmarks/results. No app settings change.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEMPLATE = "00000_少し低めlittleAlterなし_naru_20260906_kushinada_hubert_large_test1_2150"
CASES = [
    ("none", False, False, False, "reduce-overhead"),
    ("crepe", True, False, False, "reduce-overhead"),
    ("embedder", False, True, False, "reduce-overhead"),
    ("rvc", False, False, True, "reduce-overhead"),
    ("crepe_embedder", True, True, False, "reduce-overhead"),
    ("crepe_rvc", True, False, True, "reduce-overhead"),
    ("embedder_rvc", False, True, True, "reduce-overhead"),
    ("all_reduce", True, True, True, "reduce-overhead"),
    ("all_default", True, True, True, "default"),
    ("all_max", True, True, True, "max-autotune"),
]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def gpu_status():
    proc = subprocess.run([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw",
        "--format=csv,noheader",
    ], capture_output=True, text=True)
    return proc.stdout.strip()


def prepare(out):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import yaml

    template_path = ROOT / "templates" / "real_time" / (TEMPLATE + ".yaml")
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    model_path = ROOT / template["modelTab"]["voice"]["model_path"]
    source_dir = model_path.parent / "sliced_audios_16k"
    sources = sorted(source_dir.glob("*.wav"))[:32]
    assert len(sources) == 32
    waves, manifest = [], []
    for path in sources:
        wave, sr = sf.read(path, dtype="float32")
        assert sr == 16000 and wave.ndim == 1
        waves.append(wave)
        manifest.append({"file": str(path.relative_to(ROOT)), "samples": len(wave),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    wave = resample_poly(np.concatenate(waves), 3, 1).astype(np.float32)
    np.save(out / "input_48k.npy", wave)
    metadata = {
        "template": template, "template_name": TEMPLATE,
        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "sources": manifest, "input_samples": len(wave), "input_seconds": len(wave) / 48000,
        "input_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "platform": platform.platform(), "cpu": platform.processor(),
        "gpu_before": gpu_status(),
        "git_head": subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    write_json(out / "manifest.json", metadata)


def child(args):
    import numpy as np
    import torch
    import torchcrepe
    import faiss
    import triton
    from unittest.mock import patch
    from torch._dynamo.utils import counters
    from rvc.realtime import compile_session
    from rvc.realtime.callbacks import AudioCallbacks
    from tabs.settings.sections import torch_compile as legacy

    out = Path(args.out)
    name, crepe, embedder, rvc, mode = next(c for c in CASES if c[0] == args.case)
    meta = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    template = meta["template"]
    audio, model, perf = template["audioTab"], template["modelTab"], template["performanceTab"]
    inf, values = model["inference"], model["parameterValues"]
    stem = f"{name}_r{args.round}"
    settings_path = out / f"{stem}_config.json"
    write_json(settings_path, {
        "torch_compile_enabled": crepe, "torch_compile_mode": mode,
        "torch_compile_disable_triton": False,
        "realtime_compile_embedder": embedder, "realtime_compile_rvc": rvc,
    })
    wave = np.load(out / "input_48k.npy")
    block_units = int(perf["chunk_size"] * 48000 / 1000 / 128)
    block_size = block_units * 128
    chunks = [np.ascontiguousarray(wave[i:i+block_size])
              for i in range(0, len(wave)-block_size+1, block_size)]
    assert len(chunks) >= args.samples
    request = dict(
        f0_up_key=values["pitch"], index_rate=values["index_rate"], protect=values["protect"],
        volume_envelope=values["volume_envelope"], f0_autotune=inf["autotune"],
        f0_autotune_strength=inf["autotune_strength"], proposed_pitch=inf["proposed_pitch"],
        proposed_pitch_threshold=inf["proposed_pitch_threshold"],
    )
    torch.manual_seed(20260912)
    np.random.seed(20260912)
    data = {
        "case": name, "round": args.round, "flags": [crepe, embedder, rvc], "mode": mode,
        "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
        "python": sys.version, "torch_threads": torch.get_num_threads(),
        "faiss_threads": faiss.omp_get_max_threads(), "gpu_before": gpu_status(),
        "samples": args.samples, "profile_samples": args.profile_samples,
    }
    with patch.object(legacy, "CONFIG_PATH", str(settings_path)), patch.object(
        compile_session, "CONFIG_PATH", str(settings_path)
    ):
        # Match the application startup cache policy; never delete existing caches.
        legacy.setup_torch_compile_cache()
        legacy.apply_triton_settings()
        torch.cuda.synchronize()
        started = time.perf_counter()
        callback = AudioCallbacks(
            read_chunk_size=block_units, cross_fade_overlap_size=perf["crossfade_overlap_size"],
            extra_convert_size=perf["extra_convert_size"],
            model_path=model["voice"]["model_path"], index_path=model["voice"]["index_path"],
            f0_method=inf["f0_method"], embedder_model=inf["embedder_model"],
            embedder_model_custom=inf["embedder_model_custom"],
            embedder_precision=inf["embedder_precision"], silent_threshold=perf["silence_threshold"],
            vad_enabled=audio["vad_enabled"], sid=inf["speaker_id"],
            hybrid_blend_ratio=values["hybrid_blend_ratio"],
            input_audio_gain=audio["input"]["gain"] / 100,
            output_audio_gain=audio["output"]["gain"] / 100,
            **request,
        )
        torch.cuda.synchronize()
        data["construction_seconds"] = time.perf_counter() - started
        runtime = callback.vc.vc_model
        pipeline = runtime.pipeline
        session = pipeline.compile_session
        data["input_window_samples_16k"] = runtime.convert_buffer.numel()
        data["faiss_index_vectors"] = pipeline.index.ntotal if pipeline.index is not None else None
        assert pipeline.index is not None

        def verify():
            assert (session.embedder.compiled is not None) == embedder, session.status()
            assert (session.rvc.compiled is not None) == rvc, session.status()
            assert hasattr(torchcrepe.infer.model, "_orig_mod") == crepe

        def call(chunk):
            result, volume, _, error = callback.change_voice(chunk, **request)
            if error is not None:
                raise RuntimeError(str(error))
            return result

        started = time.perf_counter()
        for i in range(12):
            result = call(chunks[i])
            assert np.isfinite(result).all()
        torch.cuda.synchronize()
        data["extra_warmup_seconds"] = time.perf_counter() - started
        verify()
        data["graphs_after_warmup"] = dict(counters["stats"])
        print(f"READY {name} round={args.round} startup={data['construction_seconds']:.2f}s", flush=True)

        # Discard warmup audio state and prefill on the same last three chunks.
        def reset():
            runtime.flush_buffers()
            runtime.consecutive_silence_frames = 0
            callback.vc.sola_buffer.zero_()
            for chunk in chunks[-3:]:
                call(chunk)
            torch.cuda.synchronize()

        reset()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for chunk in chunks[:args.samples]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = call(chunk)
            torch.cuda.synchronize()
            timings.append((time.perf_counter()-started)*1000)
            assert result.shape == (block_size,) and np.isfinite(result).all()
        data["total_ms"] = timings
        data["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        data["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        data["graphs_after_total"] = dict(counters["stats"])
        verify()

        # Separate profiling pass: explicit synchronization attributes CPU + GPU
        # work to each stage. These timings are not the headline latency samples.
        stages = {key: [] for key in ("f0", "embedder", "rvc", "index")}
        current = {}

        class Timed:
            def __init__(self, key, function):
                self.key, self.function = key, function

            def __getattr__(self, key):
                return getattr(self.function, key)

            def __call__(self, *a, **kw):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                output = self.function(*a, **kw)
                torch.cuda.synchronize()
                current[self.key] = current.get(self.key, 0) + (time.perf_counter()-tick)*1000
                return output

        reset()
        originals = (pipeline.get_f0, session.embedder, session.rvc, pipeline._retrieve_speaker_embeddings)
        pipeline.get_f0 = Timed("f0", originals[0])
        session.embedder = Timed("embedder", originals[1])
        session.rvc = Timed("rvc", originals[2])
        pipeline._retrieve_speaker_embeddings = Timed("index", originals[3])
        profiled, other = [], []
        for chunk in chunks[:args.profile_samples]:
            current.clear()
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = call(chunk)
            torch.cuda.synchronize()
            total = (time.perf_counter()-started)*1000
            assert np.isfinite(result).all()
            profiled.append(total)
            for key in stages:
                stages[key].append(current[key])
            other.append(total-sum(current.values()))
        data["stages_ms"] = stages
        data["profiled_total_ms"] = profiled
        data["other_ms"] = other
        data["graphs_after_profile"] = dict(counters["stats"])
        pipeline.get_f0, session.embedder, session.rvc, pipeline._retrieve_speaker_embeddings = originals
        verify()
        data["verified_compiled"] = [hasattr(torchcrepe.infer.model, "_orig_mod"),
                                     session.embedder.compiled is not None, session.rvc.compiled is not None]
        data["gpu_after"] = gpu_status()
        session.close()
    data["success"] = True
    write_json(out / f"{stem}.json", data)
    print(f"DONE {name} round={args.round} mean={np.mean(timings):.3f}ms", flush=True)


def orchestrate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "manifest.json").exists():
        prepare(out)
    order = []
    for repeat in range(args.rounds):
        cases = list(CASES)
        if repeat:
            random.Random(20260912 + repeat).shuffle(cases)
        for case in cases:
            order.append({"case": case[0], "round": repeat})
    write_json(out / "order.json", order)
    for task in order:
        stem = f"{task['case']}_r{task['round']}"
        destination = out / f"{stem}.json"
        if destination.exists() and json.loads(destination.read_text(encoding="utf-8")).get("success"):
            continue
        print(f"START {stem} {time.strftime('%H:%M:%S')} GPU {gpu_status()}", flush=True)
        command = [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()),
                   "--case", task["case"], "--round", str(task["round"]), "--out", str(out),
                   "--samples", str(args.samples), "--profile-samples", str(args.profile_samples)]
        with (out / f"{stem}.log").open("w", encoding="utf-8") as log:
            try:
                proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                if proc.returncode:
                    print(f"FAILED {stem}: exit {proc.returncode}; see log", flush=True)
                    continue
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT {stem}: see log", flush=True)
                continue
        result = json.loads(destination.read_text(encoding="utf-8"))
        times = result["total_ms"]
        print(f"FINISH {stem} mean={sum(times)/len(times):.3f}ms", flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    os.environ["PYTHONUTF8"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--case", choices=[c[0] for c in CASES])
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--profile-samples", type=int, default=20)
    parser.add_argument("--out", default=str(ROOT / "benchmarks/results/torchcompile_20260912"))
    args = parser.parse_args()
    try:
        if args.all:
            orchestrate(args)
        else:
            child(args)
    except Exception:
        traceback.print_exc()
        raise
