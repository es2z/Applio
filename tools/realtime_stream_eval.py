"""Convert a recording through the realtime path, without audio devices, and score it.

    env\\python.exe -X utf8 tools\\realtime_stream_eval.py --template <realtime template> \\
        --input voice.wav [--input more.wav] [--preset legacy] [--preset trim] \\
        [--preset "mine={\\"output_position\\": \\"trim\\", \\"chunk_ms\\": 240}"] \\
        [--offline] [--seed 1234] [--out-dir logs/stream_eval/run1]

For each input and each preset this runs exactly what the Realtime tab runs - the same
VoiceChanger, warm-up and RNG seed - feeding the recording block by block, and writes
`<input>__<preset>.wav` next to a `results.json`. Listen to the files; the numbers say
where to listen:

- delay_ms: the window layout's own delay (no devices, no processing time).
- process_p50 / p95: GPU time per block.
- seam: spectral change where blocks are joined divided by the change elsewhere. 1.0
  means the joins cannot be told from the rest; higher means they can.
- mel_vs_offline (with --offline): log-mel distance from the same recording converted
  offline in one piece - how much the streaming itself costs.

Presets are realtime option sets (rvc/realtime/options.py); each is complete, so an
option it does not name is off. Built in: legacy (all off), template (the template's
own options), trim, stage1 (trim + stateful input resampling + aligned gain), locked (stage1 + time
-locked noise, phase continuity, F0 cache and the fixed APEX-GAN noise envelope), fixed
(what the Realtime tab's "keep the current chunk's sound at a shorter chunk" button sets,
worked out from the template's own chunk / crossfade / extra with everything else off). A
preset of the form name=JSON may also override chunk_ms, crossfade_s and extra_s, and
name@N runs a built-in preset at a chunk of N ms. Everything else (model, index, F0
method, embedder, pitch, index rate, protect, chunk, crossfade, extra, VAD) comes from
--template, or from the flags below.

Does a shorter chunk keep the sound? With the template at the chunk that sounds right:

    --preset legacy --preset fixed@240 --preset fixed@160 --preset legacy@240 \\
        --compare-to legacy

- mel_vs_<name> (with --compare-to): log-mel distance from that preset's output, lined
  up for the two delays. The generator's random noise differs between any two layouts,
  so this never reaches 0; legacy@N against fixed@N at the same N says how far each
  moved from the reference.
"""

import argparse
import datetime
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np  # noqa: E402

BUILT_IN_PRESETS = {
    "legacy": {},
    "template": None,  # whatever the template's own options are
    "trim": {"output_position": "trim"},
    "stage1": {
        "output_position": "trim",
        "input_resample": "stateful",
        "gain_mode": "aligned",
    },
    "locked": {
        "output_position": "trim",
        "input_resample": "stateful",
        "gain_mode": "aligned",
        "apex_noise_envelope": "fixed",
        "time_locked_noise": True,
        "phase_continuity": True,
        "f0_cache": True,
    },
    # Worked out in main() from the template's layout (chunk_independent_options).
    "fixed": {"__chunk_independent__": True},
}
LAYOUT_KEYS = ("chunk_ms", "crossfade_s", "extra_s")
CHUNK_INDEPENDENT = "__chunk_independent__"


def parse_preset(text):
    if "=" in text:
        name, body = text.split("=", 1)
        return name.strip(), json.loads(body)
    base, chunk = text, None
    if "@" in text:
        base, chunk = text.split("@", 1)
        try:
            chunk = float(chunk)
        except ValueError:
            raise SystemExit(f"Preset {text!r}: the part after @ is a chunk in ms")
    if base not in BUILT_IN_PRESETS:
        raise SystemExit(
            f"Unknown preset {text!r}; built in: {', '.join(BUILT_IN_PRESETS)}, "
            "or give name=JSON or name@chunk_ms"
        )
    preset = BUILT_IN_PRESETS[base]
    if chunk is None:
        return text, None if preset is None else dict(preset)
    if preset is None:
        raise SystemExit(f"Preset {text!r}: 'template' cannot take a chunk; use name=JSON")
    return text, {**preset, "chunk_ms": chunk}


def resolve_options(body, settings):
    """The option dict a parsed preset stands for, given the template settings."""
    body = dict(settings["options"]) if body is None else dict(body)
    if body.pop(CHUNK_INDEPENDENT, False):
        from rvc.realtime.options import RealtimeOptions, chunk_independent_options

        layout = {key: body[key] for key in LAYOUT_KEYS if key in body}
        body = {
            **chunk_independent_options(
                RealtimeOptions(),
                settings["chunk_ms"],
                settings["crossfade_s"],
                settings["extra_s"],
            ),
            **layout,
        }
    return body


def load_template(name):
    from tabs.realtime.template import RealtimeTemplateManager

    data = RealtimeTemplateManager().load_template(name)
    if data is None:
        raise SystemExit(f"Template {name!r} not found in templates/real_time")
    model = data.get("modelTab", {})
    voice, inference = model.get("voice", {}), model.get("inference", {})
    values, perf = model.get("parameterValues", {}), data.get("performanceTab", {})
    return {
        "model": voice.get("model_path", ""),
        "index": voice.get("index_path", ""),
        "f0_method": inference.get("f0_method", "rmvpe"),
        "fcn_profile": inference.get("fcn_profile") or None,
        "embedder": inference.get("embedder_model", "contentvec"),
        "embedder_custom": inference.get("embedder_model_custom") or None,
        "embedder_precision": inference.get("embedder_precision", "fp32"),
        "sid": int(inference.get("speaker_id", 0) or 0),
        "autotune": bool(inference.get("autotune", False)),
        "autotune_strength": float(inference.get("autotune_strength", 1)),
        "proposed_pitch": bool(inference.get("proposed_pitch", False)),
        "proposed_pitch_threshold": float(inference.get("proposed_pitch_threshold", 155.0)),
        "pitch": float(values.get("pitch", 0)),
        "index_rate": float(values.get("index_rate", 0.75)),
        "volume_envelope": float(values.get("volume_envelope", 1)),
        "protect": float(values.get("protect", 0.5)),
        "chunk_ms": float(perf.get("chunk_size", 512)),
        "crossfade_s": float(perf.get("crossfade_overlap_size", 0.05)),
        "extra_s": float(perf.get("extra_convert_size", 0.5)),
        "silence_threshold": float(perf.get("silence_threshold", -90)),
        "vad": bool(data.get("audioTab", {}).get("vad_enabled", False)),
        "options": data.get("experimentTab", {}),
    }


def load_input(path):
    import librosa

    audio, _ = librosa.load(path, sr=48000, mono=True)
    return audio.astype(np.float32)


def build_changer(settings, options, layout):
    from rvc.realtime.core import VoiceChanger
    from rvc.realtime.options import RealtimeOptions

    chunk_ms = layout.get("chunk_ms", settings["chunk_ms"])
    return VoiceChanger(
        read_chunk_size=int(chunk_ms * 48000 / 1000 / 128),
        cross_fade_overlap_size=layout.get("crossfade_s", settings["crossfade_s"]),
        extra_convert_size=layout.get("extra_s", settings["extra_s"]),
        model_path=settings["model"],
        index_path=settings["index"],
        f0_method=settings["f0_method"],
        embedder_model=settings["embedder"],
        embedder_model_custom=settings["embedder_custom"],
        embedder_precision=settings["embedder_precision"],
        silent_threshold=int(settings["silence_threshold"]),
        vad_enabled=settings["vad"],
        sid=settings["sid"],
        fcn_profile=settings["fcn_profile"],
        options=RealtimeOptions.from_dict(options),
    )


def conversion_args(settings):
    return (
        settings["pitch"],
        settings["index_rate"],
        settings["protect"],
        settings["volume_envelope"],
        settings["autotune"],
        settings["autotune_strength"],
        settings["proposed_pitch"],
        settings["proposed_pitch_threshold"],
    )


def offline_reference(settings, input_path, output_path):
    from rvc.infer.infer import VoiceConverter

    VoiceConverter().convert_audio(
        audio_input_path=input_path,
        audio_output_path=output_path,
        model_path=settings["model"],
        index_path=settings["index"],
        pitch=settings["pitch"],
        f0_method=settings["f0_method"],
        index_rate=settings["index_rate"],
        volume_envelope=settings["volume_envelope"],
        protect=settings["protect"],
        embedder_model=settings["embedder"],
        embedder_model_custom=settings["embedder_custom"],
        sid=settings["sid"],
        export_format="WAV",
    )
    import librosa

    audio, _ = librosa.load(output_path, sr=48000, mono=True)
    return audio.astype(np.float32)


def compare_runs(stream_eval, outputs, reference_name):
    """Add mel_vs_<reference> to every run of one input, lined up for the two delays."""
    reference, reference_row = outputs[reference_name]
    key = f"mel_vs_{reference_name}"
    for name, (output, row) in outputs.items():
        if name == reference_name:
            continue
        expected = (row["delay_ms"] - reference_row["delay_ms"]) * 48
        lag = stream_eval.align(reference, output, expected)
        row[key] = stream_eval.mel_distance(reference, output, lag)
        print(f"  {name}: {key} {row[key]:.4f} (lag {lag / 48.0:+.0f} ms)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--template", help="realtime template name (templates/real_time)")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--preset", action="append", default=[])
    parser.add_argument("--offline", action="store_true", help="also convert offline and compare")
    parser.add_argument(
        "--compare-to", help="a preset name given with --preset; every run is compared to its output"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir")
    for name in ("model", "index", "f0_method", "embedder"):
        parser.add_argument(f"--{name.replace('_', '-')}")
    for name in ("pitch", "index_rate", "protect", "chunk_ms", "crossfade_s", "extra_s"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float)
    parser.add_argument("--no-vad", action="store_true")
    args = parser.parse_args()

    settings = load_template(args.template) if args.template else {
        "model": "", "index": "", "f0_method": "rmvpe", "fcn_profile": None,
        "embedder": "contentvec", "embedder_custom": None, "embedder_precision": "fp32",
        "sid": 0, "autotune": False, "autotune_strength": 1.0, "proposed_pitch": False,
        "proposed_pitch_threshold": 155.0, "pitch": 0.0, "index_rate": 0.75,
        "volume_envelope": 1.0, "protect": 0.5, "chunk_ms": 512.0, "crossfade_s": 0.05,
        "extra_s": 0.5, "silence_threshold": -90.0, "vad": False, "options": {},
    }
    for name in ("model", "index", "f0_method", "embedder"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    for name in ("pitch", "index_rate", "protect", "chunk_ms", "crossfade_s", "extra_s"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    if args.no_vad:
        settings["vad"] = False
    if not settings["model"]:
        raise SystemExit("No model: give --template or --model")

    presets = [parse_preset(text) for text in (args.preset or ["legacy", "stage1"])]
    if args.compare_to and args.compare_to not in [name for name, _ in presets]:
        raise SystemExit(f"--compare-to {args.compare_to!r} is not one of the --preset names")
    out_dir = args.out_dir or os.path.join(
        "logs", "stream_eval", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    import soundfile as sf
    import torch

    from rvc.realtime import stream_eval
    from rvc.realtime.rng import apply_seed
    from rvc.realtime.warmup import warm_up

    results = {"settings": {k: v for k, v in settings.items() if k != "options"}, "runs": []}
    for input_path in args.input:
        audio = load_input(input_path)
        stem = os.path.splitext(os.path.basename(input_path))[0]
        reference = None
        if args.offline:
            reference_path = os.path.join(out_dir, f"{stem}__offline.wav")
            print(f"[offline] {reference_path}")
            reference = offline_reference(settings, input_path, reference_path)
        outputs = {}
        for name, body in presets:
            body = resolve_options(body, settings)
            layout = {key: body.pop(key) for key in LAYOUT_KEYS if key in body}
            # A preset is a complete option set: anything it does not name is off.
            options = body
            print(f"[{stem} / {name}] options={options} layout={layout}")
            changer = build_changer(settings, options, layout)
            warm_up(changer, *conversion_args(settings))
            apply_seed(args.seed)
            output, timings = stream_eval.stream(
                changer, audio, conversion_args(settings)
            )
            path = os.path.join(out_dir, f"{stem}__{name}.wav")
            sf.write(path, output, 48000)
            runtime = changer.vc_model
            row = {
                "input": input_path,
                "preset": name,
                "options": options,
                "layout": layout,
                "wav": path,
                "block_ms": changer.block_frame / 48.0,
                "delay_ms": float(runtime.algorithmic_delay_ms),
                "seam": stream_eval.seam_score(
                    output, changer.block_frame, changer.crossfade_frame
                ),
                **stream_eval.summarise(timings),
            }
            if reference is not None:
                lag = stream_eval.align(reference, output, runtime.algorithmic_delay_ms * 48)
                row["lag_ms"] = lag / 48.0
                row["mel_vs_offline"] = stream_eval.mel_distance(reference, output, lag)
            outputs[name] = (output, row)
            results["runs"].append(row)
            print(
                "  block {block_ms:.0f} ms  delay {delay_ms:.0f} ms  process p50/p95 "
                "{p50_ms:.1f}/{p95_ms:.1f} ms  seam {seam:.3f}".format(**row)
                + (f"  mel_vs_offline {row['mel_vs_offline']:.4f}" if "mel_vs_offline" in row else "")
            )
            del changer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if args.compare_to:
            compare_runs(stream_eval, outputs, args.compare_to)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
