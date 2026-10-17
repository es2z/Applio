"""Training and inference load of training recipes, side by side.

    env\\python.exe -X utf8 tools\\benchmark_recipe_load.py --model <logs folder> [--recipes A B ...]

Measures, for each recipe, on the GPU and without torch.compile:
  - a whole training step in train.py's order - G forward, D update, G update - under the
    precision train.py would pick from assets/config.json (autocast + GradScaler for fp16),
    cuDNN autotuned: median over --steps after --warmup, and peak memory allocated /
    reserved over the whole run;
  - the discriminator alone, real and fake forward + backward;
  - offline inference, Synthesizer.infer on --infer-seconds of input in eval mode, fp32
    (what rvc/infer/infer.py runs).

Each recipe runs in its own process, and the recipes alternate for --rounds rounds, so a
drift in clocks or background load lands on all of them. Models are built from the folder's
config.json the way train.py builds them and warm started from its G/D (read only), so the
generator's output is bounded from the first step - a fresh decoder can overflow fp16 and
have its first steps skipped. The batch is real clips from the folder cropped to
--clip-seconds, loaded once onto the GPU, so data loading is not timed; only clips whose
spectrogram is already cached are used, so nothing is written into the folder.

Extra losses train.py builds for a generator (SiFi-GAN's source regularisation,
CodenameRingFormer's spectral loss) are included, so the numbers are what a run costs.
"""

import argparse
import contextlib
import io
import json
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="Folder under logs/ to take config, G/D and clips from.")
    parser.add_argument("--recipes", nargs="+", default=["APEX-GAN", "APEX-GAN+"])
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--infer-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-warm-start", action="store_true", help="Train from fresh weights instead.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--recipe", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Worker: one recipe, one process.
# --------------------------------------------------------------------------------------


def train_dtype():
    """The precision train.py picks from assets/config.json."""
    import torch

    try:
        with open(os.path.join(REPO, "assets", "config.json"), encoding="utf-8") as f:
            precision = json.load(f)["precision"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return torch.float32
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def latest_checkpoint(folder, prefix):
    import glob
    import re

    paths = glob.glob(os.path.join(folder, f"{prefix}_*.pth"))
    if not paths:
        return None
    return max(paths, key=lambda p: int("".join(re.findall(r"\d", os.path.basename(p))) or 0))


def load_batch(config, batch_size, clip_seconds, device):
    """batch_size real clips cropped to clip_seconds, stacked in the collate's field order."""
    import torch
    from data_utils import TextAudioLoaderMultiNSFsid

    dataset = TextAudioLoaderMultiNSFsid(config.data)
    frames = round(clip_seconds * config.data.sample_rate / config.data.hop_length)
    hop = config.data.hop_length
    # Only clips whose spectrogram is cached: loading one that is not would write it.
    candidates = [
        i
        for i, (path, *_rest) in enumerate(dataset.audiopaths_and_text)
        if "mute" not in os.path.basename(path)
        and os.path.exists(path.replace(".wav", ".spec.pt"))
    ]
    if not candidates:
        raise SystemExit("No clip in this folder has a cached spectrogram; train it once first.")
    step = max(1, len(candidates) // (batch_size * 2))
    items = []
    for i in candidates[::step] + candidates:
        spec, wav, phone, pitch, pitchf, sid = dataset[i]
        if spec.shape[-1] < frames:
            continue
        items.append(
            (spec[:, :frames], wav[:, : frames * hop], phone[:frames], pitch[:frames], pitchf[:frames], sid)
        )
        if len(items) == batch_size:
            break
    if not items:
        raise SystemExit(f"No clip in this folder is {clip_seconds} s long.")
    while len(items) < batch_size:
        items.append(items[len(items) % max(1, len(items))])
    spec, wav, phone, pitch, pitchf, sid = zip(*items)
    lengths = torch.full((batch_size,), frames, dtype=torch.long)
    batch = (
        torch.stack(phone),
        lengths,
        torch.stack(pitch),
        torch.stack(pitchf),
        torch.stack(spec),
        lengths.clone(),
        torch.stack(wav),
        lengths * hop,
        torch.cat(sid),
    )
    return [t.to(device) for t in batch]


def median_ms(fn, warmup, repeats):
    import torch

    times = []
    for i in range(warmup + repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times), times


def run_worker(args):
    sys.path.insert(0, REPO)
    # train.py's own directory is on its path; data_utils, losses and mel_processing are
    # imported as top-level modules from there.
    sys.path.insert(0, os.path.join(REPO, "rvc", "train"))
    os.chdir(REPO)

    import torch
    from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
    from mel_processing import MultiScaleMelSpectrogramLoss, mel_spectrogram_torch

    from rvc.lib.algorithm import commons
    from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
    from rvc.lib.algorithm.synthesizers import Synthesizer
    from rvc.train.extract.preparing_files import VOCODER_MODEL_KEYS, vocoder_model_config
    from rvc.train.utils import HParams
    from rvc.train.vocoder_recipe import (
        discriminator_version,
        generator_vocoder,
        sample_rate_error,
        uses_multiscale_mel_loss,
    )
    from rvc.train.warm_start import warm_start

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs CUDA.")
    device = torch.device("cuda")
    folder = os.path.join(REPO, "logs", args.model)
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    recipe = args.recipe
    sample_rate = raw["data"]["sample_rate"]
    error = sample_rate_error(recipe, sample_rate)
    if error:
        raise SystemExit(error)
    vocoder, disc_version = generator_vocoder(recipe), discriminator_version(recipe)

    # The decoder keys this vocoder needs, applied the way train.py and the reset apply
    # them, so a recipe whose generator differs from the folder's still builds correctly.
    settings = vocoder_model_config(vocoder, sample_rate)
    for key in VOCODER_MODEL_KEYS:
        if key not in settings:
            raw["model"].pop(key, None)
    raw["model"].update(settings)
    config = HParams(**raw)
    config.data.training_files = os.path.join(folder, "filelist.txt")
    data, train = config.data, config.train
    dtype = train_dtype()
    use_amp = dtype in (torch.float16, torch.bfloat16)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    with contextlib.redirect_stdout(io.StringIO()):
        net_g = Synthesizer(
            data.filter_length // 2 + 1,
            train.segment_size // data.hop_length,
            **raw["model"],
            use_f0=True,
            sr=sample_rate,
            vocoder=vocoder,
            randomized=True,
        )
    net_d = MultiPeriodDiscriminator(
        raw["model"].get("use_spectral_norm", False),
        version=disc_version,
        sample_rate=sample_rate,
        segment_size=train.segment_size,
    )
    inherited = {}
    if not args.no_warm_start:
        identity = {"vocoder": vocoder, "sample_rate": sample_rate, "disc_version": disc_version}
        for tag, net in (("G", net_g), ("D", net_d)):
            path = latest_checkpoint(folder, tag)
            if path is None:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                transfer = warm_start(net, path, tag, target_identity=identity, verbose=False)
            total = sum(v.numel() for v in transfer.target.values())
            inherited[tag] = sum(transfer.target[k].numel() for k in transfer.loaded) / max(total, 1)
    net_g, net_d = net_g.to(device).train(), net_d.to(device).train()

    fn_reg_loss = fn_spectral_loss = None
    if vocoder == "SiFi-GAN" and float(getattr(train, "c_reg", 1.0)) > 0:
        from rvc.train.source_loss import ResidualLoss

        fn_reg_loss = ResidualLoss(sample_rate=sample_rate, hop_size=data.hop_length).to(device)
    if vocoder == "CodenameRingFormer" and float(getattr(train, "c_sd", 0.7)) > 0:
        from rvc.train.spectral_loss import SpectralDistanceLoss

        fn_spectral_loss = SpectralDistanceLoss(
            n_fft=config.model.gen_istft_n_fft, hop_size=config.model.gen_istft_hop_size
        ).to(device)
    multiscale = uses_multiscale_mel_loss(recipe)
    fn_mel_loss = (
        MultiScaleMelSpectrogramLoss(sample_rate=sample_rate) if multiscale else torch.nn.L1Loss()
    )

    optim_g = torch.optim.AdamW(net_g.parameters(), train.learning_rate, betas=train.betas, eps=train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), train.learning_rate, betas=train.betas, eps=train.eps)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)

    with contextlib.redirect_stdout(io.StringIO()):
        batch = load_batch(config, args.batch, args.clip_seconds, device)
    phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave_full, _, sid = batch

    def mel(x):
        return mel_spectrogram_torch(
            x.float().squeeze(1), data.filter_length, data.n_mel_channels, sample_rate,
            data.hop_length, data.win_length, data.mel_fmin, data.mel_fmax,
        )

    skipped = [0]

    def training_step():
        # train.py's step, logging aside.
        with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=dtype):
            y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q), dec_extra = net_g(
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
            )
            wave = commons.slice_segments(
                wave_full, ids_slice * data.hop_length, train.segment_size, dim=3
            )
        with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=dtype):
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
        loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad()
        if dtype == torch.float16:
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            commons.grad_norm(net_d.parameters())
            scaler.step(optim_d)
        else:
            loss_disc.backward()
            commons.grad_norm(net_d.parameters())
            optim_d.step()

        with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=dtype):
            _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
        if multiscale:
            loss_mel = fn_mel_loss(wave, y_hat) * train.c_mel / 3.0
        else:
            loss_mel = fn_mel_loss(mel(wave), mel(y_hat)) * train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * train.c_kl
        loss_gen_all = generator_loss(y_d_hat_g)[0] + feature_loss(fmap_r, fmap_g) + loss_mel + loss_kl
        if fn_reg_loss is not None:
            frames = train.segment_size // data.hop_length
            pitchf_slice = commons.slice_segments(pitchf, ids_slice, frames, 2)
            loss_gen_all = loss_gen_all + fn_reg_loss(
                dec_extra.float(), wave.float(), pitchf_slice.float()
            ) * float(getattr(train, "c_reg", 1.0))
        if fn_spectral_loss is not None:
            magnitude, _ = dec_extra
            loss_gen_all = loss_gen_all + fn_spectral_loss(wave, y_hat, magnitude) * float(
                getattr(train, "c_sd", 0.7)
            )
        # train.py compares the loss to its lowest value every step, which syncs; so does
        # this, and it counts the steps GradScaler will skip.
        if not (torch.isfinite(loss_gen_all) and torch.isfinite(loss_disc)):
            skipped[0] += 1
        optim_g.zero_grad()
        if dtype == torch.float16:
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            commons.grad_norm(net_g.parameters())
            scaler.step(optim_g)
            scaler.update()
        else:
            loss_gen_all.backward()
            commons.grad_norm(net_g.parameters())
            optim_g.step()

    torch.cuda.reset_peak_memory_stats()
    step_ms, step_times = median_ms(training_step, args.warmup, args.steps)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30

    # The discriminator alone: real and fake forward + backward, as in its update.
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=dtype):
        y_hat, ids_slice, *_ = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
        wave = commons.slice_segments(wave_full, ids_slice * data.hop_length, train.segment_size, dim=3)
    y_hat = y_hat.detach()

    def discriminator_step():
        net_d.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=dtype):
            y_d_r, y_d_g, _, _ = net_d(wave, y_hat)
        loss, _, _ = discriminator_loss(y_d_r, y_d_g)
        scaler.scale(loss).backward() if dtype == torch.float16 else loss.backward()

    disc_ms, _ = median_ms(discriminator_step, 5, 20)
    net_d.zero_grad(set_to_none=True)
    optim_g.zero_grad(set_to_none=True)
    del optim_g, optim_d

    # Offline inference: fp32, eval, the first clip tiled to infer_seconds.
    net_g.eval().float()
    frames = round(args.infer_seconds * sample_rate / data.hop_length)
    repeat = -(-frames // phone.shape[1])
    infer_inputs = (
        phone[:1].float().repeat(1, repeat, 1)[:, :frames],
        torch.tensor([frames], device=device),
        pitch[:1].repeat(1, repeat)[:, :frames],
        pitchf[:1].float().repeat(1, repeat)[:, :frames],
        sid[:1],
    )
    outputs = []

    def inference():
        with torch.no_grad():
            outputs.append(net_g.infer(*infer_inputs)[0].shape[-1])

    infer_ms, _ = median_ms(inference, 3, 10)

    result = {
        "recipe": recipe,
        "vocoder": vocoder,
        "disc_version": disc_version,
        "dtype": str(dtype).replace("torch.", ""),
        "gpu": torch.cuda.get_device_name(0),
        "batch": args.batch,
        "inherited": inherited,
        "step_ms": step_ms,
        "step_p10_p90": [
            statistics.quantiles(step_times, n=10)[0],
            statistics.quantiles(step_times, n=10)[-1],
        ],
        "skipped_steps": skipped[0],
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "disc_ms": disc_ms,
        "infer_ms": infer_ms,
        "infer_samples": outputs[-1],
        "infer_seconds": args.infer_seconds,
    }
    print("RESULT " + json.dumps(result))


# --------------------------------------------------------------------------------------
# Parent: alternate the recipes, one process each, and print a table.
# --------------------------------------------------------------------------------------


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
        return

    shared = [
        "--model", args.model, "--batch", str(args.batch), "--warmup", str(args.warmup),
        "--steps", str(args.steps), "--clip-seconds", str(args.clip_seconds),
        "--infer-seconds", str(args.infer_seconds), "--seed", str(args.seed),
    ] + (["--no-warm-start"] if args.no_warm_start else [])
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    runs = {recipe: [] for recipe in args.recipes}
    started = time.time()
    for round_index in range(args.rounds):
        for recipe in args.recipes:
            command = [sys.executable, "-X", "utf8", os.path.abspath(__file__), "--worker", "--recipe", recipe, *shared]
            process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, cwd=REPO)
            lines = [l for l in process.stdout.splitlines() if l.startswith("RESULT ")]
            if process.returncode or not lines:
                print(f"{recipe}: failed (exit {process.returncode})")
                print((process.stdout + process.stderr)[-3000:])
                sys.exit(1)
            result = json.loads(lines[-1][len("RESULT "):])
            runs[recipe].append(result)
            print(
                f"round {round_index + 1} {recipe:<20} step {result['step_ms']:7.1f} ms  "
                f"peak {result['peak_allocated_gib']:.2f} / {result['peak_reserved_gib']:.2f} GiB  "
                f"D {result['disc_ms']:6.1f} ms  infer {result['infer_ms']:6.1f} ms  "
                f"({time.time() - started:.0f} s)",
                flush=True,
            )

    first = runs[args.recipes[0]][0]
    print(
        f"\n{first['gpu']}, {first['dtype']} training (assets/config.json), batch {args.batch} x "
        f"{args.clip_seconds:g} s clips, {args.warmup} + {args.steps} steps per run, "
        f"{args.rounds} alternating runs per recipe, no torch.compile; inference fp32 on "
        f"{args.infer_seconds:g} s"
    )
    # One value per run, in run order, so the spread between runs is visible next to any
    # difference between recipes.
    rows = [("recipe", "discriminator", "step median (ms)", "peak allocated (GiB)", "peak reserved (GiB)", "D alone (ms)", "infer (ms)")]
    for recipe in args.recipes:
        rs = runs[recipe]
        rows.append(
            (
                recipe,
                rs[0]["disc_version"],
                " / ".join(f"{r['step_ms']:.1f}" for r in rs),
                " / ".join(f"{r['peak_allocated_gib']:.2f}" for r in rs),
                " / ".join(f"{r['peak_reserved_gib']:.2f}" for r in rs),
                " / ".join(f"{r['disc_ms']:.1f}" for r in rs),
                " / ".join(f"{r['infer_ms']:.1f}" for r in rs),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    for recipe in args.recipes:
        rs = runs[recipe]
        warm = ", ".join(f"{tag} {share:.1%}" for tag, share in rs[0]["inherited"].items()) or "none"
        print(
            f"  {recipe}: warm start {warm}; skipped fp16 steps {[r['skipped_steps'] for r in rs]}; "
            f"inference output {rs[0]['infer_samples']} samples"
        )


if __name__ == "__main__":
    main()
