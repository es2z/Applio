"""Fine-tune the top layers of a truncated HuBERT Large embedder with SPIN.

    env314\\Scripts\\python.exe -X utf8 tools\\spin\\train.py --dataset temp4 \\
        --cache logs\\_spin\\cache --out logs\\_spin\\run1

Builds the 16 kHz cache on first use (tools/spin/data.py), then trains with the official
SPIN recipe (config/spin.yaml of github.com/vectominist/spin): Adam at 1e-4, weight decay
1e-6, linear warm-up from 1e-6 over 2,500 updates and linear decay to 1e-6 at 5,000,
about 256 s of audio per batch, gradient clipping at 10. bf16 autocast instead of fp16,
because a deep pre-norm transformer's residual stream can overflow fp16.

The forward pass is rvc.lib.utils.embedder_forward itself - the same input normalisation
and dtype handling every extraction uses - so the model is trained on exactly the features
RVC will read from it. Checkpoints hold the trained layers, the head and the optimiser
(ckpt_<step>.pt); tools/spin/export.py turns one into an embedder folder.
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from tools.spin.data import INDEX_FILE, SpinBatches, build_cache  # noqa: E402

# rvc.lib.utils (transformers, librosa, ...) and tools.spin.model are imported inside the
# functions that use them. DataLoader workers are spawned, and a spawned worker re-runs
# this module's top level: keeping it light keeps every worker light.

DEFAULT_MODEL = os.path.join(REPO, "rvc", "models", "embedders", "kushinada_hubert_large")
# A spawned worker holds ~2.6 GB of private DRAM (measured); 3 GB leaves room.
WORKER_DRAM_GB = 3.0
# DRAM left free for everything else on the PC.
DRAM_RESERVE_GB = 16.0


def free_dram_gb():
    """Free DRAM in GB: the smaller of free physical memory and remaining commit. None off
    Windows."""
    if os.name != "nt":
        return None
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return min(status.ullAvailPhys, status.ullAvailPageFile) / 2**30


def worker_ceiling(current):
    """How many DataLoader workers DRAM and the CPU allow, the `current` ones already counted
    in what is used."""
    by_cpu = max(1, (os.cpu_count() or 4) - 4)
    free = free_dram_gb()
    if free is None:
        return by_cpu
    return max(1, min(by_cpu, current + int((free - DRAM_RESERVE_GB) // WORKER_DRAM_GB)))


def latest_checkpoint(out):
    """The checkpoint with the highest step in `out`, or None."""
    best = None
    for name in os.listdir(out):
        if name.startswith("ckpt_") and name.endswith(".pt"):
            ckpt = torch.load(os.path.join(out, name), map_location="cpu", weights_only=True)
            if best is None or ckpt["step"] > best["step"]:
                best = ckpt
    return best


def trained_state(model, train_layers):
    from tools.spin.model import trainable_layers

    n = len(model.encoder.layers)
    return {f"encoder.layers.{n - train_layers + i}.{k}": v.detach().cpu().clone()
            for i, layer in enumerate(trainable_layers(model, train_layers))
            for k, v in layer.state_dict().items()}


def main():
    from rvc.lib.utils import embedder_forward
    from tools.spin.model import SpinHead, load_truncated, set_trainable, warmup_decay

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", type=int, default=22)
    parser.add_argument("--train-layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-seconds", type=float, default=256.0)
    # 1.5 s rather than the official 2.5 s: RVC slices its training audio into ~3 s
    # pieces and realtime feeds ~1-1.5 s windows, and at 2.5 s the target voice's short
    # lines (temp4 speakers 0 and 1) would lose 84% and 48% of their audio.
    parser.add_argument("--min-seconds", type=float, default=1.5)
    parser.add_argument("--max-seconds", type=float, default=17.0)
    parser.add_argument("--sampler", choices=["length", "crop"], default="length",
                        help="'length' is what v1 used; 'crop' draws seconds rather than files "
                             "(tools/spin/data.py)")
    parser.add_argument("--speaker-weight", nargs="*", default=[], metavar="SPEAKER=W",
                        help="crop sampler: multiply a temp4 speaker's chance, e.g. 59=2")
    parser.add_argument("--short-share", type=float, default=0.75)
    parser.add_argument("--short-max-seconds", type=float, default=4.0)
    parser.add_argument("--save", type=int, nargs="+", default=[1000, 2000, 3000, 4000, 5000])
    parser.add_argument("--save-last-every", type=int, default=250,
                        help="also overwrite ckpt_last.pt this often, for --resume")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the newest checkpoint in --out")
    parser.add_argument("--workers", default="auto",
                        help="'auto' starts at up to 4 and adds workers while the GPU waits for "
                             "data, as far as free DRAM allows; or a fixed number")
    parser.add_argument("--chunks", type=int, default=1,
                        help="split each batch into this many forward/backward passes; doubled "
                             "automatically whenever a batch runs out of VRAM")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7122)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-hours", type=float, default=3.5,
                        help="after 50 steps, halve the batch if the run would take longer")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.cache, INDEX_FILE)):
        build_cache(args.dataset, args.cache)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train.log")

    def log(text):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and args.gpu_memory_fraction < 1:
        # Under WDDM the driver silently backs allocations past the card's VRAM with DRAM,
        # so the caching allocator never sees an out-of-memory, never frees its cache, and
        # with every batch a different shape the cache keeps growing: measured 22.3 GB of
        # VRAM plus 43.7 GB of DRAM after 100 steps, and the steps slowed to match. A cap
        # makes the allocator free its cache and retry within VRAM; a batch that still does
        # not fit is split into chunks (train_step) instead of spilling or stopping.
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, index)
    model = load_truncated(args.model, args.layers).to(device)
    params = set_trainable(model, args.train_layers)
    head = SpinHead(dim=model.config.hidden_size).to(device)
    head.normalize_codebook()
    trainable = params + list(head.parameters())
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    log(f"args {json.dumps(vars(args), ensure_ascii=False)}")
    log(f"embedder layers {len(model.encoder.layers)}; trainable embedder tensors {len(names)} "
        f"({sum(p.numel() for p in params):,} parameters): {names[0]} .. {names[-1]}")
    log(f"head {sum(p.numel() for p in head.parameters()):,} parameters; frozen embedder "
        f"{sum(p.numel() for p in model.parameters() if not p.requires_grad):,} parameters")

    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: warmup_decay(s, args.warmup, args.steps, args.lr))

    batch_seconds = args.batch_seconds
    step = 0
    if args.resume:
        ckpt = latest_checkpoint(args.out)
        if ckpt is not None:
            model.load_state_dict(ckpt["state"], strict=False)
            head.load_state_dict(ckpt["head"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            step, batch_seconds = ckpt["step"], ckpt["batch_seconds"]
            log(f"resumed from step {step}")
        else:
            log("resume: no checkpoint yet, starting from step 0")

    def save(path):
        torch.save({"step": step, "layers": args.layers, "train_layers": args.train_layers,
                    "model": args.model, "state": trained_state(model, args.train_layers),
                    "head": head.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "batch_seconds": batch_seconds}, path)

    def loader(batch_seconds, first_step, workers):
        weights = {s: float(w) for s, w in (x.split("=") for x in args.speaker_weight)}
        data = SpinBatches(args.cache, args.steps, batch_seconds, args.min_seconds,
                           args.max_seconds, args.seed, args.sampler, weights,
                           args.short_share, args.short_max_seconds)
        log(f"data: {len(data.items)} files kept ({data.kept_hours:.2f} h), {data.dropped} "
            f"under {args.min_seconds} s dropped; {batch_seconds:g} s per batch; "
            f"{workers} workers; free DRAM {free_dram_gb() or 0:.1f} GB")
        subset = torch.utils.data.Subset(data, range(first_step, args.steps))
        return torch.utils.data.DataLoader(
            subset, batch_size=None, shuffle=False, num_workers=workers,
            prefetch_factor=2, persistent_workers=False)

    def train_step(originals, perturbed, chunks):
        """Forward and backward over the batch in `chunks` passes, gradients accumulated.
        Each chunk's Sinkhorn sees only that chunk's frames. Returns (loss, stats), or a
        non-finite loss before any backward."""
        B = originals.shape[0]
        bounds = [round(i * B / chunks) for i in range(chunks + 1)]
        total, totals = 0.0, {}
        for lo, hi in zip(bounds, bounds[1:]):
            if hi == lo:
                continue
            n = hi - lo
            wavs = torch.cat([originals[lo:hi], perturbed[lo:hi]]).to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                feats = embedder_forward(model, wavs)
                loss, stats = head(feats[:n].reshape(-1, feats.shape[-1]),
                                   feats[n:].reshape(-1, feats.shape[-1]))
            if not torch.isfinite(loss):
                return loss, stats
            (loss * n / B).backward()
            total += loss.item() * n / B
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + v * n / B
            del feats, loss, wavs
        return torch.tensor(total), totals

    auto = args.workers == "auto"
    workers = min(4, worker_ceiling(0)) if auto else int(args.workers)
    chunks = args.chunks
    first_step, t_start, window = step, time.time(), []
    batches, restart = None, True
    while step < args.steps:
        if restart:
            batches = None  # shut the old workers down before spawning new ones
            batches = iter(loader(batch_seconds, step, workers))
            restart, since_restart, waits = False, 0, []
        t0 = time.time()
        try:
            originals, perturbed, failures = next(batches)
        except StopIteration:
            break
        wait = time.time() - t0
        B = originals.shape[0]
        while True:
            try:
                loss, stats = train_step(originals, perturbed, chunks)
                break
            except torch.OutOfMemoryError:
                if chunks >= B:
                    raise
            # Outside the except block, so the traceback no longer holds the failed pass's
            # tensors when the cache is emptied.
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            chunks = min(B, chunks * 2)
            log(f"step {step + 1}: out of VRAM, from now on {chunks} chunks per batch")
        if not torch.isfinite(loss):
            log(f"step {step + 1}: non-finite loss {loss.item()}, stopping")
            sys.exit(2)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        optimizer.step()
        scheduler.step()
        head.normalize_codebook()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        since_restart += 1
        window.append((loss.item(), stats["acc"], stats["code_ppl"], float(grad_norm), B,
                       originals.shape[1] / 16000, failures, wait))
        # Past the batches the workers prefetched while starting, the time spent waiting
        # for the next batch says whether there are enough of them.
        if since_restart > 2 * workers + 5:
            waits.append((wait, time.time() - t0))

        if step == first_step + 50:
            elapsed = time.time() - t_start
            projected = elapsed / 50 * (args.steps - first_step) / 3600
            log(f"speed: {elapsed / 50:.2f} s/step over the first 50, projected {projected:.2f} h")
            if projected > args.max_hours and batch_seconds == args.batch_seconds:
                batch_seconds = args.batch_seconds / 2
                log(f"projected over {args.max_hours} h: batch halved to {batch_seconds:g} s")
                restart = True
        if auto and not restart and len(waits) >= 30:
            ceiling = worker_ceiling(workers)
            share = sum(w for w, _ in waits) / sum(t for _, t in waits)
            new = workers
            if ceiling < workers:
                new = ceiling
            elif share > 0.1 and workers < ceiling:
                new = min(ceiling, workers + 4)
            waits = []
            if new != workers:
                log(f"workers {workers} -> {new}: waiting for data {share:.0%} of the time, "
                    f"DRAM and CPU allow {ceiling}")
                workers, restart = new, True
        if step % 50 == 0 or step == args.steps:
            n = len(window)
            mean = [sum(w[i] for w in window) / n for i in range(4)]
            rate = (time.time() - t_start) / (step - first_step)
            log(f"step {step} lr {scheduler.get_last_lr()[0]:.2e} loss {mean[0]:.4f} "
                f"acc {mean[1]:.3f} code_ppl {mean[2]:.1f} grad {mean[3]:.2f} "
                f"batch {window[-1][4]}x{window[-1][5]:.1f}s praat_fail "
                f"{sum(w[6] for w in window)} wait {sum(w[7] for w in window) / n:.2f}s "
                f"workers {workers} chunks {chunks} eta {(args.steps - step) * rate / 60:.0f} min")
            window = []
        if step in args.save:
            path = os.path.join(args.out, f"ckpt_{step}.pt")
            save(path)
            log(f"saved {path}")
        elif args.save_last_every and step % args.save_last_every == 0:
            save(os.path.join(args.out, "ckpt_last.pt"))
    log(f"done: {step} steps in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
