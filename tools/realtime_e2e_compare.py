"""Check that a realtime change keeps the default path bit-identical - no GPU, no weights.

Runs the realtime VoiceChanger end to end on the CPU with a small randomly initialised
APEX-GAN model, a fake embedder (one fixed convolution) and a fake F0 predictor (voiced
where the input is loud), and saves the output. Run it once against a checkout of the
commit to compare with and once against the working tree, then compare the two files:

    git worktree add ../applio-base 5ad003c
    env\\python.exe tools\\realtime_e2e_compare.py run --repo ..\\applio-base --out base.npy
    env\\python.exe tools\\realtime_e2e_compare.py run --repo . --out new.npy
    env\\python.exe tools\\realtime_e2e_compare.py compare base.npy new.npy
    git worktree remove ../applio-base

`--options '{"output_position": "trim"}'` runs with realtime options instead (only on a
checkout that has rvc/realtime/options.py). `PROTECT=0.33` in the environment sets the
protect value, so the Protect paths run too. The fakes are deterministic but not
time-invariant (the fake F0 depends on the frame index inside the window), so only the
default path is expected to match bit for bit; option runs are smoke tests.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def run(repo, out_path, options_json):
    sys.path.insert(0, os.path.abspath(repo))
    import torch

    from rvc.realtime import core
    from rvc.realtime.pipeline import RealtimeVoiceConverter, Realtime_Pipeline
    from tests.test_apex_gan import build_synthesizer

    class FakeHubert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(5)
            self.conv = torch.nn.Conv1d(1, 768, 400, stride=320)

        def forward(self, feats, **kwargs):
            return {"last_hidden_state": self.conv(feats.unsqueeze(1)).transpose(1, 2)}

    class FakeF0:
        def get_f0(self, x, filter_radius=0.03, **kwargs):
            frames = x.shape[0] // 160 + 1
            level = np.array(
                [np.abs(x[i * 160:(i + 1) * 160]).mean() if i * 160 < len(x) else 0 for i in range(frames)]
            )
            f0 = np.where(level > 0.02, 180.0 + 20 * np.sin(np.arange(frames) / 7.0), 0.0)
            return f0.astype(np.float32)

    options = json.loads(options_json) if options_json else None
    torch.manual_seed(0)
    net_g = build_synthesizer("APEX-GAN").eval()
    vc = RealtimeVoiceConverter.__new__(RealtimeVoiceConverter)
    vc.config = SimpleNamespace(device="cpu")
    vc.tgt_sr = 48000
    vc.net_g = net_g
    vc.cpt = {}
    vc.version = "v2"
    vc.use_f0 = 1
    vc.synthesis = None
    args = (vc, FakeHubert().eval(), None, None, "rmvpe-filled", 0)
    kwargs = {}
    if options is not None:
        from rvc.realtime.options import RealtimeOptions

        kwargs["options"] = RealtimeOptions.from_dict(options)
        pipeline = Realtime_Pipeline(*args, options=kwargs["options"])
    else:
        pipeline = Realtime_Pipeline(*args)
    pipeline.f0_model = FakeF0()
    with patch.object(core, "create_pipeline", return_value=pipeline):
        changer = core.VoiceChanger(90, 0.08, 0.5, silent_threshold=0, **kwargs)

    block = changer.block_frame
    t = np.arange(block * 10) / 48000
    envelope = (np.sin(2 * np.pi * 1.3 * t) > -0.2).astype(np.float32)
    x = (0.1 * envelope * (np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t))).astype(np.float32)
    protect = float(os.environ.get("PROTECT", "0.5"))
    torch.manual_seed(123)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), block):
            outs.append(changer.on_request(x[i:i + block], protect=protect)[0])
    y = np.concatenate(outs)
    np.save(out_path, y)
    print("saved", out_path, y.shape, "peak", float(np.abs(y).max()), "rms", float(np.sqrt(np.mean(y ** 2))))


def compare(a, b):
    x, y = np.load(a), np.load(b)
    same = x.shape == y.shape and np.array_equal(x, y)
    diff = float(np.abs(x - y).max()) if x.shape == y.shape else float("nan")
    print("bit-identical" if same else f"DIFFERENT (max abs diff {diff})")
    return 0 if same else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", default=".")
    r.add_argument("--out", required=True)
    r.add_argument("--options")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = parser.parse_args()
    if args.cmd == "run":
        run(args.repo, args.out, args.options)
        return 0
    return compare(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
