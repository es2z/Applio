"""Turn a SPIN checkpoint into an embedder folder that load_embedding can read.

    env314\\Scripts\\python.exe -X utf8 tools\\spin\\export.py logs\\_spin\\run1\\ckpt_5000.pt <out dir>

The folder holds the truncated model (config.json with num_hidden_layers set to the cut,
and its weights) plus the source embedder's preprocessor_config.json, so do_normalize is
read exactly as for the source. The SPIN head and codebook are not exported: they only
exist to train the layers.
"""

import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from tools.spin.model import load_truncated  # noqa: E402


def load_spin_checkpoint(path):
    """The truncated embedder with a checkpoint's trained layers loaded into it."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = load_truncated(ckpt["model"], ckpt["layers"])
    missing = [k for k in ckpt["state"] if k not in model.state_dict()]
    if missing:
        raise KeyError(f"checkpoint tensors with no place in the model: {missing[:5]}")
    model.load_state_dict(ckpt["state"], strict=False)
    return model, ckpt


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    args = parser.parse_args()
    if os.path.exists(args.out) and os.listdir(args.out):
        sys.exit(f"{args.out} is not empty")
    model, ckpt = load_spin_checkpoint(args.checkpoint)
    model.save_pretrained(args.out)
    shutil.copyfile(os.path.join(ckpt["model"], "preprocessor_config.json"),
                    os.path.join(args.out, "preprocessor_config.json"))
    with open(os.path.join(args.out, "SPIN.txt"), "w", encoding="utf-8") as f:
        f.write(f"SPIN fine-tune of {ckpt['model']}\n"
                f"cut to {ckpt['layers']} layers, top {ckpt['train_layers']} trained\n"
                f"checkpoint {os.path.abspath(args.checkpoint)} (step {ckpt['step']})\n")
    print(f"exported step {ckpt['step']} to {args.out}")


if __name__ == "__main__":
    main()
