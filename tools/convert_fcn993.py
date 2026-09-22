"""Convert the original FCN_993 weights.h5; never downloads model assets."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rvc.lib.predictors.fcn.model import ARCHITECTURE, FCNModel

SOURCE_COMMIT = "8a2b530af821319b6badca93c8a0ed1f14bfee3c"
CONVERTER_VERSION = 1


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def convert(source, output, source_commit=SOURCE_COMMIT):
    model = FCNModel()
    state = model.state_dict()
    expected = {}
    for i in range(1, 8):
        layer = f"conv{i}" if i < 7 else "classifier"
        for original, target in (("kernel", "weight"), ("bias", "bias")):
            expected[f"{layer}/{layer}/{original}:0"] = f"{layer}.{target}"
        if i < 7:
            for original, target in (
                ("gamma", "weight"),
                ("beta", "bias"),
                ("moving_mean", "running_mean"),
                ("moving_variance", "running_var"),
            ):
                expected[f"{layer}-BN/{layer}-BN/{original}:0"] = f"bn{i}.{target}"
    datasets = {}
    with h5py.File(source, "r") as handle:

        def collect(name, value):
            if isinstance(value, h5py.Dataset):
                datasets[name] = value[()]

        handle.visititems(collect)
    if set(datasets) != set(expected):
        raise ValueError(
            f"HDF5 parameter mismatch: missing={set(expected) - set(datasets)}, "
            f"unknown={set(datasets) - set(expected)}"
        )
    for name, key in expected.items():
        value = datasets[name]
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid dtype or nonfinite parameter: {name}")
        if name.endswith("kernel:0"):
            if value.ndim != 4 or value.shape[1] != 1:
                raise ValueError(f"Invalid convolution kernel: {name}: {value.shape}")
            value = value[:, 0].transpose(2, 1, 0).copy()
        if tuple(value.shape) != tuple(state[key].shape):
            raise ValueError(
                f"Invalid shape: {name}: {value.shape}, expected {state[key].shape}"
            )
        if key.endswith("running_var") and (value < 0).any():
            raise ValueError(f"Negative BN variance: {name}")
        state[key] = torch.from_numpy(value.copy())
    model.load_state_dict(state, strict=True)
    manifest = {
        "architecture": ARCHITECTURE,
        "source_commit": source_commit,
        "source_url": "https://github.com/ardaillon/FCN-f0",
        "source_sha256": sha256(source),
        "converter_version": CONVERTER_VERSION,
        "parameters": {name: list(value.shape) for name, value in datasets.items()},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "metadata": manifest}, output)
    manifest["weight_sha256"] = sha256(output)
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output, args.source_commit), indent=2))
