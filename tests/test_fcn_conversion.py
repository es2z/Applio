import json

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")

from rvc.lib.predictors.fcn.model import FCNModel
from tools.convert_fcn993 import convert, sha256


def write_hdf5(path):
    state = FCNModel().state_dict()
    with h5py.File(path, "w") as handle:
        for i in range(1, 8):
            layer = f"conv{i}" if i < 7 else "classifier"
            kernel = state[f"{layer}.weight"].numpy().transpose(2, 1, 0)[:, None]
            handle.create_dataset(f"{layer}/{layer}/kernel:0", data=kernel)
            handle.create_dataset(
                f"{layer}/{layer}/bias:0", data=state[f"{layer}.bias"].numpy()
            )
            if i < 7:
                for name, key in (
                    ("gamma", "weight"),
                    ("beta", "bias"),
                    ("moving_mean", "running_mean"),
                    ("moving_variance", "running_var"),
                ):
                    handle.create_dataset(
                        f"{layer}-BN/{layer}-BN/{name}:0",
                        data=state[f"bn{i}.{key}"].numpy(),
                    )
    return state


def test_roundtrip_parameters_and_manifest(tmp_path):
    source, output = tmp_path / "weights.h5", tmp_path / "converted.pt"
    expected = write_hdf5(source)
    manifest = convert(source, output)
    checkpoint = torch.load(output, weights_only=True)
    for key, value in expected.items():
        torch.testing.assert_close(checkpoint["state_dict"][key], value, rtol=0, atol=0)
    assert manifest["source_sha256"] == sha256(source)
    assert manifest["weight_sha256"] == sha256(output)
    assert json.loads(output.with_suffix(".manifest.json").read_text()) == manifest


@pytest.mark.parametrize(
    "corruption", ["missing", "unknown", "shape", "nan", "dtype", "variance"]
)
def test_reject_corrupt_hdf5(tmp_path, corruption):
    source, output = tmp_path / "weights.h5", tmp_path / "converted.pt"
    write_hdf5(source)
    name = "conv1/conv1/bias:0"
    with h5py.File(source, "a") as handle:
        if corruption == "missing":
            del handle[name]
        elif corruption == "unknown":
            handle.create_dataset("extra", data=np.zeros(1, np.float32))
        elif corruption in ("shape", "dtype"):
            del handle[name]
            handle.create_dataset(
                name,
                data=np.zeros(
                    255 if corruption == "shape" else 256,
                    np.float64 if corruption == "dtype" else np.float32,
                ),
            )
        elif corruption == "nan":
            handle[name][0] = np.nan
        else:
            handle["conv1-BN/conv1-BN/moving_variance:0"][0] = -1
    with pytest.raises(ValueError):
        convert(source, output)
    assert not output.exists()
