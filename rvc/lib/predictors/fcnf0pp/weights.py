"""The official FCNF0++ checkpoint, stripped to its weights and checksum-pinned."""

import hashlib
import json
import os
from pathlib import Path

import torch

DEFAULT_WEIGHT = (
    Path(__file__).resolve().parents[3] / "models" / "predictors" / "fcnf0++.pt"
)
SOURCE_URL = (
    "https://huggingface.co/maxrmorrison/fcnf0-plus-plus/resolve/main/fcnf0%2B%2B.pt"
)
SOURCE_REPOSITORY = "maxrmorrison/fcnf0-plus-plus"
SOURCE_REVISION = "74911e26f43ad38790a42592e77f9d8be0a5dd1c"
# The file as published on HuggingFace: a training checkpoint holding the Adam
# moments next to the model, three times the size of the weights.
SOURCE_SHA256 = "28d89add04722461f831249a338b74516a96dedd3301ed794c18953d51d1b960"
PARAMETER_COUNT = 8_934_624


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest_path(weight_path):
    return Path(weight_path).with_suffix(".manifest.json")


def strip_checkpoint(source, output=DEFAULT_WEIGHT):
    """Keep only checkpoint["model"] and write the manifest next to it.

    The output keeps penn's {"model": state_dict} layout, so penn.infer(checkpoint=...)
    still loads it, which is what the parity tests compare against.
    """
    source, output = Path(source), Path(output)
    source_sha256 = sha256(source)
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"{source} is not the published FCNF0++ checkpoint "
            f"(sha256 {source_sha256}, expected {SOURCE_SHA256})"
        )
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = {key: value.contiguous() for key, value in checkpoint["model"].items()}
    count = sum(value.numel() for value in state.values())
    if count != PARAMETER_COUNT:
        raise ValueError(f"Unexpected FCNF0++ parameter count {count}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pt.tmp")
    torch.save(
        {"model": state, "step": int(checkpoint.get("step", -1))}, temporary
    )
    os.replace(temporary, output)
    manifest = {
        "architecture": "penn-fcnf0",
        "source_url": SOURCE_URL,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "step": int(checkpoint.get("step", -1)),
        "parameters": count,
        "weight_sha256": sha256(output),
    }
    manifest_path(output).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_state_dict(weight_path=DEFAULT_WEIGHT):
    """Return (state_dict, weight_sha256) after checking the manifest."""
    weight_path = Path(weight_path)
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"FCNF0++ weight missing: {weight_path}. Download {SOURCE_URL} and run "
            "tools/strip_fcnf0pp.py, or let the prerequisites download fetch it."
        )
    manifest_file = manifest_path(weight_path)
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"FCNF0++ manifest missing: {manifest_file}. Run tools/strip_fcnf0pp.py."
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    weight_sha256 = sha256(weight_path)
    if manifest["weight_sha256"] != weight_sha256:
        raise ValueError("FCNF0++ weight checksum differs from its manifest")
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    return checkpoint["model"], weight_sha256


def weight_sha256(weight_path=DEFAULT_WEIGHT):
    """The manifest-verified hash, without loading the tensors."""
    weight_path = Path(weight_path)
    manifest = json.loads(manifest_path(weight_path).read_text(encoding="utf-8"))
    digest = sha256(weight_path)
    if manifest["weight_sha256"] != digest:
        raise ValueError("FCNF0++ weight checksum differs from its manifest")
    return digest
