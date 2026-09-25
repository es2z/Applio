"""The published HPA-RMVPE checkpoints, fetched on first use and checksum-pinned.

The HuggingFace files (AnhP/HPA-RMVPE) are 203 MB training checkpoints holding the
Adam moments and scheduler next to the model, which is 68 MB of it. Each variant is
downloaded once, its sha256 checked against the published LFS object id, stripped to
checkpoint["model"] and written with a manifest next to it; every later load checks the
weight against that manifest, the same scheme as fcnf0pp/weights.py.

The checkpoints also carry best_rpa as a numpy float64, which plain weights_only=True
refuses. Rather than falling back to an unrestricted unpickler, the three numpy globals
that scalar needs are allowlisted for that one load.
"""

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

PREDICTORS_DIR = Path(__file__).resolve().parents[3] / "models" / "predictors"
SOURCE_REPOSITORY = "AnhP/HPA-RMVPE"
SOURCE_REVISION = "fa65f12635ab1877c11a6087940bcc21a7a309c0"
CODE_REPOSITORY = "https://github.com/PhamHuynhAnh16/HPA-RMVPE"
CODE_REVISION = "0cdb7db22b381f0ee053bd540e8cdc90180443b7"
# Both variants are the same architecture: 581 tensors, 88 of them BatchNorm's
# num_batches_tracked counters.
PARAMETER_COUNT = 16_939_452

VARIANTS = {
    "76000": {
        "path_in_repo": "model_76000.pt",
        "source_sha256": "82bb44b31774e53b976002bf1a9facd9f7dee354e1738e370b7cb6e7109d0120",
        "source_size": 203_408_185,
    },
    "112000": {
        "path_in_repo": "exp/model_112000.pt",
        "source_sha256": "096812f4053b534086c5e4eec45b30772ebf17bc7f941b0d407607602e6f6e39",
        "source_size": 203_409_714,
    },
}


def _variant(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown HPA-RMVPE variant {variant!r}; expected one of {list(VARIANTS)}")
    return VARIANTS[variant]


def source_url(variant):
    path = _variant(variant)["path_in_repo"]
    return f"https://huggingface.co/{SOURCE_REPOSITORY}/resolve/{SOURCE_REVISION}/{path}"


def weight_path(variant):
    _variant(variant)
    return PREDICTORS_DIR / f"hpa-rmvpe-{variant}.pt"


def manifest_path(path):
    return Path(path).with_suffix(".manifest.json")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _numpy_scalar_globals():
    # numpy 2 moved numpy.core to numpy._core; 1.x has only a stub under that name.
    if int(np.__version__.split(".")[0]) >= 2:
        from numpy._core.multiarray import scalar
    else:
        from numpy.core.multiarray import scalar
    return [scalar, np.dtype, type(np.dtype(np.float64))]


def strip_checkpoint(variant, source, output=None):
    """Keep only checkpoint["model"] from a published checkpoint and write the manifest.

    Refuses any file whose sha256 is not the published one. The output replaces any
    previous file atomically, so concurrent callers cannot leave a torn weight behind.
    """
    spec = _variant(variant)
    source = Path(source)
    output = Path(output) if output is not None else weight_path(variant)
    source_sha256 = sha256(source)
    if source_sha256 != spec["source_sha256"]:
        raise ValueError(
            f"{source} is not the published HPA-RMVPE {variant} checkpoint "
            f"(sha256 {source_sha256}, expected {spec['source_sha256']})"
        )
    with torch.serialization.safe_globals(_numpy_scalar_globals()):
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = {key: value.contiguous() for key, value in checkpoint["model"].items()}
    count = sum(value.numel() for value in state.values())
    if count != PARAMETER_COUNT:
        raise ValueError(f"Unexpected HPA-RMVPE parameter count {count}")
    iteration = int(checkpoint.get("iteration", -1))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    # Saved through a buffer: given a path, torch.save names the archive after the
    # file, so the per-process temporary name would give every install its own hash,
    # and a re-download would then force a needless re-extraction of the pitch.
    buffer = io.BytesIO()
    torch.save({"model": state, "iteration": iteration}, buffer)
    temporary.write_bytes(buffer.getvalue())
    weight_sha256 = sha256(temporary)
    manifest = {
        "architecture": "hpa-rmvpe",
        "variant": variant,
        "code_repository": CODE_REPOSITORY,
        "code_revision": CODE_REVISION,
        "source_url": source_url(variant),
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_sha256": spec["source_sha256"],
        "iteration": iteration,
        "parameters": count,
        "weight_sha256": weight_sha256,
    }
    manifest_file = manifest_path(output)
    manifest_temporary = manifest_file.with_name(f"{manifest_file.name}.{os.getpid()}.tmp")
    manifest_temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Weight first: a manifest never points at a weight that is not there yet.
    os.replace(temporary, output)
    os.replace(manifest_temporary, manifest_file)
    return manifest


def is_installed(variant):
    path = weight_path(variant)
    return path.is_file() and manifest_path(path).is_file()


def download(variant, progress=True):
    """Fetch, verify and strip one variant, even if it is already installed."""
    import requests
    from tqdm import tqdm

    spec = _variant(variant)
    output = weight_path(variant)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = output.with_name(f"hpa-rmvpe-{variant}.source.{os.getpid()}.pt")
    url = source_url(variant)
    print(f"[HPA-RMVPE] Downloading the {variant} checkpoint ({spec['source_size'] / 1e6:.0f} MB) from {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(source, "wb") as file, tqdm(
                total=spec["source_size"], unit="iB", unit_scale=True,
                desc=f"HPA-RMVPE {variant}", disable=not progress,
            ) as bar:
                for data in response.iter_content(1 << 20):
                    file.write(data)
                    bar.update(len(data))
        size = source.stat().st_size
        if size != spec["source_size"]:
            # Checked before hashing so a dropped connection says what happened.
            raise IOError(
                f"HPA-RMVPE {variant} download is incomplete ({size} of "
                f"{spec['source_size']} bytes); try again"
            )
        return strip_checkpoint(variant, source, output)
    finally:
        if source.exists():
            source.unlink()


def ensure_weight(variant):
    """Return the stripped weight's path, downloading it the first time it is needed."""
    if not is_installed(variant):
        download(variant)
    return weight_path(variant)


def _verified_manifest(variant):
    path = weight_path(variant)
    manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    digest = sha256(path)
    if manifest["weight_sha256"] != digest:
        raise ValueError(
            f"HPA-RMVPE {variant} weight checksum differs from its manifest; delete "
            f"{path} and its manifest to download it again"
        )
    return manifest, digest


def weight_sha256(variant):
    """The manifest-verified hash, fetching the weight first if needed."""
    ensure_weight(variant)
    return _verified_manifest(variant)[1]


def load_state_dict(variant):
    """Return (state_dict, weight_sha256), fetching the weight first if needed."""
    path = ensure_weight(variant)
    _, digest = _verified_manifest(variant)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["model"], digest
