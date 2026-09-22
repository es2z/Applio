"""Pitch extraction transaction metadata, including failed/incomplete runs."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from rvc.lib.predictors.f0_methods import FCN_METHODS


def extraction_spec(method, profile=None):
    if method not in FCN_METHODS:
        return {"method": method}
    from rvc.lib.predictors.fcn.adapter import DEFAULT_WEIGHT
    from rvc.lib.predictors.fcn.profiles import resolve_profile

    profile = resolve_profile(method, profile)
    if not DEFAULT_WEIGHT.is_file():
        raise FileNotFoundError(
            f"FCN weight missing: {DEFAULT_WEIGHT}; run tools/convert_fcn993.py first"
        )
    manifest = json.loads(DEFAULT_WEIGHT.with_suffix(".manifest.json").read_text())
    with DEFAULT_WEIGHT.open("rb") as stream:
        weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if manifest["weight_sha256"] != weight_hash:
        raise ValueError("FCN weight checksum differs from conversion manifest")
    return {
        "method": method,
        "profile": profile.to_dict(),
        "fingerprint": profile.fingerprint(weight_hash),
        "weight_sha256": weight_hash,
        "architecture": "fcn-993",
        "resampler": "resampy-0.4.3-kaiser_best-cuda-ordered-v1",
        "implementation": "fcn-cuda-v1-fp32-tf32-off",
        "normalization": "994-population-wrap"
        if method == "fcn-993"
        else "994-population-zero",
        "decoder": "local-average-cents-9",
        "grid": {"sample_rate": 16000, "origin": 0, "hop": 160},
        "coarse": {
            "minimum": profile.coarse_min,
            "maximum": profile.coarse_max,
            "bins": 256,
        },
    }


def input_signature(files):
    digest = hashlib.sha256()
    for source, *_ in sorted(files):
        path = Path(source)
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def can_reuse(previous, specification, signature):
    if not previous:
        return specification["method"] not in FCN_METHODS
    return bool(
        previous.get("complete")
        and previous.get("specification") == specification
        and previous.get("input_signature") == signature
    )


def validate_pitch_files(files):
    import soundfile as sf

    for source, coarse_path, hz_path, _ in files:
        expected = sf.info(source).frames // 160
        coarse, hz = (
            np.load(coarse_path, allow_pickle=False),
            np.load(hz_path, allow_pickle=False),
        )
        if (
            coarse.shape != hz.shape
            or hz.ndim != 1
            or not np.isfinite(hz).all()
            or not np.isfinite(coarse).all()
        ):
            raise ValueError(f"Invalid F0 pair: {source}")
        if (
            len(hz) != expected
            or (hz < 0).any()
            or (coarse < 1).any()
            or (coarse > 255).any()
        ):
            raise ValueError(f"Invalid FCN grid or coarse range: {source}")


def write_metadata(path, data):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, path)
