"""Pitch extraction transaction metadata, including failed/incomplete runs."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from rvc.lib.predictors.f0_methods import (
    FCN_METHODS,
    FCNF0PP_METHODS,
    HPA_RMVPE_METHODS,
    PROFILE_METHODS,
    hpa_rmvpe_variant,
)
from rvc.lib.predictors.f0_quantization import (
    DEFAULT_COARSE_MAX,
    align_profile_coarse,
    validate_coarse_max,
)


def fcnf0pp_extraction_spec(method, profile=None, coarse_max=DEFAULT_COARSE_MAX):
    from rvc.lib.predictors.fcnf0pp.profiles import resolve_profile, specification
    from rvc.lib.predictors.fcnf0pp.weights import weight_sha256

    profile = align_profile_coarse(resolve_profile(method, profile), coarse_max)
    weight_hash = weight_sha256()
    return {
        "method": method,
        "fingerprint": profile.fingerprint(weight_hash),
        **specification(profile, weight_hash),
        "coarse": {
            "minimum": profile.coarse_min,
            "maximum": profile.coarse_max,
            "bins": 256,
        },
    }


def extraction_spec(method, profile=None, coarse_max=DEFAULT_COARSE_MAX):
    """What this extraction depends on; any change here re-extracts every F0 file.

    The profile methods carry the coarse range inside their profile. For the others it
    is added only when it is not the default, so a spec recorded before the range was
    selectable, which was always extracted at 1680 Hz, still compares equal.
    """
    coarse_max = validate_coarse_max(coarse_max)
    coarse = {} if coarse_max == DEFAULT_COARSE_MAX else {"coarse_max": coarse_max}
    if method in FCNF0PP_METHODS:
        return fcnf0pp_extraction_spec(method, profile, coarse_max)
    if method in HPA_RMVPE_METHODS:
        from rvc.lib.predictors.hpa_rmvpe.weights import weight_sha256

        # Runs in the parent before any worker starts, so a first use downloads the
        # checkpoint exactly once here rather than once per GPU.
        return {
            "method": method,
            "weight_sha256": weight_sha256(hpa_rmvpe_variant(method)[0]),
            **coarse,
        }
    if method not in FCN_METHODS:
        return {"method": method, **coarse}
    from rvc.lib.predictors.f0_methods import fcn_variant
    from rvc.lib.predictors.fcn.adapter import default_weight
    from rvc.lib.predictors.fcn.profiles import normalization_name, resolve_profile

    profile = align_profile_coarse(resolve_profile(method, profile), coarse_max)
    architecture = fcn_variant(method)[0]
    weight = default_weight(architecture)
    if not weight.is_file():
        raise FileNotFoundError(
            f"FCN weight missing: {weight}; run tools/convert_fcn993.py --architecture {architecture} first"
        )
    manifest = json.loads(weight.with_suffix(".manifest.json").read_text())
    with weight.open("rb") as stream:
        weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if manifest["weight_sha256"] != weight_hash:
        raise ValueError("FCN weight checksum differs from conversion manifest")
    return {
        "method": method,
        "profile": profile.to_dict(),
        "fingerprint": profile.fingerprint(weight_hash),
        "weight_sha256": weight_hash,
        "architecture": architecture,
        "resampler": "resampy-0.4.3-kaiser_best-cuda-ordered-v1",
        "implementation": "fcn-cuda-v1-fp32-tf32-off",
        "normalization": normalization_name(method),
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
        # Unrecorded pitch files predate these methods, so they cannot be theirs; and
        # they were quantized at the default coarse range, so they are not reusable at
        # another one either.
        return (
            specification["method"] not in PROFILE_METHODS + HPA_RMVPE_METHODS
            and "coarse_max" not in specification
        )
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
            raise ValueError(f"Invalid F0 grid or coarse range: {source}")


def write_metadata(path, data):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, path)
