"""The coarse-pitch contract shared by training, offline conversion and realtime.

The generator's pitch embedding has 256 rows, indexed by a mel-scale bin of the F0
between COARSE_MIN and a model's coarse maximum. That maximum is chosen when features
are extracted (1000 Hz, 1100 Hz as in upstream RVC, or 1680 Hz), recorded in
model_info.json, the resume checkpoints and the exported .pth as "f0_coarse_max", and
read back by every conversion path, so a pitch lands on the same row it was trained on.
It also bounds the F0 search range of the predictors that take one (CREPE, SWIFT,
FCNF0++), as it always has during extraction.
"""

import dataclasses

import numpy as np

COARSE_MIN = 50.0
COARSE_MAXIMA = (750.0, 1000.0, 1100.0, 1680.0)
# FCN-993 / FCN-929 output nothing above 1000 Hz and, measured on harmonic tones, are
# accurate only to ~750 Hz, so these two ranges are offered for them alone; every other
# method tracks well past 1000 Hz and would lose pitches to either. A model extracted at
# one of them is still converted at it with any method - the restriction is on choosing
# it for an extraction.
FCN_ONLY_COARSE_MAXIMA = (750.0, 1000.0)
# What a model that recorded nothing was trained with: extraction has used 1680 Hz for
# every method since 2025-10-15, and the FCN profiles default to it.
DEFAULT_COARSE_MAX = 1680.0
# FCN's 486 output bins end at exactly 1000 Hz (and measured on harmonic tones it is
# only accurate to ~750 Hz), so bins above 1000 Hz can only ever be reached by a pitch
# shift at conversion time. 1000 Hz spends all 254 bins where FCN can put a pitch.
FCN_COARSE_MAX = 1000.0


def default_coarse_max(method):
    """The range a new extraction uses when none is chosen."""
    from rvc.lib.predictors.f0_methods import FCN_METHODS

    return FCN_COARSE_MAX if method in FCN_METHODS else DEFAULT_COARSE_MAX


def quantize_f0(f0, minimum=COARSE_MIN, maximum=DEFAULT_COARSE_MAX):
    low, high = 1127 * np.log1p(np.array([minimum, maximum]) / 700)
    mel = 1127 * np.log1p(np.asarray(f0) / 700)
    return np.rint(np.clip((mel - low) * 254 / (high - low) + 1, 1, 255)).astype(
        np.int64
    )


def validate_coarse_max(value):
    value = float(value)
    if value not in COARSE_MAXIMA:
        raise ValueError(
            f"F0 coarse maximum must be one of {', '.join(f'{v:g}' for v in COARSE_MAXIMA)} Hz, got {value:g}"
        )
    return value


def extraction_coarse_maxima(method):
    """The coarse ranges an extraction with this F0 method may choose."""
    from rvc.lib.predictors.f0_methods import FCN_METHODS

    if method in FCN_METHODS:
        return COARSE_MAXIMA
    return tuple(v for v in COARSE_MAXIMA if v not in FCN_ONLY_COARSE_MAXIMA)


def validate_extraction_coarse_max(value, method):
    value = validate_coarse_max(value)
    allowed = extraction_coarse_maxima(method)
    if value not in allowed:
        raise ValueError(
            f"A {value:g} Hz coarse range is only offered for FCN-993 / FCN-929; "
            f"{method} can use {', '.join(f'{v:g}' for v in allowed)} Hz"
        )
    return value


def recorded_coarse_max(record):
    """The coarse maximum a model_info.json or exported .pth was trained with.

    Prefers the explicit "f0_coarse_max"; a model extracted with an FCN or FCNF0++
    profile before that key existed recorded the same number in its f0_extraction
    spec; anything else predates both and was extracted at 1680 Hz.
    """
    record = record or {}
    value = record.get("f0_coarse_max")
    if value is None:
        value = ((record.get("f0_extraction") or {}).get("coarse") or {}).get("maximum")
    return validate_coarse_max(DEFAULT_COARSE_MAX if value is None else value)


def align_profile_coarse(profile, maximum):
    """An FCN / FCNF0++ profile whose coarse range is the model's.

    The profile's own coarse_max predates the per-model setting; the model's value is
    authoritative, so a profile that says otherwise is followed in everything but that.
    """
    maximum = validate_coarse_max(maximum)
    if profile.coarse_max == maximum:
        return profile
    print(
        f"F0 coarse range follows the model ({COARSE_MIN:g}-{maximum:g} Hz), "
        f"not the profile's {profile.coarse_max:g} Hz."
    )
    return dataclasses.replace(profile, coarse_min=COARSE_MIN, coarse_max=maximum)


def coarse_bin_centers(maximum):
    """The F0 in Hz each coarse bin 1..255 stands for under a range (index 0 unused)."""
    low, high = 1127 * np.log1p(np.array([COARSE_MIN, maximum]) / 700)
    mel = (np.arange(256) - 1) * (high - low) / 254 + low
    hz = 700 * np.expm1(mel / 1127)
    hz[0] = 0.0
    return hz


def remap_pitch_embedding(weight, source_max, target_max):
    """enc_p.emb_pitch.weight re-indexed from one coarse range to another.

    Row b of the result is the source row for the pitch bin b stands for under the
    target range, so every pitch keeps the embedding it was trained with. Pitches the
    source range never reached (above a 1000 Hz source, for a 1680 Hz target) share the
    source's top row, which is what the source would have quantized them to anyway.
    Row 0 is never indexed by the quantizer and is carried over as is.
    """
    import torch

    source_max, target_max = validate_coarse_max(source_max), validate_coarse_max(target_max)
    rows = quantize_f0(coarse_bin_centers(target_max)[1:], COARSE_MIN, source_max)
    index = np.concatenate(([0], rows))
    return weight[torch.as_tensor(index, device=weight.device)].clone()
