"""Versioned FCNF0++ settings and explicit per-run/checkpoint overrides."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rvc.lib.predictors.f0_quantization import COARSE_MAXIMA

PROFILE_DIR = Path(__file__).with_name("profiles")
DEFAULT_PATHS = {
    "fcnf0++": PROFILE_DIR / "baseline-v1.json",
    "fcnf0++-rvc": PROFILE_DIR / "rvc-default-v1.json",
    "fcnf0++-aligned": PROFILE_DIR / "baseline-aligned-v1.json",
    "fcnf0++-rvc-aligned": PROFILE_DIR / "rvc-aligned-v1.json",
}
# Methods that zero frames at or below periodicity_threshold.
GATED_METHODS = ("fcnf0++-rvc", "fcnf0++-rvc-aligned")
# Methods that place each window later to cancel the model's own lag on speech; the
# others are PENN's framing unchanged. See lag_compensation_ms and docs/fcnf0pp.md.
ALIGNED_METHODS = ("fcnf0++-aligned", "fcnf0++-rvc-aligned")
MAX_LAG_COMPENSATION_MS = 30.0
DECODERS = ("viterbi", "argmax")
# Running-median window of the -rvc methods, in FCNF0++'s own 10 ms frames: 3 is 30 ms.
# (FCN-993-RVC's median_frames counts its native 1 ms frames, so its 5 is 5 ms.)
MEDIAN_FRAMES = (0, 3, 5)
# "zero" puts frame i at t = i * 10 ms, the grid every other method here uses.
# "half-hop" (+5 ms) is what the earlier integration used; kept only to reproduce it.
CENTERS = ("zero", "half-hop")


@dataclass(frozen=True)
class FCNF0PPProfile:
    method: str = "fcnf0++"
    version: int = 1
    decoder: str = "viterbi"
    periodicity_threshold: float | None = None
    center: str = "zero"
    coarse_min: float = 50.0
    coarse_max: float = 1680.0
    calibrated: bool = False
    # How much later than t = i * 10 ms frame i's window is centred. FCNF0++ reports the
    # pitch of ~11 ms before its window centre on harmonic speech-range signals, so a
    # positive value moves the reported pitch back onto frame i's instant.
    lag_compensation_ms: float = 0.0
    # FCN-993-RVC's median, on FCNF0++'s 10 ms frames: a frame stays voiced only if both
    # its periodicity and the running median of periodicity are above the threshold,
    # and a voiced frame's pitch is the median (in cents) of the voiced frames around it.
    # 0 is off, which is what every profile recorded before this existed means.
    median_frames: int = 0

    def __post_init__(self):
        if self.method not in DEFAULT_PATHS or self.version != 1:
            raise ValueError("Unknown FCNF0++ method/profile version")
        if self.decoder not in DECODERS:
            raise ValueError(f"FCNF0++ decoder must be one of {DECODERS}")
        if self.center not in CENTERS:
            raise ValueError(f"FCNF0++ center must be one of {CENTERS}")
        if self.coarse_min != 50.0 or self.coarse_max not in COARSE_MAXIMA:
            raise ValueError("FCNF0++ F0 range must be 50–750, 50–1000, 50–1100 or 50–1680 Hz")
        if type(self.calibrated) is not bool:
            raise ValueError("calibrated must be true or false")
        if self.method in ALIGNED_METHODS:
            if not 0 < self.lag_compensation_ms <= MAX_LAG_COMPENSATION_MS:
                raise ValueError(
                    f"{self.method} needs 0 < lag_compensation_ms <= {MAX_LAG_COMPENSATION_MS:g}"
                )
            if self.center != "zero":
                raise ValueError("lag_compensation_ms is defined relative to center 'zero'")
        elif self.lag_compensation_ms != 0:
            raise ValueError(
                f"{self.method} keeps PENN's framing; use its -aligned method for lag compensation"
            )
        if type(self.median_frames) is not int or self.median_frames not in MEDIAN_FRAMES:
            raise ValueError(f"FCNF0++ median_frames must be one of {MEDIAN_FRAMES} (10 ms frames)")
        if self.median_frames and self.method not in GATED_METHODS:
            raise ValueError(
                f"{self.method} is PENN's pitch untouched; median_frames is for the -rvc methods"
            )
        if self.method not in GATED_METHODS:
            if self.periodicity_threshold is not None:
                raise ValueError(
                    f"{self.method} is ungated; use an -rvc method for a periodicity threshold"
                )
        else:
            if self.periodicity_threshold is None:
                raise ValueError(
                    f"An explicit {self.method} profile needs periodicity_threshold. "
                    "Leave the profile blank to use the bundled default."
                )
            if not 0 <= self.periodicity_threshold <= 1:
                raise ValueError("Expected 0 <= periodicity_threshold <= 1")

    def to_dict(self):
        return asdict(self)

    def fingerprint(self, weight_sha256):
        return hashlib.sha256(
            json.dumps(specification(self, weight_sha256), sort_keys=True).encode()
        ).hexdigest()


def recorded_profile(profile):
    """The profile as extraction records it. median_frames is left out while it is 0, so
    the specification and fingerprint of every run recorded before it existed still
    match and those pitch files are reused rather than re-extracted."""
    values = profile.to_dict()
    if not values["median_frames"]:
        del values["median_frames"]
    return values


def specification(profile, weight_sha256):
    """Everything that decides the F0 an extraction run writes to disk."""
    return {
        "profile": recorded_profile(profile),
        "weight_sha256": weight_sha256,
        "architecture": "penn-fcnf0",
        "implementation": "penn-1.0.0-preprocess-autocast-v1",
        "resampler": "torchaudio-Resample-16000-8000-default",
        "periodicity": "penn-entropy",
        "local_expected_value": 19,
        "interp_unvoiced_at": None,
        "grid": {"sample_rate": 16000, "origin": 0, "hop": 160},
        "coarse_bins": 256,
    }


def default_profile(method):
    path = DEFAULT_PATHS.get(method)
    if path is None:
        raise ValueError(f"Unknown FCNF0++ method: {method}")
    return FCNF0PPProfile(**json.loads(path.read_text(encoding="utf-8")))


def recommended_profile_json(method):
    """The UI displays exactly the same versioned values used by CLI/workers."""
    return json.dumps(default_profile(method).to_dict(), indent=2)


def resolve_profile(method, explicit=None, checkpoint=None):
    """Explicit JSON / JSON path / dict, else a matching checkpoint profile, else default."""
    value = explicit
    if isinstance(value, str) and not value.strip():
        value = None
    if value is None and checkpoint is not None and checkpoint.get("method") == method:
        value = checkpoint
    if isinstance(value, str) and value.lstrip().startswith("{"):
        value = json.loads(value)
    elif isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if value is None or value == {}:
        return default_profile(method)
    profile = value if isinstance(value, FCNF0PPProfile) else FCNF0PPProfile(**value)
    if profile.method != method:
        raise ValueError(
            f"F0 profile is for {profile.method}, but {method} is selected"
        )
    return profile
