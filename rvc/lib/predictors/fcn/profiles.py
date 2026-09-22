"""Versioned FCN defaults and explicit per-run/checkpoint overrides."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

RVC_DEFAULT_PATH = Path(__file__).with_name("profiles") / "rvc-balanced-v1.json"


@dataclass(frozen=True)
class FCNProfile:
    method: str = "fcn-993"
    version: int = 1
    enter_threshold: float | None = None
    exit_threshold: float | None = None
    median_frames: int = 0
    coarse_min: float = 50.0
    coarse_max: float = 1680.0
    calibrated: bool = False
    compile_model: bool = False
    compile_mode: str = "default"

    def __post_init__(self):
        if type(self.compile_model) is not bool or self.compile_mode not in (
            "default",
            "reduce-overhead",
            "max-autotune",
        ):
            raise ValueError("Invalid FCN compile settings")
        if self.method not in ("fcn-993", "fcn-993-rvc") or self.version != 1:
            raise ValueError("Unknown FCN method/profile version")
        if (self.coarse_min, self.coarse_max) not in ((50.0, 1680.0), (50.0, 1100.0)):
            raise ValueError("FCN coarse range must be 50–1680 or 50–1100 Hz")
        if self.method == "fcn-993":
            if (
                self.enter_threshold is not None
                or self.exit_threshold is not None
                or self.median_frames != 0
            ):
                raise ValueError(
                    "Baseline does not allow voicing thresholds or filtering"
                )
        else:
            if self.enter_threshold is None or self.exit_threshold is None:
                raise ValueError(
                    "An explicit FCN-993-RVC profile needs enter_threshold and exit_threshold. Leave the profile blank to use Balanced v1."
                )
            if not 0 <= self.exit_threshold <= self.enter_threshold <= 1:
                raise ValueError("Expected 0 <= exit_threshold <= enter_threshold <= 1")
            if self.median_frames not in (0, 3, 5, 9):
                raise ValueError("median_frames must be 0, 3, 5, or 9")

    def to_dict(self):
        return asdict(self)

    def fingerprint(self, weight_sha256):
        specification = {
            "profile": self.to_dict(),
            "weight_sha256": weight_sha256,
            "architecture": "fcn-993",
            "resampler": "resampy-0.4.3-kaiser_best-cuda-ordered-v1",
            "implementation": "fcn-cuda-v1-fp32-tf32-off",
            "normalization": "994-population-wrap"
            if self.method == "fcn-993"
            else "994-population-zero",
            "grid": {"origin": 0, "sample_rate": 16000, "hop": 160},
            "decoder": "local-average-cents-9",
            "coarse_bins": 256,
        }
        return hashlib.sha256(
            json.dumps(specification, sort_keys=True).encode()
        ).hexdigest()


def default_profile(method):
    if method == "fcn-993-rvc":
        return FCNProfile(**json.loads(RVC_DEFAULT_PATH.read_text(encoding="utf-8")))
    return FCNProfile(method=method)


def recommended_profile_json(method):
    """The UI displays exactly the same versioned values used by CLI/workers."""
    return json.dumps(default_profile(method).to_dict(), indent=2)


def resolve_profile(method, explicit=None, checkpoint=None):
    value = explicit
    if value is None and checkpoint is not None and checkpoint.get("method") == method:
        value = checkpoint
    if isinstance(value, str) and not value.strip():
        value = None
        if checkpoint is not None and checkpoint.get("method") == method:
            value = checkpoint
    if isinstance(value, str) and value.lstrip().startswith("{"):
        value = json.loads(value)
    elif isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if value is None or value == {}:
        return default_profile(method)
    profile = value if isinstance(value, FCNProfile) else FCNProfile(**value)
    if profile.method != method:
        raise ValueError("FCN profile method does not match selected method")
    return profile
