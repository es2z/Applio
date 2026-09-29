"""Versioned FCN defaults and explicit per-run/checkpoint overrides."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rvc.lib.predictors.f0_methods import FCN_METHODS, fcn_variant
from rvc.lib.predictors.f0_quantization import COARSE_MAXIMA

RVC_DEFAULT_PATH = Path(__file__).with_name("profiles") / "rvc-balanced-v1.json"
RVC_DEFAULT_PATHS = {
    "fcn-993-rvc": RVC_DEFAULT_PATH,
    "fcn-929-rvc": Path(__file__).with_name("profiles") / "rvc-929-balanced-v1.json",
}
# median_frames counts native frames: 1 ms for FCN-993, 0.5 ms for FCN-929, so the
# same odd widths in time are 3/5/9 and 5/9/17 respectively.
MEDIAN_FRAMES = {"fcn-993": (0, 3, 5, 9), "fcn-929": (0, 5, 9, 17)}


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
        if self.method not in FCN_METHODS or self.version != 1:
            raise ValueError("Unknown FCN method/profile version")
        if self.coarse_min != 50.0 or self.coarse_max not in COARSE_MAXIMA:
            raise ValueError("FCN coarse range must be 50–750, 50–1000, 50–1100 or 50–1680 Hz")
        architecture, is_rvc = fcn_variant(self.method)
        if not is_rvc:
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
                    f"An explicit {self.method.upper()} profile needs enter_threshold and exit_threshold. Leave the profile blank to use Balanced v1."
                )
            if not 0 <= self.exit_threshold <= self.enter_threshold <= 1:
                raise ValueError("Expected 0 <= exit_threshold <= enter_threshold <= 1")
            allowed = MEDIAN_FRAMES[architecture]
            if self.median_frames not in allowed:
                raise ValueError(
                    f"median_frames must be one of {', '.join(map(str, allowed))} for {architecture}"
                )

    def to_dict(self):
        return asdict(self)

    def fingerprint(self, weight_sha256):
        architecture = fcn_variant(self.method)[0]
        specification = {
            "profile": self.to_dict(),
            "weight_sha256": weight_sha256,
            "architecture": architecture,
            "resampler": "resampy-0.4.3-kaiser_best-cuda-ordered-v1",
            "implementation": "fcn-cuda-v1-fp32-tf32-off",
            "normalization": normalization_name(self.method),
            "grid": {"origin": 0, "sample_rate": 16000, "hop": 160},
            "decoder": "local-average-cents-9",
            "coarse_bins": 256,
        }
        return hashlib.sha256(
            json.dumps(specification, sort_keys=True).encode()
        ).hexdigest()


def normalization_name(method):
    """e.g. "994-population-wrap": sliding-norm window and edge extension."""
    from .model import normalization_window

    architecture, is_rvc = fcn_variant(method)
    boundary = "zero" if is_rvc else "wrap"
    return f"{normalization_window(architecture)}-population-{boundary}"


def default_profile(method):
    if method in RVC_DEFAULT_PATHS:
        path = RVC_DEFAULT_PATHS[method]
        return FCNProfile(**json.loads(path.read_text(encoding="utf-8")))
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
