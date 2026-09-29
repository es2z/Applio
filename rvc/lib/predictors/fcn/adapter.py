"""Offline adapters and diagnostics; state is local to each audio track."""

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rvc.lib.predictors.f0_methods import fcn_variant

from .decoder import FCNDecoder
from .model import ARCHITECTURES, FCNModel
from .preprocess import FCNPreprocessor, FCNTensorPreprocessor
from .profiles import normalization_name, resolve_profile

WEIGHT_DIR = Path(__file__).resolve().parents[3] / "models" / "predictors"
DEFAULT_WEIGHT = WEIGHT_DIR / "fcn-993.pt"


def default_weight(architecture):
    """rvc/models/predictors/<architecture>.pt, e.g. fcn-929.pt."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown FCN architecture: {architecture!r}")
    return WEIGHT_DIR / f"{architecture}.pt"


def grid_geometry(method):
    """(native frames per 10 ms, native hop in 16 kHz samples) of an FCN method.

    FCN-993 steps 8 samples at 8 kHz (1 ms, 10 frames per hop) and FCN-929 steps 4
    (0.5 ms, 20 frames per hop). Native frame i is centred on 16 kHz sample
    i * native hop, so 10 ms frame k is centred on native frame k * frames per hop.
    """
    stride = ARCHITECTURES[fcn_variant(method)[0]]["stride"]
    return 160 // (2 * stride), 2 * stride


@dataclass
class FCNTrack:
    frame_index: np.ndarray
    pitch_hz: np.ndarray
    voiced: np.ndarray
    confidence: np.ndarray

    @property
    def timestamps(self):
        return self.frame_index.astype(np.float64) * 0.01


def finite_median(values):
    """NumPy median semantics for truncated windows, including even counts."""
    count = torch.isfinite(values).sum(-1)
    ordered = values.nan_to_num(nan=float("inf")).sort(dim=-1).values
    lo = ((count - 1).clamp_min(0) // 2)[..., None]
    hi = (count // 2)[..., None]
    median = (
        ordered.gather(-1, lo).squeeze(-1) + ordered.gather(-1, hi).squeeze(-1)
    ) * 0.5
    return torch.where(count > 0, median, float("nan"))


def tensor_grid(cents, confidence, audio, centers, profile, previous=False):
    """Reduce native tensors to selected 10 ms centers; no waveform CPU copy."""
    import torch.nn.functional as F

    if not fcn_variant(profile.method)[1]:
        hz = torch.nan_to_num(10 * 2 ** (cents[centers] / 1200), nan=0).float()
        return hz, hz > 0, confidence[centers]
    frames_per_hop, native_hop = grid_geometry(profile.method)
    half = frames_per_hop // 2
    radius = profile.median_frames // 2
    smoothed = confidence
    if radius:
        smoothed = finite_median(
            F.pad(confidence, (radius, radius), value=float("nan")).unfold(
                0, 2 * radius + 1, 1
            )
        )
    valid = (
        (confidence > 0)
        & (confidence >= profile.exit_threshold)
        & (smoothed >= profile.exit_threshold)
        & torch.isfinite(cents)
    )
    pitch = cents
    if radius:
        supported = torch.where(valid, cents, float("nan"))
        pitch = finite_median(
            F.pad(supported, (radius, radius), value=float("nan")).unfold(
                0, 2 * radius + 1, 1
            )
        )
    indices = centers[:, None] + torch.arange(-half, half, device=cents.device)
    exists = (indices >= 0) & (indices < len(cents))
    indices = indices.clamp(0, len(cents) - 1)
    candidates = valid[indices] & exists
    frame_conf = finite_median(torch.where(exists, smoothed[indices], float("nan")))
    ordered, order = torch.where(candidates, pitch[indices], float("inf")).sort(
        dim=-1, stable=True
    )
    weights = torch.where(candidates, smoothed[indices], 0).gather(-1, order)
    selected = (
        (weights.cumsum(-1) >= weights.sum(-1, keepdim=True) * 0.5).long().argmax(-1)
    )
    selected_cents = ordered.gather(-1, selected[:, None]).squeeze(-1)
    samples = centers[:, None] * native_hop + torch.arange(
        -80, 80, device=cents.device
    )
    present = (samples >= 0) & (samples < len(audio))
    audible = ((audio[samples.clamp(0, len(audio) - 1)] != 0) & present).any(-1)
    supported = candidates.any(-1) & audible
    enter = supported & (frame_conf >= profile.enter_threshold)
    stay = supported & (frame_conf >= profile.exit_threshold)
    clock = torch.arange(len(centers), device=cents.device)
    last_enter = torch.where(enter, clock, -1 if previous else -2).cummax(0).values
    last_exit = torch.where(~stay, clock, -2 if previous else -1).cummax(0).values
    voiced = last_enter > last_exit
    hz = torch.where(voiced, 10 * 2 ** (selected_cents / 1200), 0).float()
    return hz, voiced, frame_conf


def weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    total = weights.sum()
    if len(values) == 0 or total <= 0:
        return np.nan
    index = np.searchsorted(np.cumsum(weights[order]), total / 2, side="left")
    return values[order[min(index, len(order) - 1)]]


class FCNRVCAdapter:
    def __init__(self, profile):
        self.profile = profile

    def __call__(self, cents, confidence, p_len, audio):
        profile = self.profile
        frames_per_hop, _ = grid_geometry(profile.method)
        half = frames_per_hop // 2
        smoothed = confidence.copy()
        radius = profile.median_frames // 2
        for i in range(len(confidence)):
            if radius:
                smoothed[i] = np.median(confidence[max(0, i - radius) : i + radius + 1])
        valid = (
            (confidence > 0)
            & (confidence >= profile.exit_threshold)
            & (smoothed >= profile.exit_threshold)
            & np.isfinite(cents)
        )
        pitches = cents.copy()
        if radius:
            for i in np.flatnonzero(valid):
                start, stop = max(0, i - radius), i + radius + 1
                pitches[i] = np.median(cents[start:stop][valid[start:stop]])
        f0 = np.zeros(p_len, np.float32)
        output_confidence = np.zeros(p_len, np.float32)
        voiced = np.zeros(p_len, bool)
        previous = False
        for k in range(p_len):
            start = max(0, k * frames_per_hop - half)
            stop = min(len(cents), k * frames_per_hop + half)
            output_confidence[k] = np.median(smoothed[start:stop])
            candidates = valid[start:stop]
            # Only exact digital silence is gated, with no arbitrary dB floor.
            silence = not np.any(
                audio[max(0, k * 160 - 80) : min(len(audio), k * 160 + 80)]
            )
            threshold = profile.exit_threshold if previous else profile.enter_threshold
            previous = bool(
                candidates.any() and output_confidence[k] >= threshold and not silence
            )
            voiced[k] = previous
            if previous:
                pitch = weighted_median(
                    pitches[start:stop][candidates], smoothed[start:stop][candidates]
                )
                f0[k] = 10 * 2 ** (pitch / 1200)
        return FCNTrack(np.arange(p_len, dtype=np.int64), f0, voiced, output_confidence)


class FCNPredictor:
    def __init__(
        self,
        device="cpu",
        method="fcn-993",
        profile=None,
        weight_path=None,
        block_frames=256,
    ):
        self.profile = resolve_profile(method, profile)
        self.architecture_id = fcn_variant(method)[0]
        self.architecture = ARCHITECTURES[self.architecture_id]
        self.frames_per_hop, self.native_hop = grid_geometry(method)
        self.device = torch.device(device)
        self.weight_path = Path(
            default_weight(self.architecture_id) if weight_path is None else weight_path
        )
        if not self.weight_path.is_file():
            upstream = self.architecture_id.replace("fcn-", "FCN_")
            raise FileNotFoundError(
                f"FCN weight missing: {self.weight_path}. Run tools/convert_fcn993.py --architecture {self.architecture_id} with original {upstream}/weights.h5 first."
            )
        manifest_path = self.weight_path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with self.weight_path.open("rb") as stream:
            self.weight_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        if manifest["weight_sha256"] != self.weight_sha256:
            raise ValueError("FCN weight checksum differs from conversion manifest")
        stat = self.weight_path.stat()
        self.asset_signature = (stat.st_size, stat.st_mtime_ns)
        checkpoint = torch.load(self.weight_path, map_location="cpu", weights_only=True)
        if checkpoint["metadata"]["architecture"] != self.architecture:
            raise ValueError(
                f"Unexpected FCN architecture in {self.weight_path}; expected {self.architecture_id}"
            )
        self.model = FCNModel(self.architecture_id).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        input_size = self.architecture["receptive_field"]
        self.preprocessor = FCNPreprocessor(
            "constant" if fcn_variant(method)[1] else "wrap", input_size
        )
        self.tensor_preprocessor = (
            FCNTensorPreprocessor(self.device, self.preprocessor.boundary, input_size)
            if self.device.type == "cuda"
            else None
        )
        self.decoder = FCNDecoder()
        if not isinstance(block_frames, int) or block_frames < 1:
            raise ValueError("block_frames must be a positive integer")
        self.block_frames = block_frames
        self.forward = self.model
        if self.profile.compile_model:
            from .compile import FCNForward

            self.forward = FCNForward(self.model, self.profile, self.device)

    @property
    def fingerprint(self):
        return self.profile.fingerprint(self.weight_sha256)

    def with_profile(self, method, profile=None):
        """Session-owned adapter sharing the same immutable network/weights."""
        architecture, is_rvc = fcn_variant(method)
        if architecture != self.architecture_id:
            raise ValueError(
                f"{method} needs the {architecture} network; this predictor holds {self.architecture_id}"
            )
        other = copy.copy(self)
        other.profile = resolve_profile(method, profile)
        other.preprocessor = FCNPreprocessor(
            "constant" if is_rvc else "wrap", self.architecture["receptive_field"]
        )
        if self.tensor_preprocessor is not None:
            other.tensor_preprocessor = copy.copy(self.tensor_preprocessor)
            other.tensor_preprocessor.boundary = other.preprocessor.boundary
        if other.profile != self.profile:
            other.forward = other.model
            if other.profile.compile_model:
                from .compile import FCNForward

                other.forward = FCNForward(other.model, other.profile, other.device)
        return other

    def compile_status(self):
        return (
            self.forward.status() if self.profile.compile_model else "FCN compile: OFF"
        )

    def metadata(self):
        return {
            "method": self.profile.method,
            "profile": self.profile.to_dict(),
            "fingerprint": self.fingerprint,
            "weight_sha256": self.weight_sha256,
            "architecture": self.architecture_id,
            "resampler": "resampy-0.4.3-kaiser_best-cuda-ordered-v1",
            "implementation": "fcn-cuda-v1-fp32-tf32-off",
            "normalization": normalization_name(self.profile.method),
            "decoder": "local-average-cents-9",
            "grid": {"sample_rate": 16000, "origin": 0, "hop": 160},
            "coarse": {
                "minimum": self.profile.coarse_min,
                "maximum": self.profile.coarse_max,
                "bins": 256,
            },
        }

    def native_tensor(self, audio, return_activation=False):
        if not torch.is_tensor(audio):
            audio = np.ascontiguousarray(audio, dtype=np.float32)
        if self.tensor_preprocessor is not None:
            with (
                torch.inference_mode(),
                torch.autocast(self.device.type, enabled=False),
                torch.backends.cudnn.flags(allow_tf32=False),
            ):
                normalized = self.tensor_preprocessor(
                    torch.as_tensor(audio, dtype=torch.float32, device=self.device)
                )
        else:
            normalized = self.preprocessor(audio)
        receptive_field = self.architecture["receptive_field"]
        stride = self.architecture["stride"]
        frames = max(0, (len(normalized) - receptive_field) // stride + 1)
        decoded, activations = [], []
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, enabled=False),
            torch.backends.cudnn.flags(allow_tf32=False),
        ):
            for start in range(0, frames, self.block_frames):
                stop = min(start + self.block_frames, frames)
                chunk = torch.as_tensor(
                    normalized[start * stride : (stop - 1) * stride + receptive_field],
                    device=self.device,
                )
                activation = self.forward(chunk[None, None])[0]
                decoded.append(self.decoder(activation))
                if return_activation:
                    activations.append(activation)
        arrays = tuple(
            torch.cat([block[i] for block in decoded])
            if decoded
            else torch.empty(0, device=self.device)
            for i in range(3)
        )
        if return_activation:
            return (
                *arrays,
                torch.cat(activations)
                if activations
                else torch.empty((0, 486), device=self.device),
            )
        return arrays

    def native(self, audio, return_activation=False):
        return tuple(
            value.cpu().numpy()
            for value in self.native_tensor(audio, return_activation)
        )

    def extract_track(self, audio, p_len=None):
        if self.device.type == "cuda":
            if not torch.is_tensor(audio):
                audio = np.ascontiguousarray(audio, dtype=np.float32)
            audio = torch.as_tensor(
                audio, device=self.device, dtype=torch.float32
            ).detach()
            if audio.ndim != 1 or not torch.isfinite(audio).all():
                raise ValueError("FCN requires finite mono 16 kHz audio")
            available = len(audio) // 160
            p_len = available if p_len is None else p_len
            if not isinstance(p_len, (int, np.integer)) or not 0 <= p_len <= available:
                raise ValueError(
                    "p_len exceeds the valid 10 ms grid; explicitly pad audio first"
                )
            if p_len:
                cents, _, confidence = self.native_tensor(audio)
                centers = torch.arange(p_len, device=self.device) * self.frames_per_hop
                hz, voiced, confidence = tensor_grid(
                    cents, confidence, audio, centers, self.profile
                )
                return FCNTrack(
                    np.arange(p_len, dtype=np.int64),
                    hz.cpu().numpy(),
                    voiced.cpu().numpy(),
                    confidence.cpu().numpy(),
                )
            return FCNTrack(
                np.empty(0, np.int64),
                np.empty(0, np.float32),
                np.empty(0, bool),
                np.empty(0, np.float32),
            )
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("FCN requires finite mono 16 kHz audio")
        available = len(audio) // 160
        p_len = available if p_len is None else p_len
        if not isinstance(p_len, (int, np.integer)) or not 0 <= p_len <= available:
            raise ValueError(
                "p_len exceeds the valid 10 ms grid; explicitly pad audio first"
            )
        if p_len == 0:
            return FCNTrack(
                np.empty(0, np.int64),
                np.empty(0, np.float32),
                np.empty(0, bool),
                np.empty(0, np.float32),
            )
        cents, hz, confidence = self.native(audio)
        if fcn_variant(self.profile.method)[1]:
            return FCNRVCAdapter(self.profile)(cents, confidence, p_len, audio)
        indices = np.arange(p_len) * self.frames_per_hop
        return FCNTrack(
            np.arange(p_len, dtype=np.int64),
            hz[indices],
            hz[indices] > 0,
            confidence[indices],
        )

    def get_f0(self, x, p_len=None):
        return self.extract_track(x, p_len).pitch_hz
