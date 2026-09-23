"""FCNF0++ through penn's own framing, network, decoders and periodicity.

Nothing here re-implements penn. What differs from penn.from_audio is only where
things are cached:
- the 16 -> 8 kHz torchaudio Resample is built once (penn builds one per call);
- the model is loaded once, with weights_only=True (penn.infer caches it in a
  function attribute and loads without weights_only);
- each decoder is held by the predictor, with its transition matrix already on the
  device (penn.postprocess keeps one global decoder and rebuilds it on every switch).

Pitch is never interpolated (interp_unvoiced_at is always None). fcnf0++ returns
penn's pitch untouched; fcnf0++-rvc additionally zeroes the frames penn itself
would call unvoiced at the profile's periodicity threshold (periodicity > threshold
is voiced, as in penn.voicing.threshold).

The -aligned methods are the one departure from penn: each window is centred
lag_compensation_ms later than t = i * 10 ms. On harmonic speech-range signals the
model reports the pitch of ~11 ms before its window centre, so this puts that pitch
back on frame i. It is a framing change only - the network, decoders and periodicity
are penn's - and on high voices (>300 Hz), which show no such lag, it makes the
output that much early instead.
"""

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .profiles import resolve_profile
from .weights import DEFAULT_WEIGHT, load_state_dict

SAMPLE_RATE = 16000
HOP = 160
HOP_SECONDS = 0.01


def _penn():
    import penn

    return penn


@dataclass
class FCNF0PPTrack:
    frame_index: np.ndarray
    pitch_hz: np.ndarray  # after the profile's voicing, 0 = unvoiced
    raw_pitch_hz: np.ndarray  # penn's pitch before any voicing
    periodicity: np.ndarray
    voiced: np.ndarray

    @property
    def timestamps(self):
        return self.frame_index.astype(np.float64) * HOP_SECONDS


class FCNF0PPPredictor:
    def __init__(
        self,
        device="cpu",
        method="fcnf0++",
        profile=None,
        weight_path=DEFAULT_WEIGHT,
        batch_size=2048,
    ):
        import torchaudio

        penn = _penn()
        self.penn = penn
        self.method = method
        self.profile = resolve_profile(method, profile)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.weight_path = Path(weight_path)
        state, self.weight_sha256 = load_state_dict(self.weight_path)
        stat = self.weight_path.stat()
        self.asset_signature = (stat.st_size, stat.st_mtime_ns)

        self.model = penn.Model()
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.resampler = torchaudio.transforms.Resample(
            SAMPLE_RATE, penn.SAMPLE_RATE
        ).to(self.device)
        self.batch_size = batch_size
        self._decoders = {}

    def with_profile(self, method, profile=None):
        """A predictor sharing this network and these decoders under another profile."""
        other = copy.copy(self)
        other.method = method
        other.profile = resolve_profile(method, profile)
        return other

    def decoder(self, name):
        decoder = self._decoders.get(name)
        if decoder is None:
            if name == "viterbi":
                decoder = self.penn.decode.Viterbi()
                # functools.cached_property: store the device copies on the instance, so
                # torbi does not copy the 1440 x 1440 matrix to the GPU on every call.
                decoder.__dict__["transition"] = decoder.transition.to(self.device)
                decoder.__dict__["initial"] = decoder.initial.to(self.device)
            elif name == "argmax":
                decoder = self.penn.decode.Argmax()
            else:
                raise ValueError(f"Unknown FCNF0++ decoder: {name}")
            self._decoders[name] = decoder
        return decoder

    def _audio(self, x):
        audio = x if torch.is_tensor(x) else torch.from_numpy(np.asarray(x))
        audio = audio.detach().to(self.device, torch.float32)
        if audio.ndim != 1:
            raise ValueError("FCNF0++ expects 1-D mono audio at 16 kHz")
        if not torch.isfinite(audio).all():
            raise ValueError("FCNF0++ input contains NaN or Inf")
        return audio

    @torch.inference_mode()
    def logits(self, audio):
        """Model logits for every penn frame, shape (frames, PITCH_BINS, 1)."""
        penn = self.penn
        audio8k = self.resampler(audio[None])
        center = self.profile.center
        # Samples at 8 kHz that each window is moved later by (the -aligned methods).
        lag = round(self.profile.lag_compensation_ms * penn.SAMPLE_RATE / 1000)
        if lag:
            # penn's "zero" framing reflect-pads WINDOW_SIZE // 2 on both sides; moving
            # that padding lag samples from the front to the back centres window i at
            # i * hop + lag. Frame count and everything downstream are unchanged, and
            # "half-window" is penn's own no-padding framing of the padded signal.
            half = penn.WINDOW_SIZE // 2
            padding = half + lag
            left, right = half - lag, half + lag
            center = "half-window"
        else:
            padding = (
                penn.WINDOW_SIZE // 2
                if center == "zero"
                else (penn.WINDOW_SIZE - penn.HOPSIZE) // 2
            )
        if audio8k.shape[-1] <= padding:
            raise ValueError(
                f"FCNF0++ needs more than {2 * padding} samples at 16 kHz "
                f"({padding / penn.SAMPLE_RATE * 1000:.0f} ms) for its reflect padding"
            )
        if lag:
            audio8k = torch.nn.functional.pad(audio8k, (left, right), mode="reflect")
        chunks = []
        for frames in penn.preprocess(
            audio8k,
            penn.SAMPLE_RATE,
            HOP_SECONDS,
            self.batch_size,
            center,
        ):
            with penn.core.inference_context(self.model):
                chunks.append(self.model(frames.to(self.device)).detach())
        return torch.cat(chunks, 0)

    @torch.inference_mode()
    def postprocess(self, logits):
        """penn.core.postprocess with this predictor's decoder and F0 range."""
        penn = self.penn
        minidx = penn.convert.frequency_to_bins(torch.tensor(self.profile.coarse_min))
        maxidx = penn.convert.frequency_to_bins(
            torch.tensor(self.profile.coarse_max), torch.ceil
        )
        logits[:, :minidx] = -float("inf")
        logits[:, maxidx:] = -float("inf")
        _, pitch = self.decoder(self.profile.decoder)(logits)
        periodicity = penn.periodicity.entropy(logits)
        return pitch.T, periodicity.T

    def extract_track(self, x, p_len=None):
        audio = self._audio(x)
        available = len(audio) // HOP
        p_len = available if p_len is None else int(p_len)
        if p_len < 0 or p_len > available:
            raise ValueError(
                f"p_len {p_len} is outside the 10 ms grid of this audio ({available} frames)"
            )
        pitch, periodicity = self.postprocess(self.logits(audio))
        # "zero" gives available + 1 frames and "half-hop" exactly available.
        raw = pitch[0, :p_len].float().cpu().numpy()
        periodicity = periodicity[0, :p_len].float().cpu().numpy()
        threshold = self.profile.periodicity_threshold
        if threshold is None:
            voiced = np.ones(p_len, dtype=bool)
        else:
            voiced = periodicity > threshold
        return FCNF0PPTrack(
            frame_index=np.arange(p_len, dtype=np.int64),
            pitch_hz=np.where(voiced, raw, 0.0).astype(np.float32),
            raw_pitch_hz=raw,
            periodicity=periodicity,
            voiced=voiced,
        )

    def get_f0(self, x, p_len=None):
        return self.extract_track(x, p_len).pitch_hz
