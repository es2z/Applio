"""Finite-context CUDA streaming with absolute 16 kHz sample positions.

Baseline's noncausal wrap at the true start is replaced by zero extension for
the first frame only. Internal chunk boundaries always retain real context.
"""

import copy
from dataclasses import dataclass

import torch

from .adapter import tensor_grid


@dataclass
class FCNStreamTrack:
    frame_index: torch.Tensor
    pitch_hz: torch.Tensor
    voiced: torch.Tensor
    confidence: torch.Tensor

    @property
    def timestamps(self):
        return self.frame_index.double() * 0.01


class FCNStream:
    def __init__(self, predictor):
        if predictor.device.type != "cuda":
            raise ValueError("FCNStream requires CUDA")
        self.predictor = copy.copy(predictor)
        self.predictor.tensor_preprocessor = copy.copy(predictor.tensor_preprocessor)
        self.predictor.tensor_preprocessor.boundary = "constant"
        radius = predictor.profile.median_frames // 2
        temporal = (
            (4 + 2 * radius) * 16 if predictor.profile.method == "fcn-993-rvc" else 0
        )
        # Original raw support + FIR wing + temporal support. Round to the RVC
        # grid and reserve 16 samples for capture-resampler lookahead.
        self.holdback_samples = ((1986 + 99 + temporal + 16 + 159) // 160) * 160
        self.context_samples = self.holdback_samples + 160
        self.reset()

    def reset(self):
        self.buffer = torch.empty(0, device=self.predictor.device)
        self.buffer_start = 0
        self.sample_count = 0
        self.next_frame = 0
        self.previous_voiced = False
        self.closed = False

    def _empty(self):
        return FCNStreamTrack(
            torch.empty(0, dtype=torch.int64, device=self.predictor.device),
            self.buffer[:0],
            torch.empty(0, dtype=torch.bool, device=self.predictor.device),
            self.buffer[:0],
        )

    @torch.inference_mode()
    def push(self, audio, final=False):
        if self.closed:
            raise RuntimeError("FCN stream is flushed; reset before feeding more audio")
        audio = torch.as_tensor(
            audio, dtype=torch.float32, device=self.predictor.device
        )
        if audio.ndim != 1 or not torch.isfinite(audio).all():
            raise ValueError("FCN stream requires finite mono 16 kHz samples")
        self.buffer = torch.cat((self.buffer, audio))
        self.sample_count += len(audio)
        stop = (
            self.sample_count // 160
            if final
            else max(0, (self.sample_count - self.holdback_samples) // 160)
        )
        if stop <= self.next_frame:
            self.closed = final
            return self._empty()
        indices = torch.arange(self.next_frame, stop, device=self.predictor.device)
        centers = (indices * 160 - self.buffer_start) // 16
        cents, _, confidence = self.predictor.native_tensor(self.buffer)
        hz, voiced, confidence = tensor_grid(
            cents,
            confidence,
            self.buffer,
            centers,
            self.predictor.profile,
            self.previous_voiced,
        )
        self.previous_voiced = bool(voiced[-1].item())
        self.next_frame = stop
        retain_from = max(0, stop * 160 - self.context_samples)
        drop = retain_from - self.buffer_start
        if drop > 0:
            self.buffer = self.buffer[drop:].clone()
            self.buffer_start = retain_from
        self.closed = final
        return FCNStreamTrack(indices, hz, voiced, confidence)

    def flush(self):
        return self.push(self.buffer[:0], final=True)
