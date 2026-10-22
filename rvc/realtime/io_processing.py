"""Input / output processing around the realtime conversion (all opt-in except the
level readout, which only measures).

- InputHighpass: the training preprocess's "Noise filter" (5th order Butterworth
  high-pass at 48 Hz, rvc/train/preprocess/preprocess.py) applied to the live input as
  one continuous stream. It matches training only if the dataset was preprocessed with
  that filter on.
- gate_fade_out: a short fade into silence when the silence gate stops the output,
  instead of the hard cut from the last converted sample to zero.
- LevelMeter: the generator's own peak (before anything clips it) and the converted
  output's peak, so it is visible whether a limiter after Applio has anything to do.
"""

import math
import threading
from collections import deque

import numpy as np
import torch
from scipy import signal

SAMPLE_RATE = 48000
HIGHPASS_ORDER = 5  # what the training preprocess uses
METER_SECONDS = 3.0


class InputHighpass:
    """A Butterworth high-pass whose state carries across blocks."""

    def __init__(self, cutoff_hz, sample_rate=SAMPLE_RATE, order=HIGHPASS_ORDER):
        self.cutoff_hz = float(cutoff_hz)
        self.sos = signal.butter(
            order, self.cutoff_hz, btype="high", fs=sample_rate, output="sos"
        )
        self.reset()

    def reset(self):
        # Zero state: the stream starts from silence, as the preprocess's lfilter does.
        self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)

    def process(self, block):
        out, self.zi = signal.sosfilt(self.sos, np.asarray(block, dtype=np.float64), zi=self.zi)
        return out.astype(np.float32)


def gate_fade_out(tail, length):
    """The first `length` samples of `tail` (the audio that would have followed the last
    block) under a cos^2 fade from 1 to 0."""
    length = int(min(length, tail.shape[0]))
    if length <= 0:
        return tail[:0]
    ramp = torch.cos(
        0.5 * math.pi * torch.linspace(0.0, 1.0, length, device=tail.device, dtype=tail.dtype)
    ) ** 2
    return tail[:length] * ramp


def dbfs(value):
    return 20.0 * math.log10(max(float(value), 1e-6))


class LevelMeter:
    """Peaks per block: the generator's before clipping (a device tensor, read only when
    the status line asks, so the audio thread never waits for the GPU) and the
    converted output's (before the Output Gain slider)."""

    def __init__(self, block_ms):
        blocks = max(1, int(math.ceil(METER_SECONDS * 1000.0 / max(block_ms, 1.0))))
        self._lock = threading.Lock()
        self._generator = deque(maxlen=blocks)
        self._output = deque(maxlen=blocks)
        self._generator_max = None
        self._over = None
        self._output_max = 0.0

    def record(self, generator_peak, output_block):
        output_peak = float(np.max(np.abs(output_block))) if len(output_block) else 0.0
        with self._lock:
            if generator_peak is not None:
                peak = generator_peak.detach().reshape(())
                self._generator.append(peak)
                over = (peak > 1.0).to(torch.int64)
                if self._generator_max is None:
                    self._generator_max, self._over = peak.clone(), over
                else:
                    self._generator_max = torch.maximum(self._generator_max, peak)
                    self._over = self._over + over
            self._output.append(output_peak)
            self._output_max = max(self._output_max, output_peak)

    def snapshot(self):
        with self._lock:
            generator = list(self._generator)
            output = list(self._output)
            generator_max, over, output_max = self._generator_max, self._over, self._output_max
        if not output:
            return None
        result = {
            "output_recent": max(output),
            "output_max": output_max,
        }
        if generator:
            result["generator_recent"] = float(torch.stack(generator).max())
            result["generator_max"] = float(generator_max)
            result["generator_over_blocks"] = int(over)
        return result

    def status(self):
        snap = self.snapshot()
        if snap is None:
            return ""
        parts = []
        if "generator_recent" in snap:
            parts.append(
                "generator peak {:.1f} dBFS (max {:.1f}, {} blocks over 0 dBFS)".format(
                    dbfs(snap["generator_recent"]),
                    dbfs(snap["generator_max"]),
                    snap["generator_over_blocks"],
                )
            )
        parts.append(
            "output peak {:.1f} dBFS (max {:.1f})".format(
                dbfs(snap["output_recent"]), dbfs(snap["output_max"])
            )
        )
        return "\nLevel: " + " | ".join(parts)
