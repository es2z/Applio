"""Clock drift correction for the ring transport's two-stream mode.

Two devices on two clocks drift apart by up to a few hundred ppm. The output is
resampled by a ratio within +-1000 ppm of 1:1 that the ReserveController moves every
block (rvc/realtime/ring_buffer.py). soxr's variable-rate mode, used for this first,
cannot do it cleanly (soxr 1.1.0, measured on a 440 Hz sine):
- a slewed ratio change steps the waveform ~64 samples after every chunk boundary,
  160x a clean sine's second difference, once a block: choppy output;
- an immediate change that takes the ratio from 1:1 or above to below it corrupts
  several samples (190x), and drift correction around equal rates crosses 1:1 by
  nature;
- its output per chunk wanders by ~10 ms with its internal buffering, which the
  reserve then has to cover.

This is a windowed-sinc interpolator for exactly that range: the output position
advances by 1 / ratio input samples per output sample in float64, each output is a
64-tap Kaiser-windowed sinc of the input around it, with the tap weights linearly
interpolated between 512 precomputed phases. The ratio can change between any two
calls, the output count of a call is fixed by its input, and the delay is 32 samples
(0.67 ms at 48 kHz).

Measured with the ratio moving between -1000 and +1000 ppm and crossing 1:1: flat
(0.000 dB) to 20 kHz, -0.09 dB at 21 kHz, -2.1 dB at 22 kHz; everything that is not the
tone under -96 dB. The arithmetic runs in torch on the CPU, which releases the GIL so
the audio callbacks are not held up: 4.8 ms for a 1080 ms block (numpy: 16-47 ms).
"""

import numpy as np
import torch

TAPS = 64
PHASES = 512
CUTOFF = 0.94  # of Nyquist
KAISER_BETA = 9.0
PIECE = 16384  # outputs computed per pass, to bound the gathered taps' memory


def _table():
    half = TAPS // 2
    offsets = np.arange(TAPS) - half + 1  # tap j reads input sample floor(p) + offsets[j]
    phases = np.arange(PHASES + 1) / PHASES  # fractional position of p
    distance = offsets[None, :] - phases[:, None]
    window = np.kaiser(TAPS * 64 + 1, KAISER_BETA)
    # Window sampled at the distance, centred at 0 and reaching 0 at +-half.
    w_index = np.clip((distance / half + 1) / 2 * (len(window) - 1), 0, len(window) - 1)
    lower = np.floor(w_index).astype(int)
    upper = np.minimum(lower + 1, len(window) - 1)
    frac = w_index - lower
    w = window[lower] * (1 - frac) + window[upper] * frac
    kernel = CUTOFF * np.sinc(CUTOFF * distance) * w
    kernel /= kernel.sum(axis=1, keepdims=True)  # unit gain at DC for every phase
    return torch.from_numpy(offsets), torch.from_numpy(kernel.astype(np.float32))


_OFFSETS, _KERNEL = _table()


class DriftResampler:
    def __init__(self):
        half = TAPS // 2
        # Primed so the first output sits on the first input sample.
        self._buffer = torch.zeros(half - 1, dtype=torch.float32)
        self._position = float(half - 1)

    @property
    def delay_samples(self):
        return TAPS // 2

    def process(self, samples, ratio):
        """Resample `samples` to `ratio` outputs per input (output rate / input rate)."""
        step = 1.0 / float(ratio)
        samples = torch.as_tensor(np.asarray(samples, dtype=np.float32).reshape(-1))
        buffer = torch.cat((self._buffer, samples))
        last = len(buffer) - TAPS // 2 - 1  # floor(p) may go up to here
        position = self._position
        count = int(np.floor((last + 1 - position) / step)) if position < last + 1 else 0
        out = torch.empty(max(count, 0), dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, count, PIECE):
                n = min(PIECE, count - start)
                # Positions in float64: a float32 position is off by ~1e-3 samples
                # once the buffer is a block long.
                points = position + step * torch.arange(start, start + n, dtype=torch.float64)
                base = torch.floor(points)
                phase = (points - base) * PHASES
                index = torch.clamp(phase.long(), max=PHASES - 1)
                blend = (phase - index).float()
                taps = buffer[base.long()[:, None] + _OFFSETS[None, :]]
                lower = (taps * _KERNEL[index]).sum(1)
                upper = (taps * _KERNEL[index + 1]).sum(1)
                out[start : start + n] = lower + (upper - lower) * blend
        position += step * count
        keep_from = max(0, int(np.floor(position)) - TAPS // 2 + 1)
        self._buffer = buffer[keep_from:].clone()
        self._position = position - keep_from
        return out.numpy()
