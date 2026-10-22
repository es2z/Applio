"""48 kHz -> 16 kHz as one continuous stream.

`torchaudio.transforms.Resample` zero-pads both ends of whatever it is given. Realtime
used to call it on each block on its own, so every block boundary of the 16 kHz signal
the F0 predictor and the embedder read carried a small discontinuity: measured on
band-limited noise, about 7 samples per boundary (0.4 ms at 16 kHz) off by up to ~18% of
the signal peak, and 48.6 dB SNR for the whole signal at 960 ms blocks (39.3 dB at
80 ms). This keeps the history the filter needs and delays the output by a few samples
so the right side never runs past the input either; the result equals resampling the
whole recording at once, shifted by `delay_samples`.
"""

import torch


class StatefulResampler:
    """Streaming wrapper around a torchaudio Resample with an integer down ratio.

    Each push of N input samples (N divisible by the ratio) returns exactly N / ratio
    output samples, identical to the stateless resampler run over the whole stream,
    `delay_samples` input samples late.
    """

    def __init__(self, resampler, device="cpu"):
        orig = int(getattr(resampler, "orig_freq"))
        new = int(getattr(resampler, "new_freq"))
        # torchaudio reduces by the gcd internally; recover the reduced ratio here.
        from math import gcd

        divisor = gcd(orig, new)
        orig, new = orig // divisor, new // divisor
        if new != 1:
            raise ValueError(
                f"StatefulResampler needs an integer down ratio, got {orig}:{new}"
            )
        self.resampler = resampler
        self.ratio = orig
        self.device = device
        width = int(resampler.width)
        # Output j reads input [ratio*j - width, ratio*j + width + ratio).
        lookahead = width + self.ratio
        self.delay_samples = -(-lookahead // self.ratio) * self.ratio
        history = self.delay_samples + width
        self.history_samples = -(-history // self.ratio) * self.ratio
        self.reset()

    def reset(self):
        self.history = torch.zeros(self.history_samples, device=self.device)

    def push(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.reshape(-1).to(self.history.dtype).to(self.history.device)
        count = samples.shape[0]
        if count % self.ratio:
            raise ValueError(
                f"Push a multiple of {self.ratio} samples; got {count}"
            )
        segment = torch.cat((self.history, samples))
        converted = self.resampler(segment)
        start = (self.history_samples - self.delay_samples) // self.ratio
        output = converted[start : start + count // self.ratio]
        self.history = segment[-self.history_samples :].clone()
        return output
