"""STFT / iSTFT pair for the CodenameRingFormer decoder.

Ported from rvc/lib/algorithm/conformer/stft.py of the Codename RVC fork (BSD 3-Clause,
Prem Seetharaman, https://github.com/pseeth/pytorch-stft). Only TorchSTFT is reachable
from the generator, so only TorchSTFT is kept.

Two adaptations, neither of which changes the state_dict:

1. The window is a non-persistent buffer instead of a plain tensor pinned to a hardcoded
   "cuda" device. Upstream takes a `device` argument and calls `.to(device)` in the
   constructor, so the module neither follows `.to()` afterwards nor works on CPU - which
   every test in tests/ is. Non-persistent keeps it out of the state_dict, exactly where
   upstream's plain attribute already was.
2. transform/inverse run in fp32. This decoder's STFT sits inside the autocast region and
   the CUDA FFT rejects half precision; DiscriminatorR forces fp32 for the same reason.
"""

import numpy as np
import torch
from scipy.signal import get_window


class TorchSTFT(torch.nn.Module):
    """Magnitude/phase STFT and its inverse, both in fp32."""

    def __init__(
        self,
        filter_length: int = 800,
        hop_length: int = 200,
        win_length: int = 800,
        window: str = "hann",
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer(
            "window",
            torch.from_numpy(
                get_window(window, win_length, fftbins=True).astype(np.float32)
            ),
            persistent=False,
        )

    def transform(self, input_data: torch.Tensor):
        """[B, T] waveform -> ([B, F, frames] magnitude, [B, F, frames] phase)."""
        forward_transform = torch.stft(
            input_data.float(),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window,
            return_complex=True,
        )
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor):
        """([B, F, frames], [B, F, frames]) -> [B, 1, (frames - 1) * hop_length]."""
        spectrum = magnitude.float() * torch.exp(phase.float() * 1j)
        inverse_transform = torch.istft(
            spectrum,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window,
        )
        # unsqueeze to stay consistent with the conv_transpose1d implementation
        return inverse_transform.unsqueeze(-2)

    def forward(self, input_data: torch.Tensor):
        magnitude, phase = self.transform(input_data)
        return self.inverse(magnitude, phase)
