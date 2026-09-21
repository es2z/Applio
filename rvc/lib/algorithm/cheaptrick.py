"""CheapTrick spectral envelope estimation, as a differentiable torch module.

Ported from the official SiFi-GAN implementation (MIT, Reo Yoneyama / Nagoya University):
    https://github.com/chomeyama/SiFiGAN/blob/main/sifigan/layers/cheaptrick.py

References:
    - https://www.sciencedirect.com/science/article/pii/S0167639314000697
    - https://github.com/mmorise/World

Used only by the SiFi-GAN source regularisation loss (rvc/train/source_loss.py). It is
pure torch - no numpy, no pyworld - so it runs on the GPU inside the training step and
backpropagates.

The window and lifter tables are precomputed per integer F0 in [f0_floor, f0_ceil], so
the buffers are (f0_ceil + 1, fft_size) and 2 x (f0_ceil + 1, fft_size // 2 + 1). At the
defaults this repository uses (f0_ceil 1100, fft_size 4096) that is about 36 MB.
"""

import math

import torch
import torch.fft
import torch.nn as nn


class AdaptiveWindowing(nn.Module):
    """F0-adaptive windowing: each frame is windowed at 1.5 times its own period."""

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        fft_size: int,
        f0_floor: int,
        f0_ceil: int,
    ):
        super(AdaptiveWindowing, self).__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.register_buffer("window", torch.zeros((f0_ceil + 1, fft_size)))
        self.zero_padding = nn.ConstantPad2d((fft_size // 2, fft_size // 2, 0, 0), 0)

        # Pre-calculation of the window functions
        for f0 in range(f0_floor, f0_ceil + 1):
            half_win_len = round(1.5 * self.sample_rate / f0)
            base_index = torch.arange(
                -half_win_len, half_win_len + 1, dtype=torch.int64
            )
            position = base_index / 1.5 / self.sample_rate
            left = fft_size // 2 - half_win_len
            right = fft_size // 2 + half_win_len + 1
            window = torch.zeros(fft_size)
            window[left:right] = 0.5 * torch.cos(math.pi * position * f0) + 0.5
            average = torch.sum(window * window).pow(0.5)
            self.window[f0] = window / average

    def forward(self, x: torch.Tensor, f: torch.Tensor, power: bool = False):
        """
        Args:
            x: [B, T] waveform.
            f: [B, T'] integer F0 per frame, already clamped and rounded.
            power: power spectrogram instead of magnitude.

        Returns:
            [B, T', fft_size // 2 + 1]
        """
        x = self.zero_padding(x).unfold(1, self.fft_size, self.hop_size)
        windows = self.window[f]
        x = torch.abs(torch.fft.rfft(x[:, :-1, :] * windows))
        return x.pow(2) if power else x


class AdaptiveLiftering(nn.Module):
    """F0-adaptive cepstral liftering: smooths the spectrum over one harmonic spacing."""

    def __init__(
        self,
        sample_rate: int,
        fft_size: int,
        f0_floor: int,
        f0_ceil: int,
        q1: float = -0.15,
    ):
        super(AdaptiveLiftering, self).__init__()

        self.sample_rate = sample_rate
        self.bin_size = fft_size // 2 + 1
        self.q1 = q1
        self.q0 = 1.0 - 2.0 * q1
        self.register_buffer(
            "smoothing_lifter", torch.zeros((f0_ceil + 1, self.bin_size))
        )
        self.register_buffer(
            "compensation_lifter", torch.zeros((f0_ceil + 1, self.bin_size))
        )

        # Pre-calculation of the smoothing lifters and compensation lifters
        for f0 in range(f0_floor, f0_ceil + 1):
            smoothing_lifter = torch.zeros(self.bin_size)
            compensation_lifter = torch.zeros(self.bin_size)
            quefrency = torch.arange(1, self.bin_size) / sample_rate
            smoothing_lifter[0] = 1.0
            smoothing_lifter[1:] = torch.sin(math.pi * f0 * quefrency) / (
                math.pi * f0 * quefrency
            )
            compensation_lifter[0] = self.q0 + 2.0 * self.q1
            compensation_lifter[1:] = self.q0 + 2.0 * self.q1 * torch.cos(
                2.0 * math.pi * f0 * quefrency
            )
            self.smoothing_lifter[f0] = smoothing_lifter
            self.compensation_lifter[f0] = compensation_lifter

    def forward(self, x: torch.Tensor, f: torch.Tensor, elim_0th: bool = False):
        """
        Args:
            x: [B, T', bin_size] magnitude or power spectrogram.
            f: [B, T'] integer F0 per frame.
            elim_0th: zero the 0th cepstrum, i.e. drop the overall level so that the
                source network is the one that accounts for power.

        Returns:
            [B, T', bin_size] log spectral envelope.
        """
        smoothing_lifter = self.smoothing_lifter[f]
        compensation_lifter = self.compensation_lifter[f]

        # Calculating cepstrum
        tmp = torch.cat((x, torch.flip(x[:, :, 1:-1], [2])), dim=2)
        cepstrum = torch.fft.rfft(torch.log(torch.clamp(tmp, min=1e-7))).real

        # Set the 0th cepstrum to 0
        if elim_0th:
            cepstrum = torch.cat(
                [torch.zeros_like(cepstrum[..., :1]), cepstrum[..., 1:]], dim=-1
            )

        # Liftering cepstrum with the lifters
        liftered_cepstrum = cepstrum * smoothing_lifter * compensation_lifter

        # Return the result to the spectral domain
        return torch.fft.irfft(liftered_cepstrum)[:, :, : self.bin_size]


class CheapTrick(nn.Module):
    """CheapTrick-based spectral envelope estimation."""

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        fft_size: int,
        f0_floor: int = 70,
        f0_ceil: int = 340,
        uv_threshold: float = 0.0,
        q1: float = -0.15,
    ):
        super(CheapTrick, self).__init__()

        # fft_size must be larger than 3.0 * sample_rate / f0_floor
        if fft_size <= 3.0 * sample_rate / f0_floor:
            raise ValueError(
                f"CheapTrick needs fft_size > 3 * sample_rate / f0_floor "
                f"({3.0 * sample_rate / f0_floor:.0f}), got fft_size={fft_size} for "
                f"sample_rate={sample_rate}, f0_floor={f0_floor}."
            )
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.uv_threshold = uv_threshold

        self.ada_wind = AdaptiveWindowing(
            sample_rate, hop_size, fft_size, f0_floor, f0_ceil
        )
        self.ada_lift = AdaptiveLiftering(
            sample_rate, fft_size, f0_floor, f0_ceil, q1
        )

    def forward(
        self,
        x: torch.Tensor,
        f: torch.Tensor,
        power: bool = False,
        elim_0th: bool = False,
    ):
        """
        Args:
            x: [B, T] waveform, with T == T' * hop_size.
            f: [B, T'] F0 in Hz, 0 where unvoiced.
            power: use the power rather than magnitude spectrogram.
            elim_0th: exclude the 0th cepstrum.

        Returns:
            [B, T', fft_size // 2 + 1] log spectral envelope.
        """
        # Step0: Round F0 values to integers.
        voiced = (f > self.uv_threshold) * torch.ones_like(f)
        f = voiced * f + (1.0 - voiced) * self.f0_ceil
        f = torch.round(
            torch.clamp(f, min=self.f0_floor, max=self.f0_ceil)
        ).to(torch.int64)

        # Step1: Adaptive windowing and calculate power or amplitude spectrogram.
        x = self.ada_wind(x, f, power)

        # Step2: Smoothing (log axis) and spectral recovery on the cepstrum domain.
        return self.ada_lift(x, f, elim_0th)
