"""Source regularisation loss for the SiFi-GAN vocoder.

Ported from the official implementation (MIT, Reo Yoneyama / Nagoya University):
    https://github.com/chomeyama/SiFiGAN/blob/main/sifigan/losses/reg.py  (ResidualLoss)

SiFi-GAN's generator returns two signals: the waveform and the excitation produced by its
source network. Without a loss on the excitation the source network is unsupervised, the
source-filter decomposition never forms, and what is left is an ordinary vocoder that
happens to have quasi-periodic convolutions in it. This loss supplies that supervision:
it asks the excitation's mel spectrum to match the ground truth waveform's spectrum with
the CheapTrick spectral envelope divided out, i.e. the residual after the vocal tract
filter has been removed.

Only built and evaluated when the vocoder is SiFi-GAN; every other vocoder's training
loop is untouched.

The defaults here follow this repository's F0 range (50 - 1100 Hz) rather than the
official 24 kHz speech config (100 - 840 Hz), which would clamp anything below 100 Hz.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel

from rvc.lib.algorithm.cheaptrick import CheapTrick

# fft_size must satisfy fft_size > 3 * sample_rate / f0_floor; 4096 leaves headroom for
# f0_floor 50 at 48 kHz (which needs > 2880).
DEFAULT_FFT_SIZE = 4096
DEFAULT_F0_FLOOR = 50
DEFAULT_F0_CEIL = 1100


def stft_magnitude(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: torch.Tensor,
    power: bool = False,
):
    """Magnitude (or power) spectrogram as [B, T', fft_size // 2 + 1]."""
    spec = torch.stft(
        x,
        fft_size,
        hop_size,
        win_length,
        window,
        return_complex=True,
    )
    magnitude = torch.clamp(spec.real**2 + spec.imag**2, min=1e-7)
    if not power:
        magnitude = torch.sqrt(magnitude)
    return magnitude.transpose(2, 1)


class ResidualLoss(nn.Module):
    """L1 between the excitation's mel spectrum and the envelope-flattened target."""

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        fft_size: int = DEFAULT_FFT_SIZE,
        f0_floor: int = DEFAULT_F0_FLOOR,
        f0_ceil: int = DEFAULT_F0_CEIL,
        n_mels: int = 80,
        fmin: int = 0,
        fmax: int = None,
        power: bool = False,
        elim_0th: bool = True,
    ):
        super(ResidualLoss, self).__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.cheaptrick = CheapTrick(
            sample_rate=sample_rate,
            hop_size=hop_size,
            fft_size=fft_size,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
        )
        self.win_length = fft_size
        self.register_buffer("window", torch.hann_window(self.win_length))

        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2
        melmat = librosa_mel(
            sr=sample_rate,
            n_fft=fft_size,
            n_mels=n_mels,
            fmin=fmin,
            fmax=self.fmax,
        ).T
        self.register_buffer("melmat", torch.from_numpy(melmat).float())

        self.power = power
        self.elim_0th = elim_0th

    def forward(self, s: torch.Tensor, y: torch.Tensor, f: torch.Tensor):
        """
        Args:
            s: [B, 1, T] excitation predicted by the source network.
            y: [B, 1, T] ground truth waveform.
            f: [B, 1, T'] or [B, T'] F0 in Hz, 0 where unvoiced, with T == T' * hop_size.

        Returns:
            Scalar loss.
        """
        s, y = s.squeeze(1), y.squeeze(1)
        f = f.squeeze(1) if f.dim() == 3 else f

        with torch.no_grad():
            # Log spectral envelope of the target, and its spectrogram.
            e = self.cheaptrick.forward(y, f, self.power, self.elim_0th)
            y = stft_magnitude(
                y,
                self.fft_size,
                self.hop_size,
                self.win_length,
                self.window,
                power=self.power,
            )
            # adjust length, (B, T', C)
            minlen = min(e.size(1), y.size(1))
            e, y = e[:, :minlen, :], y[:, :minlen, :]

            if self.elim_0th:
                y_mean = y.mean(dim=-1, keepdim=True)

            # Target of the source signal: the target spectrum with the envelope removed.
            y = torch.log(torch.clamp(y, min=1e-7))
            t = (y - e).exp()
            if self.elim_0th:
                t_mean = t.mean(dim=-1, keepdim=True)
                t = y_mean / t_mean * t

            t = torch.matmul(t, self.melmat)
            t = torch.log(torch.clamp(t, min=1e-7))

        s = stft_magnitude(
            s,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            power=self.power,
        )
        minlen = min(minlen, s.size(1))
        s, t = s[:, :minlen, :], t[:, :minlen, :]

        s = torch.matmul(s, self.melmat)
        s = torch.log(torch.clamp(s, min=1e-7))

        return F.l1_loss(s, t.detach())
