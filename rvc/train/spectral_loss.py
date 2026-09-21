"""CodenameRingFormer's spectral loss: magnitude L1 plus phase distance.

Ported from the Codename RVC fork's train.py, where it is computed inline as

    loss_magnitude = l1_loss(mag, |STFT(y)|)
    loss_phase = phase_loss(STFT(y), STFT(y_hat))
    loss_sd = (loss_magnitude + loss_phase) * 0.7

and added to the generator's total. Both terms use the decoder's own iSTFT settings, so
the magnitude term compares the spectrum the decoder *predicted* against the target's -
supervising it before the inverse transform rather than after - while the phase term
compares the two waveforms' spectra. That asymmetry is upstream's and is kept.

Only built when the vocoder is CodenameRingFormer, so no other training loop changes.
"""

import torch
import torch.nn.functional as F
from torch import nn

from rvc.train.losses import phase_loss


class SpectralDistanceLoss(nn.Module):
    """Magnitude L1 + phase distance at the decoder's iSTFT resolution.

    Args:
        n_fft: gen_istft_n_fft of the decoder.
        hop_size: gen_istft_hop_size of the decoder.
    """

    def __init__(self, n_fft: int, hop_size: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        # [B, 1, T] -> [B, T], and fp32 because the CUDA FFT rejects half precision.
        return torch.stft(
            waveform.float().reshape(-1, waveform.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )

    def forward(
        self, wave: torch.Tensor, y_hat: torch.Tensor, magnitude: torch.Tensor
    ) -> torch.Tensor:
        """Unweighted loss; the caller scales it by config.train.c_sd.

        Args:
            wave: [B, 1, T] target waveform.
            y_hat: [B, 1, T] generated waveform.
            magnitude: [B, n_fft // 2 + 1, frames] magnitude the decoder predicted.
        """
        target = self._stft(wave)
        generated = self._stft(y_hat)
        return F.l1_loss(magnitude.float(), target.abs()) + phase_loss(target, generated)
