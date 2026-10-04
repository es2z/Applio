"""The multi-scale sub-band CQT discriminator (MS-SB-CQT) RingFormer+ trains against.

MS-SB-CQT is from Gu et al., "Multi-Scale Sub-Band Constant-Q Transform Discriminator
for High-Fidelity Vocoder" (arXiv 2311.14957). The RingFormer paper trains RingFormer
against it, unmodified, next to a multi-period discriminator.

Sources, both MIT:
- DiscriminatorCQT: NVIDIA BigVGAN, https://github.com/NVIDIA/BigVGAN discriminators.py
  at commit 3dce7f2a38392fb281325a8a094f6da4356b2f10 (docs/licenses/BigVGAN-LICENSE).
- ConstantQTransform: nnAudio 0.3.4 (PyPI), features/cqt.py CQT2010v2 and the helpers it
  uses from utils.py (docs/licenses/nnAudio-LICENSE). BigVGAN builds its discriminator on
  that class.

What differs from the sources, none of which changes the arithmetic:
- nnAudio and torchaudio's Resample register their kernels as persistent buffers, so they
  would land in every D_*.pth, and the CQT kernels' shape depends on the sample rate (the
  bins_per_octave=36 kernel is 512 taps at 64 / 80 kHz and 1024 at 96 kHz). Here every
  buffer is non-persistent, Resample's kernel included, so the state_dict holds only
  trained weights. The resampler stays transforms.Resample rather than
  functional.resample: the transform builds its kernel in float64 before casting, the
  function in the input's dtype, and the two differ by up to 7e-7.
- The resample and the CQT always run in fp32 with autocast off. Both are convolutions,
  which autocast would run in half precision even on an fp32 input: measured against fp32,
  1e-3 relative error under fp16 and 7e-3 under bf16. The discriminator's own
  convolutions after them follow autocast like every other sub-discriminator.
- BigVGAN stores the CQT's real and imaginary parts in variables called z_amplitude and
  z_phase. They are the real and imaginary parts, and are named so here.
- The logits are flattened to (batch, n) like every other sub-discriminator's; the losses
  take a mean, so the value is unchanged.
- The defaults are the RingFormer and Amphion configuration (filters 32), where BigVGAN v2
  uses 128.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms
from scipy import signal
from scipy.signal import get_window
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.lib.algorithm.residuals import LRELU_SLOPE

# nnAudio's default lowest bin, C1. With 9 octaves the top bin is ~16.5 kHz, which is why
# BigVGAN doubles the sample rate before the transform: at 22.05 / 24 kHz the top octave
# would be above Nyquist. Here the doubling also keeps 32 kHz models valid.
CQT_FMIN = 32.70

# RingFormer's configs/vits2_ljs_ring.json and Amphion's MS-SB-CQT defaults. BigVGAN v2
# uses 128, which measured 2.5x the discriminator time for the same step-0 losses.
CQT_FILTERS = 32


def _create_lowpass_filter(band_center=0.5, kernel_length=256, transition_bandwidth=0.03):
    """nnAudio's create_lowpass_filter: an FIR low-pass at band_center of Nyquist."""
    passband_max = band_center / (1 + transition_bandwidth)
    stopband_min = band_center * (1 + transition_bandwidth)
    kernel = signal.firwin2(
        kernel_length, [0.0, passband_max, stopband_min, 1.0], [1.0, 1.0, 0.0, 0.0]
    )
    return kernel.astype(np.float32)


def _early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    """nnAudio's (and librosa's) count of decimations that can happen before the CQT."""
    count_by_frequency = max(
        0, int(np.ceil(np.log2(0.85 * nyquist / filter_cutoff)) - 1) - 1
    )
    count_by_hop = max(0, int(np.ceil(np.log2(hop_length))) - n_octaves + 1)
    return min(count_by_frequency, count_by_hop)


def _create_cqt_kernels(q, sample_rate, fmin, n_bins, bins_per_octave):
    """nnAudio's create_cqt_kernels for a Hann window, L1 normalisation and gamma 0.

    Time-domain kernels, each centred in a power-of-two frame long enough for the
    lowest bin. Returns (kernels, frame length).
    """
    freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
    lengths = np.ceil(q * sample_rate / freqs)
    fft_len = int(2 ** (np.ceil(np.log2(int(max(lengths))))))
    kernels = np.zeros((int(n_bins), fft_len), dtype=np.complex64)
    for k in range(int(n_bins)):
        freq = freqs[k]
        length = lengths[k]
        if length % 2 == 1:  # pad more zeros on the right
            start = int(np.ceil(fft_len / 2.0 - length / 2.0)) - 1
        else:
            start = int(np.ceil(fft_len / 2.0 - length / 2.0))
        window = get_window("hann", int(length), fftbins=True)
        kernel = (
            window
            * np.exp(
                np.r_[-length // 2 : length // 2] * 1j * 2 * np.pi * freq / sample_rate
            )
            / length
        )
        kernels[k, start : start + int(length)] = kernel / np.linalg.norm(kernel, 1)
    return kernels, fft_len


class ConstantQTransform(torch.nn.Module):
    """nnAudio's CQT2010v2 with output_format="Complex" and pad_mode="constant".

    The top octave's kernels are applied to the input, which is then halved in rate once
    per lower octave and convolved with the same kernels (Schoerkhuber 2010). The output
    is librosa-normalised and, like nnAudio's, shaped (batch, n_bins, frames, 2) with the
    real part first; the lowest frequency is bin 0.

    Measured against librosa.cqt at 96 kHz: |CQT| agrees within 0.85% on tones from 45 Hz
    to 7 kHz. The phase reference differs per bin, which does not matter to a
    discriminator that sees real and generated audio through the same transform.
    """

    def __init__(self, sample_rate, hop_length, n_bins, bins_per_octave, fmin=CQT_FMIN):
        super().__init__()
        q = 1.0 / (2 ** (1 / bins_per_octave) - 1)
        self.n_bins = n_bins
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))

        lowpass = torch.tensor(_create_lowpass_filter(0.50, 256, 0.001))
        self.register_buffer("lowpass_filter", lowpass[None, None, :], persistent=False)

        # The top octave's lowest and highest bins.
        fmin_t = fmin * 2 ** (self.n_octaves - 1)
        remainder = n_bins % bins_per_octave
        if remainder == 0:
            fmax_t = fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
        else:
            fmax_t = fmin_t * 2 ** ((remainder - 1) / bins_per_octave)
        fmin_t = fmax_t / 2 ** (1 - 1 / bins_per_octave)
        if fmax_t > sample_rate / 2:
            raise ValueError(
                f"The top CQT bin {fmax_t:.0f} Hz is above the Nyquist frequency of "
                f"{sample_rate} Hz; reduce the number of bins."
            )

        # Early downsampling, as nnAudio decides it. It is 1 (none) for every scale
        # RingFormer+ uses at 32, 40 and 48 kHz, because a hop of 256 or 512 over 9 octaves
        # leaves no factor of two to spare, but it is kept so the class stays CQT2010v2.
        filter_cutoff = fmax_t * (1 + 0.5 * 1.5 / q)  # 1.5: the Hann window's bandwidth
        count = _early_downsample_count(
            sample_rate // 2, filter_cutoff, hop_length, self.n_octaves
        )
        self.downsample_factor = 2**count
        self.hop_length = hop_length // self.downsample_factor
        sample_rate = sample_rate / float(self.downsample_factor)
        if self.downsample_factor != 1:
            early = torch.tensor(
                _create_lowpass_filter(1 / self.downsample_factor, 256, 0.03)
            )
            self.register_buffer(
                "early_downsample_filter", early[None, None, :], persistent=False
            )
        else:
            self.early_downsample_filter = None

        kernels, self.kernel_length = _create_cqt_kernels(
            q, sample_rate, fmin_t, min(bins_per_octave, n_bins), bins_per_octave
        )
        self.register_buffer(
            "kernels_real", torch.tensor(kernels.real).unsqueeze(1), persistent=False
        )
        self.register_buffer(
            "kernels_imag", torch.tensor(kernels.imag).unsqueeze(1), persistent=False
        )

        # librosa's normalisation, per bin over every octave.
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
        lengths = torch.tensor(np.ceil(q * sample_rate / freqs)).float()
        self.register_buffer("lengths", lengths, persistent=False)

    def _octave(self, x, hop_length):
        pad = self.kernel_length // 2
        x = F.pad(x, (pad, pad), mode="constant", value=0.0)
        real = F.conv1d(x, self.kernels_real, stride=hop_length)
        imag = -F.conv1d(x, self.kernels_imag, stride=hop_length)
        return torch.stack((real, imag), -1)

    @staticmethod
    def _downsample(x, kernel, factor):
        return F.conv1d(x, kernel, stride=(factor,), padding=((kernel.shape[-1] - 1) // 2,))

    def forward(self, x):
        """(batch, 1, samples) -> (batch, n_bins, frames, 2)."""
        if self.early_downsample_filter is not None:
            x = self._downsample(x, self.early_downsample_filter, self.downsample_factor)
        hop_length = self.hop_length
        cqt = self._octave(x, hop_length)
        x_down = x
        for _ in range(self.n_octaves - 1):
            hop_length = hop_length // 2
            x_down = self._downsample(x_down, self.lowpass_filter, 2)
            cqt = torch.cat((self._octave(x_down, hop_length), cqt), 1)
        cqt = cqt[:, -self.n_bins :, :]
        cqt = cqt * self.downsample_factor
        return cqt * torch.sqrt(self.lengths.view(-1, 1, 1))


def _get_2d_padding(kernel_size, dilation=(1, 1)):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class DiscriminatorCQT(torch.nn.Module):
    """One scale of MS-SB-CQT: a 2D convolutional discriminator on a complex CQT.

    The input is resampled to twice the sample rate, transformed, and each octave's bins
    get their own first convolution (the "sub-band" part) before a shared stack.
    conv_pres and the first conv of the stack are plain convolutions and the rest are
    weight-normed, exactly as in BigVGAN; use_spectral_norm swaps weight norm for spectral
    norm on those same layers, as it does for every other sub-discriminator here.

    Args:
        sample_rate (int): The training sample rate. The CQT runs at twice this.
        hop_length (int): Hop of the top octave, at twice the sample rate.
        n_octaves (int): Octaves above fmin.
        bins_per_octave (int): CQT bins per octave.
        fmin (float): Lowest bin, Hz.
        filters (int): Channels of the convolution stack.
        use_spectral_norm (bool): Whether to use spectral normalization.
    """

    def __init__(
        self,
        sample_rate,
        hop_length,
        n_octaves,
        bins_per_octave,
        fmin=CQT_FMIN,
        filters=CQT_FILTERS,
        max_filters=1024,
        filters_scale=1,
        dilations=(1, 2, 4),
        use_spectral_norm=False,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.n_octaves = int(n_octaves)
        self.bins_per_octave = int(bins_per_octave)
        self.fmin = float(fmin)
        # What discriminator_layout calls this sub-discriminator. The sample rate is left
        # out, as it is for every other kind: the bins are absolute frequencies, and a
        # discriminator is never inherited across sample rates without its generator,
        # which refuses to be.
        self.descriptor = (
            "Q",
            self.hop_length,
            self.n_octaves,
            self.bins_per_octave,
            self.fmin,
        )
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.resample = torchaudio.transforms.Resample(
            self.sample_rate, 2 * self.sample_rate
        )
        # Re-registered rather than left persistent: the kernel is a function of the two
        # rates alone, and BigVGAN's checkpoints carry it for no reason.
        self.resample.register_buffer(
            "kernel", self.resample.kernel, persistent=False
        )
        self.cqt = ConstantQTransform(
            2 * self.sample_rate,
            self.hop_length,
            self.bins_per_octave * self.n_octaves,
            self.bins_per_octave,
            fmin=self.fmin,
        )

        kernel_size = (3, 9)
        stride = (1, 2)
        # Two input channels: the CQT's real and imaginary parts.
        self.conv_pres = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(2, 2, kernel_size, padding=_get_2d_padding(kernel_size))
                for _ in range(self.n_octaves)
            ]
        )
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    2, filters, kernel_size, padding=_get_2d_padding(kernel_size)
                )
            ]
        )
        in_channels = min(filters_scale * filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_channels = min((filters_scale ** (i + 1)) * filters, max_filters)
            self.convs.append(
                norm_f(
                    torch.nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        dilation=(dilation, 1),
                        padding=_get_2d_padding(kernel_size, (dilation, 1)),
                    )
                )
            )
            in_channels = out_channels
        out_channels = min((filters_scale ** (len(dilations) + 1)) * filters, max_filters)
        self.convs.append(
            norm_f(
                torch.nn.Conv2d(
                    in_channels, out_channels, (3, 3), padding=_get_2d_padding((3, 3))
                )
            )
        )
        self.conv_post = norm_f(
            torch.nn.Conv2d(out_channels, 1, (3, 3), padding=_get_2d_padding((3, 3)))
        )
        self.activation = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []

        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.cqt(self.resample(x.float()))

        real = z[:, :, :, 0].unsqueeze(1)
        imag = z[:, :, :, 1].unsqueeze(1)
        z = torch.cat([real, imag], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, bins, frames] -> [B, C, frames, bins]

        bins = self.bins_per_octave
        x = torch.cat(
            [
                self.conv_pres[i](z[:, :, :, i * bins : (i + 1) * bins])
                for i in range(self.n_octaves)
            ],
            dim=-1,
        )

        # As in BigVGAN, conv_post's output is the logits only, not a feature map.
        for conv in self.convs:
            x = self.activation(conv(x))
            fmap.append(x)
        x = self.conv_post(x)

        return torch.flatten(x, 1, -1), fmap
