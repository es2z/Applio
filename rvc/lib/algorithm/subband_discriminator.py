"""Sub-band discriminator (SBD), as APEX-GAN's discriminator ("apex-gan") uses it.

Ported from the Codename RVC fork at commit ef613f900d0b646123081b331332996b92cd52e5
(https://github.com/DanielNawa/codename-rvc-fork-4, MIT,
docs/licenses/codename-rvc-fork-LICENSE):

* PQMF from rvc/lib/algorithm/discriminators/multi/pqmf.py, itself adapted from
  ParallelWaveGAN (https://github.com/kan-bayashi/ParallelWaveGAN, MIT,
  docs/licenses/ParallelWaveGAN-LICENSE).
* MDC, the SBD block and the band table from rvc/lib/algorithm/discriminators/multi/hmdd.py,
  which mpd_sbd_mrd_combined.py (APEX-GAN's MPD_SBD_MRD_Combined) imports. That is Avocodo's
  SBD (https://github.com/ncsoft/avocodo, BSD 3-Clause, docs/licenses/Avocodo-LICENSE).

Upstream's SBD is one module that runs one 16-band and one 64-band PQMF analysis and feeds
four blocks: bands 0-6, 0-11 and 0-16 of the 16-band split, and all 64 bands of the other
split transposed, so the block convolves across frequency with time as its channels. Here
each of the four blocks is its own sub-discriminator in MultiPeriodDiscriminator's list,
holding its own PQMF, because everything that consumes a discriminator here - the losses,
gradient checkpointing and the warm start, which matches sub-discriminators by what they
look at - expects one (logits, feature maps) pair per sub-discriminator. The 16-band
analysis is therefore computed three times instead of once: identical numbers, at a cost
that is small next to the period discriminators'.

The PQMF filters are non-persistent buffers. Upstream registers them persistently; they are
a pure function of the constants below, so keeping them out of the state_dict leaves only
trained weights in every D_*.pth, the same choice rvc/lib/algorithm/cqt_discriminator.py
makes. Like upstream, the analysis follows autocast.

Reproduced verbatim: MDC pads its strided post convolution with the *last* branch's kernel
size and dilation (Avocodo's own quirk), and the transposed block's input width is
segment_size // 64, so that block's first layer depends on the training segment length.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

# (subbands, taps, cutoff_ratio, kaiser beta)
SBD_PQMF = (16, 256, 0.03, 10.0)
SBD_TRANSPOSED_PQMF = (64, 256, 0.10, 9.0)

# One row per block, in upstream's order.
SBD_FILTERS = [
    [64, 128, 256, 256, 256],
    [64, 128, 256, 256, 256],
    [64, 128, 256, 256, 256],
    [32, 64, 128, 128, 128],
]
SBD_STRIDES = [
    [1, 1, 3, 3, 1],
    [1, 1, 3, 3, 1],
    [1, 1, 3, 3, 1],
    [1, 1, 3, 3, 1],
]
SBD_KERNEL_SIZES = [
    [[7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7]],
    [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]],
    [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]],
]
SBD_DILATIONS = [
    [[5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11]],
    [[3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7]],
    [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
    [[1, 2, 3], [1, 2, 3], [1, 2, 3], [2, 3, 5], [2, 3, 5]],
]
SBD_BAND_RANGES = [[0, 6], [0, 11], [0, 16], [0, 64]]
SBD_TRANSPOSE = [False, False, False, True]


def subband_descriptor(band: int):
    """("B", PQMF subbands, (first band, last band), transposed) of one SBD block."""
    transpose = SBD_TRANSPOSE[band]
    subbands = (SBD_TRANSPOSED_PQMF if transpose else SBD_PQMF)[0]
    low, high = SBD_BAND_RANGES[band]
    return ("B", subbands, (low, high), transpose)


def subband_descriptors():
    """Descriptors of every SBD block, in ModuleList order."""
    return [subband_descriptor(band) for band in range(len(SBD_BAND_RANGES))]


def design_prototype_filter(taps: int = 62, cutoff_ratio: float = 0.142, beta: float = 9.0):
    """The Kaiser-windowed prototype lowpass filter of a PQMF bank."""
    assert taps % 2 == 0, "taps must be even."
    assert 0.0 < cutoff_ratio < 1.0, "cutoff_ratio must be in (0, 1)."
    omega_c = np.pi * cutoff_ratio
    n = np.arange(taps + 1)
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (n - 0.5 * taps)) / (np.pi * (n - 0.5 * taps))
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # the NaN at the centre tap
    return h_i * np.kaiser(taps + 1, beta)


class PQMF(nn.Module):
    """Pseudo quadrature mirror filter bank, analysis only (synthesis is never used)."""

    def __init__(
        self,
        subbands: int = 4,
        taps: int = 62,
        cutoff_ratio: float = 0.142,
        beta: float = 9.0,
    ):
        super().__init__()
        self.subbands = subbands
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            phase = (2 * k + 1) * (np.pi / (2 * subbands)) * (
                np.arange(taps + 1) - taps / 2
            )
            sign = (-1) ** k * np.pi / 4
            h_analysis[k] = 2 * h_proto * np.cos(phase + sign)
        self.register_buffer(
            "analysis_filter",
            torch.from_numpy(h_analysis).float().unsqueeze(1),
            persistent=False,
        )
        updown = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown, persistent=False)
        self.pad_fn = nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 1, T] -> [B, subbands, T // subbands]"""
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class MDC(nn.Module):
    """Multi-dilated convolution: parallel dilated branches, summed, then a strided conv."""

    def __init__(
        self,
        in_channels,
        out_channels,
        strides,
        kernel_size,
        dilations,
        use_spectral_norm=False,
    ):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.d_convs = nn.ModuleList()
        for _k, _d in zip(kernel_size, dilations):
            self.d_convs.append(
                norm_f(
                    Conv1d(
                        in_channels,
                        out_channels,
                        _k,
                        dilation=_d,
                        padding=_get_padding(_k, _d),
                    )
                )
            )
        # The padding uses the last branch's _k and _d, as in Avocodo; it has to be read
        # after the loop rather than computed per branch.
        self.post_conv = norm_f(
            Conv1d(
                out_channels,
                out_channels,
                3,
                stride=strides,
                padding=_get_padding(_k, _d),
            )
        )

    def forward(self, x):
        _out = None
        for conv in self.d_convs:
            _x = F.leaky_relu(conv(x).unsqueeze(-1), 0.2)
            _out = _x if _out is None else torch.cat([_out, _x], dim=-1)
        return F.leaky_relu(self.post_conv(torch.sum(_out, dim=-1)), 0.2)


class DiscriminatorSB(nn.Module):
    """One SBD block on its own: a PQMF band range in, (logits, feature maps) out.

    Its convs and post_conv are upstream's SBDBlock name for name, so upstream's
    `sbd.blocks.{band}.` keys map onto this module's by prefix alone.

    Args:
        band: which of the four SBD blocks (see SBD_BAND_RANGES).
        segment_size: training segment length in samples; sets the transposed block's
            input width (segment_size // 64).
        use_spectral_norm: spectral instead of weight normalisation.
    """

    def __init__(self, band: int, segment_size: int, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.band = band
        self.transpose = SBD_TRANSPOSE[band]
        self.band_range = tuple(SBD_BAND_RANGES[band])
        self.descriptor = subband_descriptor(band)
        self.pqmf = PQMF(*(SBD_TRANSPOSED_PQMF if self.transpose else SBD_PQMF))

        low, high = self.band_range
        segment_dim = (segment_size // high) - low if self.transpose else high - low
        filters = SBD_FILTERS[band]
        filters_in_out = [(segment_dim, filters[0])]
        for i in range(len(filters) - 1):
            filters_in_out.append((filters[i], filters[i + 1]))
        self.convs = nn.ModuleList(
            [
                MDC(
                    in_channels=f[0],
                    out_channels=f[1],
                    strides=s,
                    kernel_size=k,
                    dilations=d,
                    use_spectral_norm=use_spectral_norm,
                )
                for s, f, k, d in zip(
                    SBD_STRIDES[band], filters_in_out, SBD_KERNEL_SIZES[band], SBD_DILATIONS[band]
                )
            ]
        )
        self.post_conv = norm_f(Conv1d(filters[-1], 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor):
        low, high = self.band_range
        x = self.pqmf.analysis(x)[:, low:high, :]
        if self.transpose:
            x = torch.transpose(x, 1, 2)
        fmap = []
        for conv in self.convs:
            x = conv(x)
            fmap.append(x)
        return torch.flatten(self.post_conv(x), 1, -1), fmap
