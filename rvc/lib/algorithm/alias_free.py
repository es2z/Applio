"""Anti-aliased resampling for activations, BigVGAN's alias-free activation filters.

Ported from NVIDIA BigVGAN, alias_free_activation/torch/filter.py and resample.py at commit
65916e066cdedc2388e02ed6701b0b4acb8efe94 (https://github.com/NVIDIA/BigVGAN, MIT), which
adapts https://github.com/junjun3518/alias-free-torch (Apache 2.0) and takes the Kaiser sinc
design from adefossez's julius (MIT); docs/licenses/BigVGAN-LICENSE and
docs/licenses/alias-free-torch-LICENSE.

A pointwise nonlinearity such as Snake creates harmonics above the Nyquist frequency of the
signal it runs on, and at the 1.2 / 12 kHz feature rates of a vocoder's upsampling stages
they fold back as aliasing. BigVGAN's answer, which this is: upsample 2x with a Kaiser
windowed sinc low pass, apply the activation at twice the rate, low pass and decimate back.

The arithmetic is BigVGAN's, line for line. The one deliberate difference is that the filters
are non-persistent buffers (BigVGAN's are persistent): they are fixed coefficients, so a
model that runs its activations between these has exactly the state_dict of one that does
not, and a checkpoint of either loads into the other.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """A [1, 1, kernel_size] Kaiser windowed sinc low pass, normalised to unit DC gain.

    cutoff and half_width are fractions of the sample rate (0.5 is Nyquist).
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5 / cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        # Unit sum, or a little of the constant component would leak through.
        filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    """Depthwise low pass of a [B, C, T] signal, optionally strided."""

    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer(
            "filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size), persistent=False
        )

    def forward(self, x):
        _, C, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(
            x, self.filter.to(x.dtype).expand(C, -1, -1), stride=self.stride, groups=C
        )


class UpSample1d(nn.Module):
    """[B, C, T] -> [B, C, T * ratio], zero stuffing followed by the low pass."""

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(
                cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
            ),
            persistent=False,
        )

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.to(x.dtype).expand(C, -1, -1), stride=self.stride, groups=C
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    """[B, C, T * ratio] -> [B, C, T], the low pass followed by decimation."""

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        return self.lowpass(x)
