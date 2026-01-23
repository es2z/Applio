# BigVGAN Generator for Applio RVC
# Adapted from NVIDIA BigVGAN (https://github.com/NVIDIA/BigVGAN)
# Licensed under MIT License
#
# This implementation adapts BigVGAN for use with Applio's RVC architecture:
# - Input: latent z (inter_channels=192) instead of mel-spectrogram
# - F0 conditioning via source module (like HiFi-GAN NSF)
# - Speaker embedding (g) support

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import init_weights, get_padding


# =============================================================================
# Alias-Free Activation Components
# =============================================================================

def sinc(x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of sinc function: sin(pi * x) / (pi * x)
    """
    return torch.where(
        x == 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.sin(math.pi * x) / (math.pi * x),
    )


def kaiser_sinc_filter1d(
    cutoff: float, half_width: float, kernel_size: int
) -> torch.Tensor:
    """
    Create a Kaiser-windowed sinc filter for anti-aliasing.

    Args:
        cutoff: Normalized cutoff frequency (0 to 0.5)
        half_width: Transition band half-width
        kernel_size: Filter kernel size

    Returns:
        Filter tensor of shape [1, 1, kernel_size]
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Calculate Kaiser window beta parameter
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0

    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # Generate time indices
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    # Generate sinc filter
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        filter_ = filter_ / filter_.sum()

    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    """Low-pass filter using Kaiser-windowed sinc."""

    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        super().__init__()
        if cutoff < 0.0:
            raise ValueError("Cutoff must be non-negative.")
        if cutoff > 0.5:
            raise ValueError("Cutoff must not exceed 0.5.")

        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

        filter_tensor = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C
        )


class UpSample1d(nn.Module):
    """Anti-aliased upsampling using Kaiser-sinc filter."""

    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )

        filter_tensor = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        self.register_buffer("filter", filter_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    """Anti-aliased downsampling using low-pass filter."""

    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


# =============================================================================
# Activation Functions
# =============================================================================

class Snake(nn.Module):
    """
    Snake activation function: x + 1/a * sin^2(x*a)

    From: https://arxiv.org/abs/2006.08195
    """

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return x + (1.0 / (alpha + self.no_div_by_zero)) * torch.pow(
            torch.sin(x * alpha), 2
        )


class SnakeBeta(nn.Module):
    """
    SnakeBeta activation: x + 1/b * sin^2(x*a)

    Separates frequency (alpha) and magnitude (beta) control.
    """

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(x * alpha), 2
        )


class Activation1d(nn.Module):
    """
    Anti-aliased activation: Upsample -> Activation -> Downsample

    This prevents aliasing artifacts from nonlinear activations.
    """

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


# =============================================================================
# AMP Blocks (Anti-aliased Multi-Periodicity)
# =============================================================================

class AMPBlock1(nn.Module):
    """
    AMPBlock with two convolutions per dilation rate.
    Uses Snake/SnakeBeta activation with anti-aliasing.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = "snakebeta",
        snake_logscale: bool = True,
    ):
        super().__init__()

        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
            )
            for d in dilation
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
            )
            for _ in dilation
        ])
        self.convs2.apply(init_weights)

        num_layers = len(self.convs1) + len(self.convs2)

        # Create activation functions
        if activation == "snake":
            self.activations = nn.ModuleList([
                Activation1d(Snake(channels, alpha_logscale=snake_logscale))
                for _ in range(num_layers)
            ])
        elif activation == "snakebeta":
            self.activations = nn.ModuleList([
                Activation1d(SnakeBeta(channels, alpha_logscale=snake_logscale))
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class AMPBlock2(nn.Module):
    """
    Simplified AMPBlock with one convolution per dilation rate.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = "snakebeta",
        snake_logscale: bool = True,
    ):
        super().__init__()

        self.convs = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
            )
            for d in dilation
        ])
        self.convs.apply(init_weights)

        num_layers = len(self.convs)

        if activation == "snake":
            self.activations = nn.ModuleList([
                Activation1d(Snake(channels, alpha_logscale=snake_logscale))
                for _ in range(num_layers)
            ])
        elif activation == "snakebeta":
            self.activations = nn.ModuleList([
                Activation1d(SnakeBeta(channels, alpha_logscale=snake_logscale))
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, act in zip(self.convs, self.activations):
            xt = act(x)
            xt = conv(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)


# =============================================================================
# F0 Source Module (for pitch conditioning)
# =============================================================================

class SineGenerator(nn.Module):
    """
    Sine wave generator for F0-based source signal.
    Similar to HiFi-GAN NSF's source module.
    """

    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.dim = harmonic_num + 1

    def _f0_to_uv(self, f0: torch.Tensor) -> torch.Tensor:
        """Convert F0 to voiced/unvoiced flag."""
        return (f0 > self.voiced_threshold).float()

    def forward(
        self, f0: torch.Tensor, upp: int
    ) -> tuple:
        """
        Generate sine waves from F0.

        Args:
            f0: Fundamental frequency [B, T]
            upp: Upsampling factor

        Returns:
            sine_waves: [B, T*upp, dim]
            uv: Voiced/unvoiced flag [B, T*upp]
            noise: Random noise [B, T*upp, 1]
        """
        with torch.no_grad():
            # Expand F0 to target length
            f0 = f0.unsqueeze(-1)  # [B, T, 1]
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)

            # Generate harmonics
            for i in range(self.dim):
                f0_buf[:, :, i] = f0[:, :, 0] * (i + 1)

            # Upsample F0
            f0_buf = F.interpolate(
                f0_buf.transpose(1, 2), scale_factor=upp, mode="nearest"
            ).transpose(1, 2)

            # Generate phase
            rad_values = (f0_buf / self.sample_rate) % 1
            rand_ini = torch.rand(f0_buf.shape[0], f0_buf.shape[2], device=f0.device)
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

            # Cumulative sum for phase
            tmp_over_one = torch.cumsum(rad_values, dim=1)
            tmp_over_one = tmp_over_one - torch.floor(tmp_over_one)

            # Convert to sine
            sine_waves = torch.sin(2 * math.pi * tmp_over_one) * self.sine_amp

            # Voiced/unvoiced
            uv = self._f0_to_uv(f0[:, :, 0])
            uv = F.interpolate(
                uv.unsqueeze(1), scale_factor=upp, mode="nearest"
            ).squeeze(1)

            # Add noise for unvoiced
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp.unsqueeze(-1) * torch.randn_like(sine_waves[:, :, :1])

            # Apply UV mask to sine
            sine_waves = sine_waves * uv.unsqueeze(-1)

        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """
    Source module that generates harmonic excitation from F0.
    """

    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ):
        super().__init__()
        self.sine_gen = SineGenerator(
            sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0: torch.Tensor, upp: int) -> tuple:
        sine_waves, uv, noise = self.sine_gen(f0, upp)
        sine_waves = sine_waves.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_waves))
        return sine_merge, None, None


# =============================================================================
# BigVGAN Generator
# =============================================================================

class BigVGANGenerator(nn.Module):
    """
    BigVGAN Generator adapted for Applio RVC.

    This implementation adapts NVIDIA's BigVGAN for use with Applio's RVC:
    - Input: latent z (inter_channels) instead of mel-spectrogram
    - F0 conditioning via source module (optional, like HiFi-GAN NSF)
    - Speaker embedding (g) support

    Args:
        initial_channel: Number of input channels (inter_channels, typically 192)
        resblock_kernel_sizes: Kernel sizes for residual blocks
        resblock_dilation_sizes: Dilation rates for residual blocks
        upsample_rates: Upsampling rates for each stage
        upsample_initial_channel: Initial channel count after first conv
        upsample_kernel_sizes: Kernel sizes for upsampling layers
        gin_channels: Number of speaker embedding channels
        sr: Sample rate
        checkpointing: Whether to use gradient checkpointing
        activation: Activation type ("snake" or "snakebeta")
        snake_logscale: Whether to use log-scale for snake parameters
    """

    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
        checkpointing: bool = False,
        activation: str = "snakebeta",
        snake_logscale: bool = True,
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.checkpointing = checkpointing
        self.sr = sr

        # Calculate total upsampling factor
        self.upp = math.prod(upsample_rates)

        # F0 upsampling and source module
        self.f0_upsamp = nn.Upsample(scale_factor=self.upp)
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)

        # Pre-convolution: convert latent channels to initial channel
        self.conv_pre = weight_norm(
            Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )

        # Select resblock type
        resblock_class = AMPBlock1  # BigVGAN uses AMPBlock1 by default

        # Upsampling layers
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))

            self.ups.append(
                nn.ModuleList([
                    weight_norm(
                        ConvTranspose1d(
                            in_ch,
                            out_ch,
                            k,
                            u,
                            padding=(k - u) // 2,
                        )
                    )
                ])
            )

            # Noise convolutions for F0 source injection
            stride = math.prod(upsample_rates[i + 1:]) if i + 1 < len(upsample_rates) else 1
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            padding = 0 if stride == 1 else (kernel - stride) // 2

            self.noise_convs.append(
                Conv1d(
                    1,
                    out_ch,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

        # Residual blocks with AMP
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(
                    resblock_class(
                        ch,
                        k,
                        tuple(d),
                        activation=activation,
                        snake_logscale=snake_logscale,
                    )
                )

        # Post-activation and convolution
        ch = upsample_initial_channel // (2 ** len(self.ups))
        if activation == "snake":
            activation_post = Snake(ch, alpha_logscale=snake_logscale)
        else:
            activation_post = SnakeBeta(ch, alpha_logscale=snake_logscale)

        self.activation_post = Activation1d(activation=activation_post)
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3, bias=False))

        # Weight initialization
        for ups in self.ups:
            ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        # Speaker conditioning
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Latent features [B, initial_channel, T]
            f0: Fundamental frequency [B, T']
            g: Speaker embedding [B, gin_channels, 1] (optional)

        Returns:
            Audio waveform [B, 1, T * upp]
        """
        # Generate harmonic source from F0
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)  # [B, 1, T*upp]

        # Pre-convolution
        x = self.conv_pre(x)

        # Add speaker conditioning
        if g is not None:
            x = x + self.cond(g)

        # Upsampling with AMP blocks
        for i in range(self.num_upsamples):
            # Upsampling
            if self.training and self.checkpointing:
                x = checkpoint(self.ups[i][0], x, use_reentrant=False)
            else:
                x = self.ups[i][0](x)

            # Add harmonic source (with size matching to handle rounding differences)
            x_har = self.noise_convs[i](har_source)
            # Match dimensions - truncate the longer one to the shorter
            min_len = min(x.size(2), x_har.size(2))
            x = x[:, :, :min_len] + x_har[:, :, :min_len]

            # Apply AMP blocks
            xs = None
            for j in range(self.num_kernels):
                block_idx = i * self.num_kernels + j
                if self.training and self.checkpointing:
                    if xs is None:
                        xs = checkpoint(
                            self.resblocks[block_idx], x, use_reentrant=False
                        )
                    else:
                        xs = xs + checkpoint(
                            self.resblocks[block_idx], x, use_reentrant=False
                        )
                else:
                    if xs is None:
                        xs = self.resblocks[block_idx](x)
                    else:
                        xs = xs + self.resblocks[block_idx](x)
            x = xs / self.num_kernels

        # Post-processing
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        try:
            print("Removing weight norm from BigVGAN...")
            for ups in self.ups:
                for layer in ups:
                    remove_weight_norm(layer)
            for resblock in self.resblocks:
                resblock.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Weight norm already removed.")

    def __prepare_scriptable__(self):
        """Prepare model for TorchScript."""
        for ups in self.ups:
            for layer in ups:
                for hook in layer._forward_pre_hooks.values():
                    if getattr(hook, "__class__", None).__name__ == "WeightNorm":
                        remove_weight_norm(layer)
        for resblock in self.resblocks:
            resblock.remove_weight_norm()
        return self
