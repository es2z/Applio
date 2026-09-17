"""CodenameRingFormer generator: a Conformer decoder that synthesises spectra, not samples.

Ported from rvc/lib/algorithm/generators/ringformer.py of the Codename RVC fork
(https://github.com/duringleaves/codename-rvc-fork-4).

Unlike every other vocoder here, the decoder does not produce a waveform. It produces a
magnitude and a phase spectrum and an iSTFT turns those into samples, so its upsampling
chain runs in the *STFT frame* domain: at 48 kHz two stages of x4 take 36 latent frames to
576 iSTFT frames of 30 samples each, and the whole segment is 36 * 480 = 17280 samples,
the same as every other vocoder. A Conformer - self attention plus a convolution module -
sits in front of each upsampling stage, which is what the architecture is named after.

The harmonic source is injected as a spectrum too: the sine excitation is STFT'd and its
magnitude and phase are concatenated into `gen_istft_n_fft + 2` channels before
`noise_convs` shrinks them to each stage's frame rate. That is why none of HiFi-GAN's
`noise_convs` can be inherited here (they take a single waveform channel), while
`conv_pre` and `cond` can - see rvc/train/warm_start.py.

Deliberate differences from the upstream file, none of which change the state_dict:

* `conv_post`'s input width is computed from the upsampling chain instead of the literal
  128, so a non-stock `upsample_initial_channel` or a third stage would not silently
  mismatch. With the stock 512 / [4, 4] it is 128 either way.
* `SineGen`'s `apply_modulo` / `early_modulo` / `double_precision` branches are dropped.
  They are constructor options that the generator never enables, so the remaining path is
  exactly what upstream runs.
* TorchSTFT no longer takes a device (see rvc/lib/algorithm/conformer/stft.py).
* The Snake activation is the plain PyTorch form of upstream's fused Triton kernel, with
  the same parameter name and shape (rvc/lib/algorithm/conformer/snake.py).
"""

import math
from typing import Optional

import numpy as np
import torch
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import init_weights
from rvc.lib.algorithm.conformer.conformer import Conformer
from rvc.lib.algorithm.conformer.stft import TorchSTFT
from rvc.lib.algorithm.residuals import SnakeResBlock

# The upsampling chain and the iSTFT hop always multiply out to the config's hop_length,
# so the decoder's output is exactly segment_size. Every stock rate satisfies
# hop_length = 16 * gen_istft_hop_size and gen_istft_n_fft = 4 * gen_istft_hop_size.
DEFAULT_HARMONIC_NUM = 8


def default_istft_settings(sample_rate: int):
    """(gen_istft_n_fft, gen_istft_hop_size) implied by a stock sample rate.

    Every stock config has hop_length = sample_rate / 100 = prod(upsample_rates) *
    gen_istft_hop_size with upsample_rates [4, 4], and gen_istft_n_fft = 4 *
    gen_istft_hop_size. So a caller holding only the rate - a checkpoint written before
    those two keys existed, say - can recover both.
    """
    hop = sample_rate // 1600
    return hop * 4, hop


class SineGen(torch.nn.Module):
    """Sine excitation for the fundamental and its overtones.

    Args:
        samp_rate: waveform sample rate in Hz.
        upsample_scale: samples per F0 frame, i.e. prod(upsample_rates) * istft hop.
        harmonic_num: number of overtones above the fundamental.
        sine_amp: amplitude of the sine components.
        noise_std: standard deviation of the additive noise on voiced frames.
        voiced_threshold: F0 at or below which a frame counts as unvoiced.
    """

    def __init__(
        self,
        samp_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        return (f0 > self.voiced_threshold).to(torch.float32)

    def _f02sine(self, f0_values: torch.Tensor) -> torch.Tensor:
        """f0_values: [B, T, dim] in Hz, already upsampled to the waveform rate."""
        # Cycles per sample. The integer part does not affect the phase.
        rad_values = (f0_values / self.sampling_rate) % 1

        # Random initial phase for the overtones, none for the fundamental.
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # Accumulate the phase at the frame rate and interpolate back up, which keeps the
        # cumulative sum short enough not to lose precision over a long utterance.
        rad_values = torch.nn.functional.interpolate(
            rad_values.transpose(1, 2),
            scale_factor=1 / self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
        phase = torch.nn.functional.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        return torch.sin(phase)

    def forward(self, f0: torch.Tensor):
        """f0: [B, T, 1] in Hz, 0 where unvoiced -> ([B, T, dim], [B, T, 1], [B, T, dim])."""
        with torch.no_grad():
            # bf16 loses the phase in the cumulative sum, so this stays in fp32.
            with torch.amp.autocast(device_type=f0.device.type, enabled=False):
                harmonics = torch.arange(
                    1, self.harmonic_num + 2, device=f0.device, dtype=torch.float32
                )
                sine_waves = (
                    self._f02sine(f0.float() * harmonics.view(1, 1, -1)) * self.sine_amp
                )

                uv = self._f02uv(f0)
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
                noise = noise_amp * torch.randn_like(sine_waves)
                sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """Merges the harmonic sine bank into a single excitation.

    Distinct from rvc.lib.algorithm.generators.hifigan_nsf.SourceModuleHnNSF: that one
    generates one sine per waveform sample with harmonic_num 0, this one generates
    `harmonic_num + 1` of them at the iSTFT frame rate, so l_linear is Linear(9, 1) rather
    than Linear(1, 1) and the two cannot be inherited from one another.
    """

    def __init__(
        self,
        sample_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(
            samp_rate=sample_rate,
            upsample_scale=upsample_scale,
            harmonic_num=harmonic_num,
            sine_amp=sine_amp,
            noise_std=add_noise_std,
            voiced_threshold=voiced_threshold,
        )
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            with torch.no_grad():
                sine_wavs, uv, _ = self.l_sin_gen(x)
            sine_merge = self.l_tanh(self.l_linear(sine_wavs.float()))
            noise = torch.randn_like(uv) * self.sine_amp / 3
            return sine_merge, noise, uv


class CodenameRingFormerGenerator(torch.nn.Module):
    """Conformer + iSTFT decoder.

    Args:
        initial_channel: width of the latent coming out of the flow.
        resblock_kernel_sizes: kernel sizes of the residual blocks, one per block.
        resblock_dilation_sizes: dilations of the residual blocks, one list per block.
        upsample_rates: frame-rate multiplier of each stage, e.g. [4, 4].
        upsample_initial_channel: width after conv_pre.
        upsample_kernel_sizes: transposed convolution width of each stage, e.g. [8, 8].
        gin_channels: speaker conditioning width; 0 disables it.
        sr: waveform sample rate.
        gen_istft_n_fft: FFT size of the output iSTFT.
        gen_istft_hop_size: hop of the output iSTFT. prod(upsample_rates) * this must
            equal the config's hop_length, or the decoder's output will not be
            segment_size long.
        harmonic_num: overtones in the sine excitation.
        checkpointing: trade compute for memory in the upsampling stages.
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
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
        harmonic_num: int = DEFAULT_HARMONIC_NUM,
        conformer_depth: int = 2,
        conformer_dim_head: int = 64,
        conformer_heads: int = 8,
        conformer_ff_mult: int = 4,
        conformer_conv_expansion_factor: int = 2,
        conformer_conv_kernel_size: int = 31,
        conformer_block_size: int = 512,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        self.post_n_fft = gen_istft_n_fft
        self.checkpointing = checkpointing

        self.conv_pre = weight_norm(
            Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )

        upsample_scale = math.prod(upsample_rates) * gen_istft_hop_size
        self.m_source = SourceModuleHnNSF(
            sample_rate=sr,
            upsample_scale=upsample_scale,
            harmonic_num=harmonic_num,
            voiced_threshold=0,
        )
        self.f0_upsamp = torch.nn.Upsample(scale_factor=upsample_scale)

        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()
        self.noise_res = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            channels = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                kernel = stride_f0 * 2 - stride_f0 % 2
                padding = 0 if stride_f0 == 1 else (kernel - stride_f0) // 2
                self.noise_convs.append(
                    Conv1d(
                        gen_istft_n_fft + 2,
                        channels,
                        kernel_size=kernel,
                        stride=stride_f0,
                        padding=padding,
                    )
                )
                self.noise_res.append(SnakeResBlock(channels, 7, [1, 3, 5]))
            else:
                self.noise_convs.append(
                    Conv1d(gen_istft_n_fft + 2, channels, kernel_size=1)
                )
                self.noise_res.append(SnakeResBlock(channels, 11, [1, 3, 5]))

        # One Snake gain per stage input plus one for the final activation.
        self.alphas = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(1, upsample_initial_channel, 1))]
        )
        self.resblocks = torch.nn.ModuleList()
        for i in range(len(self.ups)):
            channels = upsample_initial_channel // (2 ** (i + 1))
            self.alphas.append(torch.nn.Parameter(torch.ones(1, channels, 1)))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(SnakeResBlock(channels, k, d))

        self.conformers = torch.nn.ModuleList(
            [
                Conformer(
                    dim=upsample_initial_channel // (2**i),
                    depth=conformer_depth,
                    dim_head=conformer_dim_head,
                    heads=conformer_heads,
                    ff_mult=conformer_ff_mult,
                    conv_expansion_factor=conformer_conv_expansion_factor,
                    conv_kernel_size=conformer_conv_kernel_size,
                    block_size=conformer_block_size,
                )
                for i in range(len(self.ups))
            ]
        )

        self.conv_post = weight_norm(
            Conv1d(
                upsample_initial_channel // (2 ** len(upsample_rates)),
                self.post_n_fft + 2,
                7,
                1,
                padding=3,
            )
        )
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))

        self.stft = TorchSTFT(
            filter_length=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft,
        )

        self.cond = (
            Conv1d(gin_channels, upsample_initial_channel, 1)
            if gin_channels != 0
            else None
        )

    def _snake(self, x: torch.Tensor, index: int) -> torch.Tensor:
        alpha = self.alphas[index].to(dtype=x.dtype)
        return x + (1 / alpha) * (torch.sin(alpha * x) ** 2)

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        """Returns (waveform [B, 1, T], magnitude [B, F, N], phase [B, F, N])."""
        if f0.dim() == 1:
            f0 = f0.unsqueeze(0)

        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, T * upsample_scale, 1]
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)  # [B, T * upsample_scale]

        # The excitation enters as a spectrum rather than as samples.
        har_spec, har_phase = self.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)

        x = self.conv_pre(x)
        if g is not None and self.cond is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = self._snake(x, i)
            x = self.conformers[i](x.transpose(1, 2)).transpose(1, 2)

            x_source = self.noise_res[i](self.noise_convs[i](har))

            if self.training and self.checkpointing:
                x = checkpoint(self.ups[i], x, use_reentrant=False)
            else:
                x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source

            xs = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j]
                if self.training and self.checkpointing:
                    current = checkpoint(block, x, use_reentrant=False)
                else:
                    current = block(x)
                xs = current if xs is None else xs + current
            x = xs / self.num_kernels

        x = self._snake(x, self.num_upsamples)
        x = self.conv_post(x)

        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return self.stft.inverse(spec, phase), spec, phase

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for layer in self.ups:
            remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
        for block in self.noise_res:
            block.remove_weight_norm()
