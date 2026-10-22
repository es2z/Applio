"""APEX-GAN generator: an iSTFT decoder whose F0 enters only as a phase reference.

Ported from rvc/lib/algorithm/generators/apex_gan.py of the Codename RVC fork at commit
ef613f900d0b646123081b331332996b92cd52e5 (https://github.com/DanielNawa/codename-rvc-fork-4,
MIT, docs/licenses/codename-rvc-fork-LICENSE). That file is unchanged at the fork's latest
main. APEX: Adaptive harmonics, Pyramid injection, complex EXcitation.

Like CodenameRingFormer, the decoder does not produce samples. Two transposed convolutions
take the latent to the STFT frame rate - at 48 kHz [12, 10], the first two stages of the
stock HiFi-GAN decoder, which leaves it at 12 kHz - and conv_post emits a magnitude and a
phase spectrum that an iSTFT (n_fft 32, hop 4) turns into the waveform. 36 latent frames
become 4321 STFT frames and (4321 - 1) * 4 = 17280 = segment_size samples.

The excitation is split in two, and that split is the point of the architecture:

* The voiced sine is normalised to unit peak, STFT'd, and only its phase is kept. Each
  stage's PhaseProjector turns that phase into a unit-norm (cos beta, sin beta) pair, which
  rotates the argument of the first Snake in every residual layer:
  x + sin(alpha x + beta)^2 / alpha. Unit norm means this path can say *where* in the cycle
  a frame is but cannot carry amplitude, so the harmonic content has to come from the
  latent rather than being explained by F0 alone.
* The noise (a 0.003 floor where voiced, full strength where unvoiced) is STFT'd and its
  [magnitude, phase x envelope] is added at every stage through noise_convs, as genuine
  aperiodic energy the latent could not cheaply produce.

Reproduced verbatim, including what is easy to "fix" by accident:

* The noise envelope is normalised by its mean over the *whole batch*, not per item.
* excitation_synthesizer.l_linear exists and is never called: upstream's combined
  sine + noise path is commented out. It is kept so this decoder's state_dict is key for key
  upstream's, and frozen (requires_grad False), which changes nothing about the function -
  upstream never gives it a gradient either - but keeps DistributedDataParallel, which this
  repository runs without find_unused_parameters, from stopping on a parameter that never
  receives one.
* No init_weights: the transposed convolutions, conv_pre, conv_post, noise_convs and the
  phase projectors keep PyTorch's default initialisation; only the residual convolutions
  are re-initialised, with the Snake-aware Kaiming normal for alpha = 1.

Deliberate differences, none of which change the function or the state_dict:

* Snake is plain PyTorch (rvc/lib/algorithm/generators/apex_gan_snake.py) rather than
  upstream's Triton kernel.
* TorchSTFT is rvc/lib/algorithm/conformer/stft.py's, the same transform and window as
  upstream's TorchSTFT with the window as a non-persistent buffer and fp32 arithmetic.
* The SiLUs are not in place, and autocast is disabled for the input's own device type
  rather than for "cuda" alone.
* `checkpointing` is honoured for the transposed convolutions and the residual blocks, the
  way the other decoders here honour it. Upstream reserves the argument and ignores it.
* Upstream's unused helper modules (FusedGeoSaw, FusedDirichlet, PchipF0Upsampler,
  SnakeBeta, the convolutional STFT) are not ported.

Only 48 kHz is supported (APEX_GAN_ISTFT): upstream's 32 and 40 kHz configs keep HiFi-GAN's
four upsampling stages, so their output would be four times the segment.
"""

import math
from itertools import chain
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.conformer.stft import TorchSTFT
from rvc.lib.algorithm.generators.apex_gan_snake import Snake, snake_kaiming_normal_

# (gen_istft_n_fft, gen_istft_hop_size) per sample rate. rvc/configs/apex_gan/ holds the
# same values for the training config; this is the fallback for a checkpoint that did not
# record them, and the n_fft is also how a checkpoint's sample rate is recognised
# (conv_post emits n_fft + 2 channels).
APEX_GAN_ISTFT = {48000: (32, 4)}


def default_istft_settings(sample_rate: int):
    """(gen_istft_n_fft, gen_istft_hop_size) of the APEX-GAN decoder at this sample rate."""
    try:
        return APEX_GAN_ISTFT[int(sample_rate)]
    except KeyError:
        supported = ", ".join(str(rate) for rate in APEX_GAN_ISTFT)
        raise ValueError(
            f"APEX-GAN has no decoder config for {sample_rate} Hz (supported: {supported})"
        ) from None


def get_conv1d_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def apply_mask(tensor: torch.Tensor, mask: Optional[torch.Tensor]):
    return tensor * mask if mask is not None else tensor


def remove_weight_norm_legacy_safe(module: nn.Module):
    """Fold weight norm away whichever API applied it."""
    if is_parametrized(module, "weight"):
        remove_parametrizations(module, "weight", leave_parametrized=True)
    else:
        remove_weight_norm(module)


def create_resblock_conv1d_layer(channels: int, kernel_size: int, dilation: int):
    conv = Conv1d(
        channels,
        channels,
        kernel_size,
        dilation=dilation,
        padding=get_conv1d_padding(kernel_size, dilation),
    )
    # The Snake in front of it starts at alpha = 1.0.
    snake_kaiming_normal_(conv.weight, kind=1.0)
    return weight_norm(conv)


class PhaseProjector(nn.Module):
    """STFT phase of the pure voiced sine -> one stage's unit-norm (cos beta, sin beta).

    Strided down to the stage's frame rate. The output is renormalised every time, so this
    module can only ever encode a rotation, never an amplitude.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        stride = max(int(stride), 1)
        kernel_size = stride * 2 if stride > 1 else 1
        padding = (stride + 1) // 2 if stride > 1 else 0
        self.proj = Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, sine_phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.proj(sine_phase)
        cos_raw, sin_raw = out.chunk(2, dim=1)
        norm = torch.sqrt(cos_raw * cos_raw + sin_raw * sin_raw + 1e-6)
        return cos_raw / norm, sin_raw / norm


class ResBlock(nn.Module):
    """HiFi-GAN's residual block with Snakes, the first of each layer phase-conditioned.

    convs1 / convs2 are name for name and shape for shape this repository's ResBlock
    (rvc/lib/algorithm/residuals.py). Only the first Snake of each layer takes beta; the
    second is left plain, upstream's choice to keep the F0 control surface small.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))
        self.snakes1 = nn.ModuleList([Snake(channels, init=1.0) for _ in dilations])
        self.snakes2 = nn.ModuleList([Snake(channels, init=1.0) for _ in dilations])

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations):
        return nn.ModuleList(
            [create_resblock_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None, beta=None):
        for conv1, conv2, s1, s2 in zip(
            self.convs1, self.convs2, self.snakes1, self.snakes2
        ):
            x_residual = x
            xt = apply_mask(s1(x, beta=beta), x_mask)
            xt = conv1(xt)
            xt = apply_mask(s2(xt), x_mask)
            xt = conv2(xt)
            x = apply_mask(xt + x_residual, x_mask)
        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm_legacy_safe(conv)


class SineGenerator(nn.Module):
    """The fundamental only, with its phase accumulated in fp32 frame by frame."""

    def __init__(self, sampling_rate: int, sine_amplitude: float = 0.1):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.sine_amplitude = sine_amplitude

    def _generate_sine_wave(self, f0: torch.Tensor, upsampling_factor: int):
        batch_size, length, _ = f0.shape
        upsampling_grid = torch.arange(
            1, upsampling_factor + 1, dtype=f0.dtype, device=f0.device
        )
        phase = (f0 / self.sampling_rate) * upsampling_grid

        # accumulate in fp32 for numerical stability
        phase_remainder = torch.fmod(phase[:, :-1, -1:].float() + 0.5, 1.0) - 0.5
        cumulative_phase = phase_remainder.cumsum(dim=1).fmod(1.0).to(f0.dtype)

        phase += F.pad(cumulative_phase, (0, 0, 1, 0))
        phase = phase.reshape(batch_size, -1, 1)
        phase.remainder_(1.0)
        phase.mul_(2 * math.pi)
        torch.sin(phase, out=phase)
        return phase

    def forward(self, f0: torch.Tensor, upsampling_factor: int):
        """f0: [B, T] in Hz -> (sine [B, T * factor, 1], voiced mask [B, T * factor, 1])."""
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine = self._generate_sine_wave(f0, upsampling_factor)
            sine.mul_(self.sine_amplitude)

            # A frame is either voiced (f0 > 0) or not, so the mask is just repeated.
            voiced = (f0 > 0.0).float().transpose(2, 1)
            voiced = F.interpolate(voiced, scale_factor=upsampling_factor, mode="nearest")
            voiced = voiced.transpose(2, 1)
            sine.mul_(voiced)
        return sine, voiced


class ExcitationSynthesizer(nn.Module):
    """F0 -> (pure voiced sine, pure noise), each [B, 1, T_audio].

    The sine is only ever used for its STFT phase; the noise is a 0.003 floor under voiced
    frames and sine_amp / 3 under unvoiced ones.
    """

    def __init__(
        self,
        sample_rate: int,
        hop_length: int = 480,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.sine_amp = sine_amp
        self.add_noise_std = add_noise_std
        self.sine_gen = SineGenerator(sample_rate, sine_amp)
        # Never called (see the module docstring); kept for upstream's state_dict.
        self.l_linear = nn.Linear(1, 1)
        self.l_linear.requires_grad_(False)
        self.l_tanh = nn.Tanh()

    def forward(self, f0: torch.Tensor, upsample_factor: Optional[int] = None):
        hop = upsample_factor if upsample_factor is not None else self.hop_length
        with torch.amp.autocast(device_type=f0.device.type, enabled=False):
            f0 = f0.float()
            harmonic_signal, voiced_mask = self.sine_gen(f0, hop)
            noise_amp = voiced_mask * self.add_noise_std + (1.0 - voiced_mask) * (
                self.sine_amp / 3.0
            )
            noise = noise_amp * torch.randn_like(harmonic_signal)
            sine_only = harmonic_signal.to(dtype=f0.dtype)
            noise_only = noise.to(dtype=f0.dtype)
        return sine_only.transpose(1, 2), noise_only.transpose(1, 2)


class APEXGANGenerator(nn.Module):
    """iSTFT decoder with phase-only F0 control and additive aperiodic energy.

    Args:
        initial_channel: width of the latent coming out of the flow.
        resblock_kernel_sizes: kernel sizes of the residual blocks, one per block.
        resblock_dilation_sizes: dilations of the residual blocks, one list per block.
        upsample_rates: frame-rate multiplier of each stage, e.g. [12, 10].
        upsample_initial_channel: width after conv_pre.
        upsample_kernel_sizes: transposed convolution width of each stage, e.g. [24, 20].
        gin_channels: speaker conditioning width; 0 disables it.
        sr: waveform sample rate.
        gen_istft_n_fft: FFT size of the output iSTFT and of the excitation's STFT.
        gen_istft_hop_size: hop of both. prod(upsample_rates) * this must equal the
            config's hop_length, or the output will not be segment_size long.
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
        gen_istft_n_fft: int = 32,
        gen_istft_hop_size: int = 4,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.total_ups_factor = math.prod(upsample_rates)
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        # Samples per latent frame: the transposed convolutions times the iSTFT hop.
        self.full_hop = self.total_ups_factor * gen_istft_hop_size
        self.checkpointing = checkpointing

        # Built in upstream's order, so the same seed initialises the same weights.
        self.excitation_synthesizer = ExcitationSynthesizer(
            sample_rate=sr,
            hop_length=self.full_hop,
            sine_amp=0.1,
            add_noise_std=0.003,
        )

        self.conv_pre = weight_norm(
            Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )

        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.phase_convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        # The excitation's STFT has n_fft // 2 + 1 bins; noise_convs read magnitude and
        # phase together, phase_convs the phase alone.
        stft_bins = gen_istft_n_fft // 2 + 1
        noise_channels = gen_istft_n_fft + 2

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2**i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(ConvTranspose1d(in_ch, out_ch, k, u, padding=(k - u) // 2))
            )

            # Brings the excitation's STFT frames down to this stage's frame rate; the
            # last stage already runs at it.
            stride_f0 = (
                math.prod(upsample_rates[i + 1 :]) if i + 1 < self.num_upsamples else 1
            )
            if stride_f0 > 1:
                self.noise_convs.append(
                    Conv1d(
                        noise_channels,
                        out_ch,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2,
                        bias=False,
                    )
                )
            else:
                self.noise_convs.append(
                    Conv1d(noise_channels, out_ch, kernel_size=1, bias=False)
                )
            self.phase_convs.append(PhaseProjector(stft_bins, out_ch, stride_f0))

            for kk, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(out_ch, kk, d))

        final_ch = upsample_initial_channel // (2**self.num_upsamples)
        # [:n_fft // 2 + 1] becomes the magnitude (exp), the rest the phase (sin).
        self.conv_post = weight_norm(
            Conv1d(final_ch, gen_istft_n_fft + 2, 7, 1, padding=3)
        )
        self.reflection_pad = nn.ReflectionPad1d((1, 0))

        self.cond = (
            Conv1d(gin_channels, upsample_initial_channel, 1) if gin_channels != 0 else None
        )

        self.stft = TorchSTFT(
            filter_length=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft,
        )

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        envelope_reference: Optional[float] = None,
    ) -> torch.Tensor:
        """x [B, C, T], f0 [B, T] in Hz, g [B, gin, 1] -> waveform [B, 1, T * full_hop].

        envelope_reference is a realtime override (rvc/realtime/synthesis.py); training
        and offline conversion do not pass it and get exactly the function upstream
        defines.
        """
        sine_only, noise_only = self.excitation_synthesizer(f0)

        # Pure voiced sine at unit peak -> STFT -> keep the phase, discard the magnitude.
        peak = sine_only.abs().amax(dim=-1, keepdim=True)
        sine_unit = sine_only / (peak + 1e-8)
        _, sine_phase = self.stft.transform(sine_unit.squeeze(1).float())
        sine_phase = sine_phase.to(x.dtype)

        # Pure noise -> STFT -> magnitude, and phase scaled by the magnitude envelope so the
        # injection sees both the frame-to-frame decorrelation and the V/UV amplitude.
        noise_mag, noise_phase = self.stft.transform(noise_only.squeeze(1).float())
        noise_mag = noise_mag.to(x.dtype)
        noise_phase = noise_phase.to(x.dtype)
        envelope = noise_mag.mean(dim=1, keepdim=True)
        if envelope_reference is None:
            envelope = envelope / (envelope.mean() + 1e-8)
        else:
            # A fixed reference instead of this input's own mean, which a realtime
            # window of gap-filled F0 (all voiced) drives far from training's batches.
            envelope = envelope / (envelope_reference + 1e-8)
        har_noise = torch.cat([noise_mag, noise_phase * envelope], dim=1)

        x = self.conv_pre(x)
        if g is not None and self.cond is not None:
            x = x + self.cond(g)

        use_checkpoint = self.training and self.checkpointing
        for i in range(self.num_upsamples):
            x = F.silu(x)
            if use_checkpoint:
                x = checkpoint(self.ups[i], x, use_reentrant=False)
            else:
                x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # Aperiodic energy, added; the F0 phase, only ever as a rotation.
            x = x + self.noise_convs[i](har_noise)
            beta = self.phase_convs[i](sine_phase)

            xs = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j]
                if use_checkpoint:
                    out = checkpoint(block, x, None, beta, use_reentrant=False)
                else:
                    out = block(x, beta=beta)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels

        x = F.silu(x)
        x = self.conv_post(x)

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            n_fft = self.gen_istft_n_fft
            x = x.float()
            spec = torch.exp(x[:, : n_fft // 2 + 1, :])
            phase = torch.sin(x[:, n_fft // 2 + 1 :, :])
            return self.stft.inverse(spec, phase)

    @torch.no_grad()
    def noise_envelope_reference(self, voiced_ratio: float, seconds: float = 4.0):
        """The window-mean envelope this decoder would see on an input whose frames are
        voiced in the given ratio - what a training batch with that voicing gives.

        The envelope is linear in the noise level, so its mean per unit of noise is
        measured once on fixed-seed white noise and scaled by the voiced (add_noise_std)
        and unvoiced (sine_amp / 3) levels ExcitationSynthesizer uses.
        """
        synth = self.excitation_synthesizer
        generator = torch.Generator(device="cpu").manual_seed(0)
        samples = int(seconds * synth.sample_rate)
        unit = torch.randn(1, samples, generator=generator).to(
            device=self.stft.window.device
        )
        magnitude, _ = self.stft.transform(unit)
        per_unit = float(magnitude.mean(dim=1).mean())
        ratio = min(max(float(voiced_ratio), 0.0), 1.0)
        level = ratio * synth.add_noise_std + (1.0 - ratio) * (synth.sine_amp / 3.0)
        return per_unit * level

    def remove_weight_norm(self):
        remove_weight_norm_legacy_safe(self.conv_pre)
        remove_weight_norm_legacy_safe(self.conv_post)
        for layer in self.ups:
            remove_weight_norm_legacy_safe(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
