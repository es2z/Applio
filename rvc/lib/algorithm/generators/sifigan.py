"""SiFi-GAN generator: a source-filter decoder with pitch-dependent convolutions.

Ported from the official implementation (MIT, Reo Yoneyama / Nagoya University):
    https://github.com/chomeyama/SiFiGAN
    - sifigan/models/generator.py  (SiFiGANGenerator)
    - sifigan/layers/residual_block.py  (AdaptiveResidualBlock)
    - sifigan/utils/index.py  (pd_indexing)
    - sifigan/utils/features.py  (dilated_factor)

The generator is split in two. The *source network* (`sn`) turns a sine excitation into
an excitation signal using quasi-periodic convolutions whose dilation follows the pitch,
and the *filter network* (`fn`) shapes it into the waveform. Both are driven by the same
latent `x`.

Two deliberate differences from the official code, both so that a warm start from an
existing HiFi-GAN decoder can carry as much as possible (see rvc/train/warm_start.py):

1. The official code asserts `upsample_kernel_sizes[i] == 2 * upsample_scales[i]`. We use
   this repository's odd-rate padding instead (the same expression as
   HiFiGANNSFGenerator), so the stock `upsample_kernel_sizes` work unchanged at 32k
   ([20,16,4,4]), 40k ([16,16,4,4]) and 48k ([24,20,4,4]). 40k does not satisfy the
   official assertion and would otherwise be unusable.
2. `fn.upsamples` is a bare weight-normed ConvTranspose1d with the LeakyReLU applied in
   forward, rather than the official `Sequential(LeakyReLU, ConvTranspose1d)`. The maths
   is identical and the parameter names then line up one-for-one with `dec.ups.{i}`.

With `filter_resblock="rvc"` (the default) the filter network is structurally identical
to this repository's HiFi-GAN decoder, so `conv_pre`, `cond`, `m_source`, `fn.upsamples`,
`fn.blocks` and `fn.output_conv` all inherit from a HiFi-GAN or RefineGAN pretrain and
only the source network starts from scratch. With `filter_resblock="official"` the
filter blocks follow the paper (no second convolution, kernel sizes 3/5/7) and cannot be
inherited.
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import get_padding, init_weights
from rvc.lib.algorithm.generators.hifigan_nsf import SourceModuleHnNSF
from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock

# Official filter-network settings, used only when filter_resblock="official".
OFFICIAL_FILTER_KERNEL_SIZES = (3, 5, 7)
OFFICIAL_FILTER_DILATIONS = ((1, 3, 5), (1, 3, 5), (1, 3, 5))

# "Number of taps in one cycle" per upsampling stage, from the official 24 kHz config.
DEFAULT_DENSE_FACTORS = (0.5, 1.0, 4.0, 8.0)

# Initial gain on the source network's contribution to the filter network. Not part of
# the official implementation, which is equivalent to 1.0. Chosen by measurement: mel L1
# on 16 real clips before any optimizer step, warm started from the stock f0G48k, all
# configurations built from the same RNG state.
#   gain  1.00 -> 2.876   0.30 -> 1.946   0.10 -> 1.245   0.03 -> 0.999   0.00 -> 1.069
#   (from scratch, for reference: 1.83 - 1.95 at every gain)
# The curve is flat between 0 and 0.1 but 0.03 is clearly better than either end, so a
# little source signal helps and a lot of it hurts. See the comment where source_scales
# is created for why.
DEFAULT_SOURCE_SCALE_INIT = 0.03


def dilated_factor(
    f0: torch.Tensor, sample_rate: int, dense_factor: float
) -> torch.Tensor:
    """Pitch-dependent dilation factors, one per frame.

    Mirrors sifigan.utils.features.dilated_factor: unvoiced frames (f0 == 0) are given
    the pitch that makes the factor exactly 1.0, i.e. no adaptation.

    Args:
        f0: [B, T] fundamental frequency in Hz, 0 where unvoiced.
        sample_rate: the waveform sample rate. The official code passes the full rate
            here for every stage rather than the stage's own rate.
        dense_factor: taps per cycle for this stage.

    Returns:
        [B, T] strictly positive factors.
    """
    base = sample_rate / dense_factor
    f0 = torch.where(f0 > 0.0, f0, torch.full_like(f0, base))
    return base / f0


def pd_indexing(
    x: torch.Tensor, d: torch.Tensor, dilation: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather the past and future sample of every position at a pitch-dependent distance.

    Mirrors sifigan.utils.index.pd_indexing, including its reflect-style wrapping. The
    official signature also takes precomputed batch/channel indices but rebuilds them
    internally and ignores the arguments, so they are dropped here.

    Args:
        x: [B, C, T] feature map.
        d: [B, 1, T] dilation factors for this stage.
        dilation: the block's own dilation, multiplied into the factors.

    Returns:
        Two [B, C, T] tensors: the past and the future samples.
    """
    batch, channels, length = x.size()
    batch_index = torch.arange(batch, dtype=torch.long, device=x.device).reshape(
        batch, 1, 1
    )
    channel_index = torch.arange(channels, dtype=torch.long, device=x.device).reshape(
        1, channels, 1
    )
    dilations = torch.clamp((d * dilation).long(), min=1)

    index_base = torch.arange(length, dtype=torch.long, device=x.device).reshape(
        1, 1, length
    )
    # past index (assume reflect padding)
    index_past = (index_base - dilations).abs() % length
    # future index (assume reflect padding)
    index_future = index_base + dilations
    overflowed = index_future >= length
    index_future[overflowed] = -(index_future[overflowed] % length)

    return (
        x[(batch_index, channel_index, index_past)],
        x[(batch_index, channel_index, index_future)],
    )


class AdaptiveResidualBlock(torch.nn.Module):
    """Quasi-periodic residual block: the source network's core.

    Each tap mixes the current sample with the sample one pitch-period-scaled step into
    the past and into the future, so the receptive field follows F0 rather than being
    fixed. Only kernel_size 3 is supported, matching the official implementation.
    """

    def __init__(
        self,
        channels: int,
        dilations: Sequence[int] = (1,),
        kernel_size: int = 3,
        bias: bool = True,
        use_additional_convs: bool = True,
        lrelu_slope: float = LRELU_SLOPE,
    ):
        super().__init__()
        if kernel_size != 3:
            raise ValueError("AdaptiveResidualBlock only supports kernel_size 3.")
        self.channels = channels
        self.dilations = tuple(dilations)
        self.use_additional_convs = use_additional_convs
        self.lrelu_slope = lrelu_slope

        def conv1x1():
            return weight_norm(torch.nn.Conv1d(channels, channels, 1, bias=bias))

        self.convsC = torch.nn.ModuleList([conv1x1() for _ in self.dilations])
        self.convsP = torch.nn.ModuleList([conv1x1() for _ in self.dilations])
        self.convsF = torch.nn.ModuleList([conv1x1() for _ in self.dilations])
        if use_additional_convs:
            self.convsA = torch.nn.ModuleList(
                [
                    weight_norm(
                        torch.nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=1,
                            padding=get_padding(kernel_size, 1),
                            bias=bias,
                        )
                    )
                    for _ in self.dilations
                ]
            )

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        for i, dilation in enumerate(self.dilations):
            xt = F.leaky_relu(x, self.lrelu_slope)
            past, future = pd_indexing(xt, d, dilation)
            xt = self.convsC[i](xt) + self.convsP[i](past) + self.convsF[i](future)
            if self.use_additional_convs:
                xt = self.convsA[i](F.leaky_relu(xt, self.lrelu_slope))
            x = xt + x
        return x

    def remove_weight_norm(self):
        modules = [self.convsC, self.convsP, self.convsF]
        if self.use_additional_convs:
            modules.append(self.convsA)
        for module_list in modules:
            for conv in module_list:
                remove_weight_norm(conv)


class OfficialResidualBlock(torch.nn.Module):
    """The paper's filter-network residual block: one convolution per dilation.

    This repository's ResBlock always has a second, undilated convolution per dilation
    (`convs2`). The official filter network sets use_additional_convs=false, so it has
    only `convs1`. Used when filter_resblock="official".
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 3, 5),
        lrelu_slope: float = LRELU_SLOPE,
    ):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.convs1 = torch.nn.ModuleList(
            [
                weight_norm(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilations
            ]
        )
        self.convs1.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs1:
            x = conv(F.leaky_relu(x, self.lrelu_slope)) + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs1:
            remove_weight_norm(conv)


def _upsample_padding(rate: int, kernel_size: int) -> int:
    """Padding that makes ConvTranspose1d exactly length-preserving up to `rate`.

    The same expression as HiFiGANNSFGenerator, so the transposed convolutions have
    identical shapes to `dec.ups` for every stock config.
    """
    if rate % 2 == 0:
        return (kernel_size - rate) // 2
    return rate // 2 + rate % 2


def _downsample_padding(rate: int, kernel_size: int) -> int:
    """Padding that makes a strided Conv1d divide the length by exactly `rate`.

    The official expression (`rate - (kernel_size % 2 == 0)`) assumes kernel == 2 * rate
    and is off by one for the stock 40k kernels, so it is generalised to
    ceil((kernel_size - rate) / 2), which satisfies the exactness condition
    kernel - rate <= 2 * padding < kernel for both.
    """
    return (kernel_size - rate + 1) // 2


class SiFiGANGenerator(torch.nn.Module):
    """SiFi-GAN decoder, called like this repository's other pitch-guided generators.

    Args:
        initial_channel: channels of the latent z (inter_channels). Independent of the
            embedder width, which only ever reaches enc_p.emb_phone.
        resblock_kernel_sizes / resblock_dilation_sizes: filter-network residual block
            settings, used when filter_resblock="rvc".
        upsample_rates / upsample_initial_channel / upsample_kernel_sizes: as HiFi-GAN.
        gin_channels: speaker conditioning width; 0 disables `cond`.
        sr: waveform sample rate, used for the sine source and the dilation factors.
        checkpointing: gradient checkpointing during training.
        filter_resblock: "rvc" (default, inheritable from HiFi-GAN) or "official".
        source_resblock_dilations: per-stage dilations of the adaptive blocks. Defaults
            to the official (1,), (1,2), (1,2,4), (1,2,4,8).
        dense_factors: per-stage taps per cycle. Defaults to the official 0.5, 1, 4, 8.
        source_scale_init: initial value of the learnable per-stage gain on the source
            network's contribution to the filter network. See `source_scales` below.
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
        filter_resblock: str = "rvc",
        source_resblock_dilations: Optional[Sequence[Sequence[int]]] = None,
        dense_factors: Sequence[float] = DEFAULT_DENSE_FACTORS,
        source_scale_init: float = DEFAULT_SOURCE_SCALE_INIT,
    ):
        super(SiFiGANGenerator, self).__init__()

        if filter_resblock not in ("rvc", "official"):
            raise ValueError(
                f"filter_resblock must be 'rvc' or 'official', got {filter_resblock!r}"
            )
        num_upsamples = len(upsample_rates)
        if len(upsample_kernel_sizes) != num_upsamples:
            raise ValueError("upsample_rates and upsample_kernel_sizes differ in length")
        if source_resblock_dilations is None:
            # (1,), (1,2), (1,2,4), (1,2,4,8) for four stages.
            source_resblock_dilations = [
                tuple(2**k for k in range(i + 1)) for i in range(num_upsamples)
            ]
        if len(source_resblock_dilations) != num_upsamples:
            raise ValueError(
                "source_resblock_dilations must have one entry per upsampling stage"
            )
        if len(dense_factors) != num_upsamples:
            raise ValueError(
                f"dense_factors must have one entry per upsampling stage "
                f"({num_upsamples}), got {len(dense_factors)}"
            )

        self.num_upsamples = num_upsamples
        self.checkpointing = checkpointing
        self.filter_resblock = filter_resblock
        self.lrelu_slope = LRELU_SLOPE
        self.sample_rate = sr
        self.upp = math.prod(upsample_rates)
        self.dense_factors = tuple(float(f) for f in dense_factors)
        # Sample offset from the frame rate after each upsampling stage: at 48k with
        # [12,10,2,2] this is [12, 120, 240, 480], the last being the hop length.
        self.prod_upsample_scales = tuple(
            math.prod(upsample_rates[: i + 1]) for i in range(num_upsamples)
        )

        # Sine excitation, identical to the HiFi-GAN NSF decoder so it can be inherited.
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)

        channels = [
            upsample_initial_channel // (2 ** (i + 1)) for i in range(num_upsamples)
        ]
        self.conv_pre = torch.nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        def make_upsample(i):
            rate, kernel = upsample_rates[i], upsample_kernel_sizes[i]
            return weight_norm(
                torch.nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    channels[i],
                    kernel,
                    rate,
                    padding=_upsample_padding(rate, kernel),
                    output_padding=rate % 2,
                )
            )

        def make_downsamples():
            # Built in reverse so that entry j feeds stage (num_upsamples - 1 - j).
            convs = torch.nn.ModuleList()
            for i in reversed(range(num_upsamples)):
                rate, kernel = upsample_rates[i], upsample_kernel_sizes[i]
                convs.append(
                    torch.nn.Conv1d(
                        channels[i],
                        upsample_initial_channel // (2**i),
                        kernel,
                        rate,
                        padding=_downsample_padding(rate, kernel),
                    )
                )
            return convs

        # --- source network -------------------------------------------------------
        self.sn = torch.nn.ModuleDict()
        self.sn["emb"] = torch.nn.Conv1d(1, channels[-1], 7, 1, padding=3)
        self.sn["downsamples"] = make_downsamples()
        self.sn["upsamples"] = torch.nn.ModuleList(
            [make_upsample(i) for i in range(num_upsamples)]
        )
        self.sn["blocks"] = torch.nn.ModuleList(
            [
                AdaptiveResidualBlock(channels[i], source_resblock_dilations[i])
                for i in range(num_upsamples)
            ]
        )
        self.sn["output_conv"] = torch.nn.Conv1d(channels[-1], 1, 7, 1, padding=3)

        # --- filter network -------------------------------------------------------
        self.fn = torch.nn.ModuleDict()
        self.fn["upsamples"] = torch.nn.ModuleList(
            [make_upsample(i) for i in range(num_upsamples)]
        )
        self.fn["downsamples"] = make_downsamples()
        if filter_resblock == "rvc":
            # Identical to HiFiGANNSFGenerator.resblocks: stage outer, kernel inner.
            self.fn["blocks"] = torch.nn.ModuleList(
                [
                    ResBlock(channels[i], k, d)
                    for i in range(num_upsamples)
                    for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
                ]
            )
            self.num_kernels = len(resblock_kernel_sizes)
        else:
            self.fn["blocks"] = torch.nn.ModuleList(
                [
                    OfficialResidualBlock(channels[i], k, d)
                    for i in range(num_upsamples)
                    for k, d in zip(
                        OFFICIAL_FILTER_KERNEL_SIZES, OFFICIAL_FILTER_DILATIONS
                    )
                ]
            )
            self.num_kernels = len(OFFICIAL_FILTER_KERNEL_SIZES)
        # bias=False and plain (not weight-normed), matching dec.conv_post.
        self.fn["output_conv"] = torch.nn.Conv1d(
            channels[-1], 1, 7, 1, padding=3, bias=False
        )

        self.sn["upsamples"].apply(init_weights)
        self.fn["upsamples"].apply(init_weights)

        # Each filter stage computes fn.upsamples[i](c) + source_scales[i] * embs[-i-1].
        # Without the gain that additive term measures 3.0x the upsampled path at the
        # deeper stages, while the HiFi-GAN residual blocks a warm start inherits were
        # trained on an additive term at 0.17-0.43x (their sine-derived noise_convs). The
        # inherited weights then run far out of distribution and the warm start is worth
        # less than nothing - measured at gain 1.0, mel L1 2.876 against 1.907 from
        # scratch. At the default gain it is 0.999, i.e. 48% better than scratch rather
        # than 51% worse. Starting the gain small hands those blocks something close to
        # what they were trained on and lets the model open the gate as the source
        # network becomes meaningful.
        # RefineGAN's AdaIN does the same thing with its 1e-4 initialised weight.
        # The source network is still supervised directly by the regularisation loss, so
        # a small gain does not starve it of gradient.
        self.source_scales = torch.nn.Parameter(
            torch.full((num_upsamples,), float(source_scale_init))
        )

    def _dilated_factors(self, f0: torch.Tensor) -> List[torch.Tensor]:
        """Per-stage dilation factors, each at that stage's time resolution.

        Equivalent to the official collater's
        `np.repeat(dilated_factor(f0, fs, df), prod_upsample_scales[i])`.
        """
        factors = []
        for dense_factor, scale in zip(self.dense_factors, self.prod_upsample_scales):
            factor = dilated_factor(f0, self.sample_rate, dense_factor)
            factors.append(factor.unsqueeze(1).repeat_interleave(scale, dim=2))
        return factors

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: [B, initial_channel, T] latent.
            f0: [B, T] fundamental frequency in Hz at the frame rate, 0 where unvoiced.
            g: [B, gin_channels, 1] speaker conditioning.

        Returns:
            (waveform, source) where waveform is [B, 1, T * prod(upsample_rates)] and
            source is the excitation signal at the same length, for the regularisation
            loss. Lengths line up by construction; a mismatch raises in the skip adds.
        """
        use_checkpoint = self.training and self.checkpointing

        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)  # [B, 1, T * upp]
        dilations = self._dilated_factors(f0)

        c = self.conv_pre(x)
        if g is not None:
            c = c + self.cond(g)
        e = c

        # --- source network: sine -> excitation -----------------------------------
        emb = self.sn["emb"](har_source)
        embs = [emb]
        for i in range(self.num_upsamples - 1):
            emb = F.leaky_relu(self.sn["downsamples"][i](emb), self.lrelu_slope)
            embs.append(emb)

        for i in range(self.num_upsamples):
            e = F.leaky_relu(e, self.lrelu_slope)
            if use_checkpoint:
                e = checkpoint(self.sn["upsamples"][i], e, use_reentrant=False)
                e = e + embs[-i - 1]
                e = checkpoint(
                    self.sn["blocks"][i], e, dilations[i], use_reentrant=False
                )
            else:
                e = self.sn["upsamples"][i](e) + embs[-i - 1]
                e = self.sn["blocks"][i](e, dilations[i])
        source = self.sn["output_conv"](F.leaky_relu(e, self.lrelu_slope))

        # --- filter network: excitation -> waveform -------------------------------
        embs = [e]
        for i in range(self.num_upsamples - 1):
            e = F.leaky_relu(self.fn["downsamples"][i](e), self.lrelu_slope)
            embs.append(e)

        for i in range(self.num_upsamples):
            c = F.leaky_relu(c, self.lrelu_slope)
            scale = self.source_scales[i]
            if use_checkpoint:
                c = checkpoint(self.fn["upsamples"][i], c, use_reentrant=False)
                c = c + embs[-i - 1] * scale
                cs = sum(
                    checkpoint(self.fn["blocks"][i * self.num_kernels + j], c,
                               use_reentrant=False)
                    for j in range(self.num_kernels)
                )
            else:
                c = self.fn["upsamples"][i](c) + embs[-i - 1] * scale
                cs = sum(
                    self.fn["blocks"][i * self.num_kernels + j](c)
                    for j in range(self.num_kernels)
                )
            c = cs / self.num_kernels

        waveform = torch.tanh(
            self.fn["output_conv"](F.leaky_relu(c, self.lrelu_slope))
        )
        return waveform, source

    def remove_weight_norm(self):
        for network in (self.sn, self.fn):
            for layer in network["upsamples"]:
                remove_weight_norm(layer)
            for block in network["blocks"]:
                block.remove_weight_norm()
