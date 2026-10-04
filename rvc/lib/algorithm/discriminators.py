import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE


# The sub-discriminators each version is built from, in order. v2 is what every HiFi-GAN
# model here has always used; v3 is what upstream Applio trains RefineGAN with. Warm
# starting matches sub-discriminators by these descriptors rather than by position,
# because a period 17 and a period 23 discriminator have identical shapes but look at
# different things, and v2 and v3 disagree about what sits at index 6.
DISCRIMINATOR_VERSIONS = {
    "v1": {"periods": [2, 3, 5, 7, 11, 17], "resolutions": []},
    "v2": {"periods": [2, 3, 5, 7, 11, 17, 23, 37], "resolutions": []},
    "v3": {
        "periods": [2, 3, 5, 7, 11],
        "resolutions": [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]],
    },
    # What the Codename RVC fork trains RingFormer against: its MPD_MSD_MRD_Combined is
    # a scale discriminator, eight periods and three STFT resolutions, and its MRD uses a
    # Hann window where upstream Applio's uses a rectangular one. Deliberately not called
    # "v4": v1 to v3 are upstream Applio's names for upstream's layouts, and this is not
    # one of them.
    "codename-ringformer": {
        "periods": [2, 3, 5, 7, 11, 17, 23, 37],
        "resolutions": [[2048, 240, 1200], [4096, 480, 2400], [1024, 100, 480]],
        "window": "hann",
    },
    # What SiFi-GAN+ trains against: v3's periods with the resolution discriminators made
    # for 48 kHz. v3's resolutions are UnivNet's 24 kHz ones, so at 48 kHz each of them
    # sees half the time span and twice the bandwidth per bin it was designed for; these
    # are the same resolutions doubled, which is what codename-ringformer already uses,
    # with the Hann window the official SiFi-GAN discriminator config uses.
    #
    # always_recorded: its weights are shaped exactly like v3's - a DiscriminatorR's
    # shapes depend on neither its resolution nor its window - so the weights alone
    # cannot tell the two apart. It was added after save_checkpoint started recording
    # disc_version, so every checkpoint holding one says so, and one that records nothing
    # is v3.
    "v3-48k": {
        "periods": [2, 3, 5, 7, 11],
        "resolutions": [[2048, 240, 1200], [4096, 480, 2400], [1024, 100, 480]],
        "window": "hann",
        "always_recorded": True,
    },
    # What RingFormer+ trains against: codename-ringformer unchanged, plus the
    # multi-scale sub-band CQT discriminator (MS-SB-CQT, Gu et al.) the RingFormer paper
    # trains against, at RingFormer's and BigVGAN's three scales - [hop_length, octaves,
    # bins per octave], the hop at twice the sample rate. The MRD stays because the CQT
    # stops at ~16.5 kHz, so above that only the MRD looks at a 40 or 48 kHz model, and
    # because keeping every codename-ringformer sub-discriminator makes the CQT the only
    # difference between the two recipes. See rvc/lib/algorithm/cqt_discriminator.py.
    #
    # Not always_recorded: no other version has a CQT discriminator, so its weights alone
    # tell it apart.
    "codename-ringformer-cqt": {
        "periods": [2, 3, 5, 7, 11, 17, 23, 37],
        "resolutions": [[2048, 240, 1200], [4096, 480, 2400], [1024, 100, 480]],
        "window": "hann",
        "cqts": [[512, 9, 24], [256, 9, 36], [256, 9, 48]],
    },
}

# The window every resolution discriminator of a version uses. Absent from v1 to v3, whose
# rectangular window predates the option, so their behaviour is unchanged.
DEFAULT_RESOLUTION_WINDOW = "ones"


def discriminator_layout(version):
    """Descriptors of a version's sub-discriminators, in ModuleList order."""
    spec = DISCRIMINATOR_VERSIONS[version]
    window = spec.get("window", DEFAULT_RESOLUTION_WINDOW)
    layout = (
        [("S",)]
        + [("P", period) for period in spec["periods"]]
        + [("R", tuple(resolution), window) for resolution in spec["resolutions"]]
    )
    if spec.get("cqts"):
        from rvc.lib.algorithm.cqt_discriminator import CQT_FMIN

        layout += [
            ("Q", hop_length, n_octaves, bins_per_octave, CQT_FMIN)
            for hop_length, n_octaves, bins_per_octave in spec["cqts"]
        ]
    return layout


def describe_discriminator(discriminator):
    """The descriptor discriminator_layout would give this sub-discriminator."""
    if isinstance(discriminator, DiscriminatorP):
        return ("P", discriminator.period)
    if isinstance(discriminator, DiscriminatorR):
        # The window belongs in the descriptor: [2048, 240, 1200] appears in both v3 and
        # codename-ringformer, and the two look at a different spectrogram, so a warm
        # start must not treat them as the same discriminator.
        return ("R", tuple(discriminator.resolution), discriminator.window)
    # A DiscriminatorCQT carries its own, so that describing the other versions' layouts
    # never imports it (and torchaudio).
    descriptor = getattr(discriminator, "descriptor", None)
    if descriptor is not None:
        return descriptor
    return ("S",)


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    Multi-period discriminator.

    This class implements a multi-period discriminator, which is used to
    discriminate between real and fake audio signals. The discriminator
    is composed of a series of convolutional layers that are applied to
    the input signal at different periods.

    Args:
        use_spectral_norm (bool): Whether to use spectral normalization.
            Defaults to False.
        version (str): Which set of sub-discriminators to build, see
            DISCRIMINATOR_VERSIONS. Defaults to "v2".
        sample_rate (int): The training sample rate. Only read by, and required for, a
            version with CQT discriminators, whose transform depends on it.
    """

    def __init__(
        self,
        use_spectral_norm: bool = False,
        checkpointing: bool = False,
        version: str = "v2",
        sample_rate: int = None,
    ):
        super().__init__()
        if version not in DISCRIMINATOR_VERSIONS:
            raise ValueError(f"Unknown discriminator version '{version}'")
        self.version = version
        spec = DISCRIMINATOR_VERSIONS[version]
        periods = spec["periods"]
        resolutions = spec["resolutions"]
        window = spec.get("window", DEFAULT_RESOLUTION_WINDOW)
        cqts = spec.get("cqts", [])
        if cqts and sample_rate is None:
            raise ValueError(
                f"Discriminator version '{version}' has CQT discriminators, which need "
                "the sample rate"
            )
        self.checkpointing = checkpointing
        discriminators = (
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
            + [
                DiscriminatorR(r, use_spectral_norm=use_spectral_norm, window=window)
                for r in resolutions
            ]
        )
        if cqts:
            from rvc.lib.algorithm.cqt_discriminator import DiscriminatorCQT

            # Built on a fork of the CPU generator, so building them leaves the global
            # stream exactly where the same layout without them leaves it. With the same
            # seed, a run against codename-ringformer-cqt then initialises every
            # sub-discriminator the two layouts share identically to, and draws the same
            # random numbers afterwards as, a run against codename-ringformer: what makes
            # an A/B between them fair from the first step. Their own initialisation
            # still follows the run's seed, from where the shared ones left the stream.
            with torch.random.fork_rng(devices=[]):
                discriminators += [
                    DiscriminatorCQT(
                        sample_rate,
                        hop_length,
                        n_octaves,
                        bins_per_octave,
                        use_spectral_norm=use_spectral_norm,
                    )
                    for hop_length, n_octaves, bins_per_octave in cqts
                ]
        self.discriminators = torch.nn.ModuleList(discriminators)

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            if self.training and self.checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(d, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """
    Discriminator for the short-term component.

    This class implements a discriminator for the short-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal.
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorP(torch.nn.Module):
    """
    Discriminator for the long-term component.

    This class implements a discriminator for the long-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal at a given
    period.

    Args:
        period (int): Period of the discriminator.
        kernel_size (int): Kernel size of the convolutional layers. Defaults to 5.
        stride (int): Stride of the convolutional layers. Defaults to 3.
        use_spectral_norm (bool): Whether to use spectral normalization. Defaults to False.
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        (kernel_size, 1),
                        (s, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_ch, out_ch, s in zip(in_channels, out_channels, strides)
            ]
        )

        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorR(torch.nn.Module):
    """
    Discriminator on a linear magnitude spectrogram at one STFT resolution.

    Ported from upstream Applio, which trains RefineGAN against these (version "v3").

    Args:
        resolution (list): [n_fft, hop_length, win_length] of the STFT.
        use_spectral_norm (bool): Whether to use spectral normalization. Defaults to False.
        window (str): "ones" for upstream Applio's rectangular window, "hann" for the one
            the Codename RVC fork's MRD uses. Defaults to "ones", so v1 to v3 are
            unchanged.
    """

    def __init__(
        self,
        resolution,
        use_spectral_norm: bool = False,
        window: str = DEFAULT_RESOLUTION_WINDOW,
    ):
        super().__init__()

        self.resolution = resolution
        self.window = window
        self.lrelu_slope = LRELU_SLOPE
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                norm_f(torch.nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(torch.nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(torch.nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(torch.nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = norm_f(torch.nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x).unsqueeze(1)

        for layer in self.convs:
            x = F.leaky_relu(layer(x), self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return torch.flatten(x, 1, -1), fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        pad = int((n_fft - hop_length) / 2)
        x = F.pad(x, (pad, pad), mode="reflect").squeeze(1)
        # The generator output arrives in fp16/bf16 under autocast, and the CUDA FFT does
        # not take half precision, so the STFT always runs in fp32.
        window = (
            torch.hann_window(win_length, device=x.device)
            if self.window == "hann"
            else torch.ones(win_length, device=x.device)
        )
        x = torch.stft(
            x.float(),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            return_complex=True,
        )

        return torch.norm(torch.view_as_real(x), p=2, dim=-1)  # [B, F, TT]
