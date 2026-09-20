"""Which discriminator and which mel loss each vocoder trains against.

train.py decides this for the run it is about to start, and reset_run.py has to decide the
same thing for the checkpoints it installs ahead of that run, so it lives in one place
rather than in two that can drift apart.
"""

from rvc.train.warm_start import CODENAME_RINGFORMER, REFINEGAN, SIFIGAN

DEFAULT_DISCRIMINATOR_VERSION = "v2"

# Upstream Applio trains RefineGAN against the v3 discriminator (five periods plus three
# STFT resolution discriminators) with the multi-scale mel loss. HiFi-GAN keeps v2, and so
# does MRF HiFi-GAN, which is the HiFi-GAN decoder under other names and is trained the
# same way in the Codename RVC fork. SiFi-GAN gets RefineGAN's treatment: the official
# implementation trains it against a UnivNet multi-resolution spectral discriminator plus
# a HiFi-GAN multi-period one, which is what v3 already is, so there is nothing to port on
# the discriminator side.
#
# The Codename RVC fork trains RingFormer against its own layout - a scale discriminator,
# eight periods and three STFT resolutions - and against a single-scale mel loss with
# c_mel 45, so its mel loss is left as HiFi-GAN's.
_DISCRIMINATOR_VERSIONS = {
    REFINEGAN: "v3",
    SIFIGAN: "v3",
    CODENAME_RINGFORMER: "codename-ringformer",
}
_MULTISCALE_MEL_LOSS = (REFINEGAN, SIFIGAN)


def discriminator_version(vocoder):
    """The discriminator layout a run with this vocoder trains against."""
    return _DISCRIMINATOR_VERSIONS.get(vocoder, DEFAULT_DISCRIMINATOR_VERSION)


def uses_multiscale_mel_loss(vocoder):
    """Whether this vocoder trains against the multi-scale mel loss rather than L1."""
    return vocoder in _MULTISCALE_MEL_LOSS
