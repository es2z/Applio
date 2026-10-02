"""Which discriminator and which mel loss each vocoder trains against.

train.py decides this for the run it is about to start, and reset_run.py has to decide the
same thing for the checkpoints it installs ahead of that run, so it lives in one place
rather than in two that can drift apart.

What the Training tab and --vocoder offer is a training recipe. For every vocoder but one
the recipe is just the vocoder's name; SiFi-GAN+ is the SiFi-GAN generator trained against
a different discriminator. Everything that builds or describes a generator works with
generator_vocoder(recipe), so a recipe never reaches Synthesizer, which would quietly build
a HiFi-GAN decoder for a name it does not know.
"""

from rvc.train.warm_start import CODENAME_RINGFORMER, REFINEGAN, SIFIGAN

DEFAULT_DISCRIMINATOR_VERSION = "v2"

# SiFi-GAN with its resolution discriminators made for 48 kHz ("v3-48k" in
# rvc/lib/algorithm/discriminators.py). The generator, its options and its losses are
# SiFi-GAN's, and so is everything written to disk: checkpoints and exported models record
# "SiFi-GAN", and only their disc_version tells the two recipes apart.
SIFIGAN_PLUS = "SiFi-GAN+"

# The generator each recipe trains, when it is not the recipe's own name.
_GENERATOR_VOCODERS = {
    SIFIGAN_PLUS: SIFIGAN,
}

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
    SIFIGAN_PLUS: "v3-48k",
    CODENAME_RINGFORMER: "codename-ringformer",
}
# Keyed by generator: a recipe that trains the SiFi-GAN generator trains it with SiFi-GAN's
# losses.
_MULTISCALE_MEL_LOSS = (REFINEGAN, SIFIGAN)

# Recipes that only make sense at some sample rates. v3-48k's resolutions are v3's
# doubled for 48 kHz; at 32 or 40 kHz they would be neither v3 nor 48 kHz's.
_SAMPLE_RATES = {
    SIFIGAN_PLUS: (48000,),
}

# Folder under rvc/models/pretraineds/ for a recipe whose name does not make one.
_PRETRAINED_DIRS = {
    SIFIGAN_PLUS: "sifi-gan-plus",
}


def generator_vocoder(recipe):
    """The vocoder whose generator this recipe trains - what Synthesizer is built with."""
    return _GENERATOR_VOCODERS.get(recipe, recipe)


def discriminator_version(recipe):
    """The discriminator layout a run with this recipe trains against."""
    return _DISCRIMINATOR_VERSIONS.get(recipe, DEFAULT_DISCRIMINATOR_VERSION)


def uses_multiscale_mel_loss(recipe):
    """Whether this recipe trains against the multi-scale mel loss rather than L1."""
    return generator_vocoder(recipe) in _MULTISCALE_MEL_LOSS


def sample_rate_error(recipe, sample_rate):
    """Why this recipe cannot train at this sample rate, or None when it can."""
    rates = _SAMPLE_RATES.get(recipe)
    if rates is None or int(sample_rate) in rates:
        return None
    supported = " / ".join(f"{rate // 1000} kHz" for rate in rates)
    return (
        f"{recipe} trains only at {supported}, and this model is {int(sample_rate)} Hz. "
        f"Its discriminator's STFT resolutions are chosen for {supported}; use "
        f"{generator_vocoder(recipe)} at other sample rates."
    )


def pretrained_dir_name(recipe):
    """The folder under rvc/models/pretraineds/ holding this recipe's pretrained G and D."""
    return _PRETRAINED_DIRS.get(recipe, recipe.lower())
