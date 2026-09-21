"""Which torchcrepe decoder the mangio-crepe methods use.

This is a setting rather than a constant because the default one is not repeatable.
Measured on an RTX 4090, five consecutive runs of the same audio inside one process:

    decoder           repeatable   worst drift
    viterbi (default)     no       3.60 Hz / 25.92 cents (mangio-crepe-full-speech)
    argmax                no       6.47 Hz
    weighted_argmax      yes       bit-identical

CUDA's ``argmax`` breaks ties arbitrarily, and CREPE's pitch bins are 20 cents apart, so
a single tie moves the estimate a whole bin. ``viterbi`` decodes a path through those
same per-frame choices and inherits the problem. ``weighted_argmax`` averages around the
peak instead, which is why the plain ``CREPE`` class has always passed it explicitly.

Switching to ``weighted_argmax`` makes the pitch track reproducible, but it is a
different estimator: the F0 values change, so the converted voice changes with them.
That is a judgement call about sound, which is why the default here stays ``viterbi``
(what mangio-crepe has always done) and the choice is exposed in the UI instead.

For reference, rmvpe, fcpe and every plain crepe method are already bit-identical.
"""

import os

from rvc.configs.config_utils import load_config, update_config
from rvc.lib.predictors.crepe_models import MANGIO_CREPE_METHOD_TO_MODEL

CONFIG_PATH = os.path.join(os.getcwd(), "assets", "config.json")

# Ordered as they are offered in the UI: the historical default first.
DECODERS = ("viterbi", "weighted_argmax", "argmax")
DEFAULT_DECODER = "viterbi"
REPEATABLE_DECODERS = ("weighted_argmax",)


def uses_mangio_crepe(f0_method):
    """True for the F0 methods this setting applies to."""
    return f0_method in MANGIO_CREPE_METHOD_TO_MODEL


def load_decoder():
    name = load_config(CONFIG_PATH).get("mangio_crepe_decoder", DEFAULT_DECODER)
    return name if name in DECODERS else DEFAULT_DECODER


def save_decoder(name):
    if name not in DECODERS:
        name = DEFAULT_DECODER
    if not update_config(CONFIG_PATH, {"mangio_crepe_decoder": name}):
        raise OSError("Could not save the mangio-crepe decoder setting")


def resolve_decoder(name=None):
    """Return the torchcrepe decode function for `name`, or for the saved setting."""
    import torchcrepe

    if name is None:
        name = load_decoder()
    return getattr(torchcrepe.decode, name if name in DECODERS else DEFAULT_DECODER)
