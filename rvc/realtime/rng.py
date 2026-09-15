"""Optional fixed RNG seed for realtime, so the voice does not drift between sessions.

`Synthesizer.infer` draws `torch.randn_like(m_p)` for the latent every chunk, and the
HiFi-GAN source module draws a random harmonic phase and a noise floor on top of it
(`rvc/lib/algorithm/synthesizers.py`, `rvc/lib/algorithm/generators/hifigan.py`). Those
draws are part of the model, not a defect - but they mean two runs of the same audio
through the same checkpoint do not sound the same. Measured on a 44 s clip, that
run-to-run difference is *larger* than the difference between eager and compiled
inference (1.188 dB against 1.135 dB of median mel distance), which is why an A/B of
any setting is inconclusive while the seed is free.

Seeding fixes the model's own randomness only. Live input still differs every take, so
the output is not bit-identical between takes - it is the *voice* that stops drifting.
Offline, with the same input and the same seed, the pipeline is bit-identical
(verified: 218.9 dB SNR between two runs).
"""

import os

from rvc.configs.config_utils import load_config, update_config

CONFIG_PATH = os.path.join(os.getcwd(), "assets", "config.json")
RANDOM = -1


def load_seed():
    """Return the configured seed, or None when realtime should stay random."""
    value = load_config(CONFIG_PATH).get("realtime_seed", RANDOM)
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return None if value < 0 else value


def save_seed(seed):
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = RANDOM
    if seed < 0:
        seed = RANDOM
    if not update_config(CONFIG_PATH, {"realtime_seed": seed}):
        raise OSError("Could not save the realtime seed")


def apply_seed(seed=None):
    """Seed torch for the audio stream that is about to start.

    Returns the seed that was applied, or None when realtime stays random. Call this
    after any warm-up: the compile warm-up runs inside ``torch.random.fork_rng`` and
    restores the state, but only the compiled path warms up at all, so seeding last is
    the one order that behaves the same with the setting on and off.
    """
    if seed is None:
        seed = load_seed()
    if seed is None:
        return None
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
