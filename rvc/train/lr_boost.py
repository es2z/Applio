"""Initial Generator LR Boost: multiply only the generator's learning rate for the first
epochs of a run.

Meant for a warm start whose generator is partly new - RefineGAN started from a HiFi-GAN
pretrain, say - where the fresh parts would otherwise crawl along at a learning rate
chosen for fine-tuning. The discriminator keeps its normal learning rate.

The boost is put on around each epoch's optimizer steps and taken back off before
anything is logged or saved, restoring the exact previous values rather than dividing.
So the learning rate stored in G_*.pth and the ExponentialLR chain are bit for bit what
they would be without it. The decision is made from the absolute epoch number, which a
resume restores from the checkpoint, so a resumed run neither restarts nor loses its
boost period. With the boost off nothing here touches the optimizer at all.

The settings live in logs/<model>/config.json under "train", next to learning_rate.
"""

import math

G_LR_BOOST_MULTIPLIER_KEY = "g_lr_boost_multiplier"
G_LR_BOOST_EPOCHS_KEY = "g_lr_boost_epochs"
DEFAULT_G_LR_BOOST_MULTIPLIER = 3.0
DEFAULT_G_LR_BOOST_EPOCHS = 10


def validate_generator_lr_boost(multiplier, epochs):
    """Return (multiplier, epochs) as (float, int), or raise ValueError saying why not.

    epochs 0 means the boost is off.
    """
    try:
        multiplier = float(multiplier)
        epochs_value = float(epochs)
    except (TypeError, ValueError):
        raise ValueError(
            "Initial Generator LR Boost needs a numeric multiplier and epoch count, "
            f"got {multiplier!r} and {epochs!r}."
        )
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError(
            f"The generator LR multiplier must be a positive number, got {multiplier}."
        )
    if not epochs_value.is_integer() or epochs_value < 0:
        raise ValueError(
            f"The boost epoch count must be a whole number, 0 or more, got {epochs}."
        )
    return multiplier, int(epochs_value)


def read_generator_lr_boost(train_settings):
    """(multiplier, epochs) that a run's config "train" section asks for."""
    epochs = (
        train_settings[G_LR_BOOST_EPOCHS_KEY]
        if G_LR_BOOST_EPOCHS_KEY in train_settings
        else 0
    )
    multiplier = (
        train_settings[G_LR_BOOST_MULTIPLIER_KEY]
        if G_LR_BOOST_MULTIPLIER_KEY in train_settings
        else DEFAULT_G_LR_BOOST_MULTIPLIER
    )
    return validate_generator_lr_boost(multiplier, epochs)


def generator_lr_boost_factor(epoch, multiplier, epochs):
    """What to multiply the generator's learning rate by during `epoch` (1-based)."""
    return multiplier if epochs > 0 and epoch <= epochs else 1.0


def scale_learning_rates(optimizer, factor):
    """Multiply every param group's lr by factor.

    Returns the previous values for restore_learning_rates, or None (and changes
    nothing) when factor is 1.
    """
    if factor == 1.0:
        return None
    saved = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * factor
    return saved


def restore_learning_rates(optimizer, saved):
    """Put back exactly the values scale_learning_rates replaced."""
    if saved is None:
        return
    for group, lr in zip(optimizer.param_groups, saved):
        group["lr"] = lr
