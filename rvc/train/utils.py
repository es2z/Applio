import os
import sys
import glob
import torch
import numpy as np
import soundfile as sf
from collections import OrderedDict
import matplotlib.pyplot as plt

from rvc.lib.utils import describe_embedder_mismatch
from rvc.train.warm_start import (
    EMBEDDER_PROJECTION_PREFIX,
    checkpoint_vocoder,
    warm_start,
)

MATPLOTLIB_FLAG = False


def replace_keys_in_dict(d, old_key_part, new_key_part):
    """
    Recursively replace parts of the keys in a dictionary.

    Args:
        d (dict or OrderedDict): The dictionary to update.
        old_key_part (str): The part of the key to replace.
        new_key_part (str): The new part of the key.
    """
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}
    for key, value in d.items():
        new_key = (
            key.replace(old_key_part, new_key_part) if isinstance(key, str) else key
        )
        updated_dict[new_key] = (
            replace_keys_in_dict(value, old_key_part, new_key_part)
            if isinstance(value, dict)
            else value
        )
    return updated_dict


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    """
    Load a checkpoint into a model and optionally the optimizer.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the checkpoint into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state from. Defaults to None.
        load_opt (int, optional): Whether to load the optimizer state. Defaults to 1.
    """
    assert os.path.isfile(
        checkpoint_path
    ), f"Checkpoint file not found: {checkpoint_path}"

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_dict = replace_keys_in_dict(
        replace_keys_in_dict(
            checkpoint_dict, ".weight_v", ".parametrizations.weight.original1"
        ),
        ".weight_g",
        ".parametrizations.weight.original0",
    )

    # Update model state_dict
    model_state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    new_state_dict = {
        k: checkpoint_dict["model"].get(k, v) for k, v in model_state_dict.items()
    }

    # Load state_dict into model
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    if optimizer and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict.get("optimizer", {}))

    print(
        f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint_dict['iteration']})"
    )
    return (
        model,
        optimizer,
        checkpoint_dict.get("learning_rate", 0),
        checkpoint_dict["iteration"],
        checkpoint_dict.get("scaler", {}),
    )


def load_pretrained(net, checkpoint_path, tag, verbose=True, target_identity=None):
    """Load a pretrained G or D, inheriting only what means the same thing in this model.

    The rules - legacy weight_g / weight_v names, a different sized embedder, a different
    vocoder, a different discriminator layout - live in rvc/train/warm_start.py. Anything
    that does not fit them stops the run rather than being skipped.

    target_identity is {"vocoder", "sample_rate"} of the model being trained; without it a
    decoder is only inherited from the same vocoder.
    """
    return warm_start(
        net, checkpoint_path, tag, target_identity=target_identity, verbose=verbose
    )


def describe_architecture_mismatch(checkpoint, current, subject="it"):
    """Return why a resume checkpoint belongs to a different architecture, or None.

    The vocoder is read off the weights, so it is known for every checkpoint ever saved.
    Sample rate and discriminator version are only compared when the checkpoint recorded
    them; before they were recorded a different sample rate already failed to load.
    """
    if not current:
        return None
    reasons = []
    was_vocoder, now_vocoder = checkpoint_vocoder(checkpoint), current.get("vocoder")
    if was_vocoder and now_vocoder and was_vocoder != now_vocoder:
        reasons.append(f"vocoder '{was_vocoder}' -> '{now_vocoder}'")
    for key, label in (
        ("sample_rate", "sample rate"),
        ("disc_version", "discriminator version"),
    ):
        was, now = checkpoint.get(key), current.get(key)
        if was is not None and now is not None and was != now:
            reasons.append(f"{label} {was} -> {now}")
    if not reasons:
        return None
    return f"{subject} was trained with a different architecture ({'; '.join(reasons)})"


def assert_resumable(experiment_dir, embedder_identity, architecture_identity=None):
    """Stop before resuming from a checkpoint this run cannot continue.

    Changing the embedder re-extracts every feature and drops the index, but the
    G_*.pth / D_*.pth in the folder survive and training would silently continue from
    them. Their enc_p.emb_phone, and the Adam moments behind it, were fitted to the old
    features; feeding it the new ones does not raise, it just produces a model that never
    recovers. Changing the vocoder is the same story for the decoder and discriminator.
    So say so and stop instead.
    """
    if not embedder_identity and not architecture_identity:
        return
    checkpoint_path = latest_checkpoint_path(experiment_dir, "G_*.pth")
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        print(f"Could not inspect {checkpoint_path} ({error}); continuing.")
        return

    name = os.path.basename(checkpoint_path)
    embedder_reason = (
        describe_embedder_mismatch(checkpoint, embedder_identity, name)
        if embedder_identity
        else None
    )
    architecture_reason = describe_architecture_mismatch(
        checkpoint, architecture_identity, name
    )
    del checkpoint
    if embedder_reason is None and architecture_reason is None:
        return

    if embedder_reason is not None:
        print(
            f"Refusing to resume: {embedder_reason}.\n"
            "Its enc_p.emb_phone was fitted to the previous features, and resuming onto "
            "the new ones would train from a broken starting point rather than fail."
        )
    if architecture_reason is not None:
        print(
            f"Refusing to resume: {architecture_reason}.\n"
            "Its generator, discriminator and the optimizer state behind them belong to "
            "the other architecture. To start this one from those weights, train under a "
            "new model name with them as the custom pretrained G and D."
        )
    print(
        f"Either train under a new model name, or delete the G_*.pth and D_*.pth in "
        f"{experiment_dir} to start again from the pretrained model."
    )
    sys.exit(1)


def save_checkpoint(
    model,
    optimizer,
    learning_rate,
    iteration,
    checkpoint_path,
    scaler,
    embedder_identity=None,
    architecture_identity=None,
):
    """
    Save the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save the state of.
        learning_rate (float): The current learning rate.
        iteration (int): The current iteration.
        checkpoint_path (str): The path to save the checkpoint to.
        embedder_identity (dict): Which embedder, scale and layer the features this was
            trained on came from, so a later resume can refuse to continue from a
            checkpoint whose enc_p.emb_phone was fitted to different features.
        architecture_identity (dict): vocoder, sample_rate and disc_version, so a resume
            can refuse a different architecture and a warm start from this file knows
            what it is inheriting from.
    """
    state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    checkpoint_data = {
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
        "scaler": scaler.state_dict(),
    }
    if embedder_identity:
        checkpoint_data.update(embedder_identity)
    if architecture_identity:
        checkpoint_data.update(architecture_identity)

    # Create a backwards-compatible checkpoint
    torch.save(
        replace_keys_in_dict(
            replace_keys_in_dict(
                checkpoint_data, ".parametrizations.weight.original1", ".weight_v"
            ),
            ".parametrizations.weight.original0",
            ".weight_g",
        ),
        checkpoint_path,
    )

    print(f"Saved model '{checkpoint_path}' (epoch {iteration})")


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sample_rate=22050,
):
    """
    Log various summaries to a TensorBoard writer.

    Args:
        writer (SummaryWriter): The TensorBoard writer.
        global_step (int): The current global step.
        scalars (dict, optional): Dictionary of scalar values to log.
        histograms (dict, optional): Dictionary of histogram values to log.
        images (dict, optional): Dictionary of image values to log.
        audios (dict, optional): Dictionary of audio values to log.
        audio_sample_rate (int, optional): Sampling rate of the audio data.
    """
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sample_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """
    Get the latest checkpoint file in a directory.

    Args:
        dir_path (str): The directory to search for checkpoints.
        regex (str, optional): The regular expression to match checkpoint files.
    """
    checkpoints = sorted(
        glob.glob(os.path.join(dir_path, regex)),
        key=lambda f: int("".join(filter(str.isdigit, f))),
    )
    return checkpoints[-1] if checkpoints else None


def plot_spectrogram_to_numpy(spectrogram):
    """
    Convert a spectrogram to a NumPy array for visualization.

    Args:
        spectrogram (numpy.ndarray): The spectrogram to plot.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def load_wav_to_torch(full_path):
    """
    Load a WAV file into a PyTorch tensor.

    Args:
        full_path (str): The path to the WAV file.
    """
    data, sample_rate = sf.read(full_path, dtype="float32")
    return torch.FloatTensor(data), sample_rate


def load_filepaths_and_text(filename, split="|"):
    """
    Load filepaths and associated text from a file.

    Args:
        filename (str): The path to the file.
        split (str, optional): The delimiter used to split the lines.
    """
    with open(filename, encoding="utf-8") as f:
        return [line.strip().split(split) for line in f]


class HParams:
    """
    A class for storing and accessing hyperparameters.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = HParams(**v) if isinstance(v, dict) else v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)
