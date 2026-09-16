import os
import shutil
from random import shuffle
from rvc.configs.config import Config
from rvc.train.lr_boost import (
    DEFAULT_G_LR_BOOST_EPOCHS,
    DEFAULT_G_LR_BOOST_MULTIPLIER,
    G_LR_BOOST_EPOCHS_KEY,
    G_LR_BOOST_MULTIPLIER_KEY,
    validate_generator_lr_boost,
)
import json

config = Config()
current_directory = os.getcwd()


def generate_config(
    sample_rate: int, model_path: str, text_enc_hidden_dim: int = None
):
    """Seed the run's config, then keep its feature width honest.

    The stock config is only copied in once, so hand edits to things like learning_rate
    survive a re-extract. text_enc_hidden_dim is different: it has to match the features
    that were just written, or the generator is built with the wrong sized emb_phone. So
    that one key is rewritten every time, and only that one.
    """
    config_path = os.path.join("rvc", "configs", f"{sample_rate}.json")
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path):
        shutil.copyfile(config_path, config_save_path)

    if text_enc_hidden_dim is None:
        return

    with open(config_save_path, "r") as f:
        config = json.load(f)
    if config.get("model", {}).get("text_enc_hidden_dim") == text_enc_hidden_dim:
        return
    config.setdefault("model", {})["text_enc_hidden_dim"] = text_enc_hidden_dim
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Set text_enc_hidden_dim to {text_enc_hidden_dim} in {config_save_path}")


# learning_rate and c_mel are not part of the extract step, but logs/<model>/config.json is
# the only place they live, so reading and writing them belongs next to generate_config.
TRAIN_SETTING_KEYS = ("learning_rate", "c_mel", "lr_decay")


def read_train_settings(model_path: str, sample_rate: int):
    """Current learning_rate / c_mel / lr_decay for a run, falling back to the stock config.

    A run that has not been extracted yet has no config.json, so the values the UI shows
    are the ones it would inherit from rvc/configs/<sample_rate>.json. Each key falls back
    on its own, so a run config that lacks one of them still shows its own values for the
    others.
    """
    settings = {}
    for path in (
        os.path.join(model_path, "config.json"),
        os.path.join("rvc", "configs", f"{sample_rate}.json"),
    ):
        if os.path.isfile(path):
            with open(path, "r") as f:
                train = json.load(f).get("train", {})
            for key in TRAIN_SETTING_KEYS:
                if key not in settings and key in train:
                    settings[key] = train[key]
    return settings if len(settings) == len(TRAIN_SETTING_KEYS) else {}


def apply_train_settings(model_path: str, **settings):
    """Write learning_rate / c_mel / lr_decay into a run's config.json, leaving the rest be.

    Only touches keys that were actually passed and actually differ, so a caller that
    hands back what read_train_settings gave it is a no-op and cannot clobber a hand edit.
    """
    lr_decay = settings.get("lr_decay")
    if lr_decay is not None and not 0 < float(lr_decay) <= 1:
        raise ValueError(
            f"lr_decay is a per-epoch multiplier and must be above 0 and at most 1, "
            f"got {lr_decay}."
        )
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, "r") as f:
        config = json.load(f)
    train = config.setdefault("train", {})
    changed = {
        key: value
        for key, value in settings.items()
        if key in TRAIN_SETTING_KEYS and value is not None and train.get(key) != value
    }
    if not changed:
        return
    train.update(changed)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    for key, value in changed.items():
        print(f"Set {key} to {value} in {config_path}")


def read_generator_lr_boost_settings(model_path: str):
    """What the Initial Generator LR Boost controls should show for a run.

    Off unless the run's config.json has a boost epoch count above 0. When it is off the
    number boxes show the defaults, so ticking the box offers something sensible.
    """
    train = {}
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            train = json.load(f).get("train", {})
    epochs = train.get(G_LR_BOOST_EPOCHS_KEY) or 0
    enabled = epochs > 0
    return {
        "enabled": enabled,
        "multiplier": train.get(
            G_LR_BOOST_MULTIPLIER_KEY, DEFAULT_G_LR_BOOST_MULTIPLIER
        ),
        "epochs": epochs if enabled else DEFAULT_G_LR_BOOST_EPOCHS,
    }


def apply_generator_lr_boost_settings(
    model_path: str, enabled=None, multiplier=None, epochs=None
):
    """Write the Initial Generator LR Boost settings into a run's config.json.

    enabled=True writes both values. enabled=False switches a previously enabled boost
    off by setting its epoch count to 0, and leaves a config that never had one alone, so
    training with the box unticked changes nothing. enabled=None (the CLI) writes only the
    values that were passed. Raises ValueError for values train.py would refuse.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, "r") as f:
        config = json.load(f)
    train = config.setdefault("train", {})

    if enabled is None:
        wanted = {}
        if multiplier is not None or epochs is not None:
            valid_multiplier, valid_epochs = validate_generator_lr_boost(
                (
                    multiplier
                    if multiplier is not None
                    else train.get(
                        G_LR_BOOST_MULTIPLIER_KEY, DEFAULT_G_LR_BOOST_MULTIPLIER
                    )
                ),
                epochs if epochs is not None else train.get(G_LR_BOOST_EPOCHS_KEY, 0),
            )
            if multiplier is not None:
                wanted[G_LR_BOOST_MULTIPLIER_KEY] = valid_multiplier
            if epochs is not None:
                wanted[G_LR_BOOST_EPOCHS_KEY] = valid_epochs
    elif enabled:
        valid_multiplier, valid_epochs = validate_generator_lr_boost(multiplier, epochs)
        if valid_epochs == 0:
            raise ValueError(
                "Initial Generator LR Boost is enabled with 0 epochs. Untick it, or give "
                "it at least 1 epoch."
            )
        wanted = {
            G_LR_BOOST_MULTIPLIER_KEY: valid_multiplier,
            G_LR_BOOST_EPOCHS_KEY: valid_epochs,
        }
    else:
        wanted = {G_LR_BOOST_EPOCHS_KEY: 0} if train.get(G_LR_BOOST_EPOCHS_KEY) else {}

    changed = {key: value for key, value in wanted.items() if train.get(key) != value}
    if not changed:
        return
    train.update(changed)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    for key, value in changed.items():
        print(f"Set {key} to {value} in {config_path}")


def generate_filelist(model_path: str, sample_rate: int, include_mutes: int = 2):
    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(model_path, f"extracted")

    f0_dir, f0nsf_dir = None, None
    f0_dir = os.path.join(model_path, "f0")
    f0nsf_dir = os.path.join(model_path, "f0_voiced")

    gt_wavs_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir))
    feature_files = set(name.split(".")[0] for name in os.listdir(feature_dir))

    f0_files = set(name.split(".")[0] for name in os.listdir(f0_dir))
    f0nsf_files = set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
    names = gt_wavs_files & feature_files & f0_files & f0nsf_files

    try:
        model_info_path = os.path.join(model_path, "model_info.json")
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
            embedder_name = model_info["embedder_model"]
    except:
        embedder_name = "contentvec"

    # The silent audio and its pitch are the same for every embedder; only the feature
    # is embedder specific. extract.py writes this run's own, which is the only way a
    # non-768 embedder can pad a batch, so prefer it and keep the shipped folders as the
    # fallback for a folder extracted before that existed.
    if embedder_name == "spin":
        mute_base_path = os.path.join(current_directory, "logs", "mute_spin")
    elif embedder_name == "spin-v2":
        mute_base_path = os.path.join(current_directory, "logs", "mute_spin-v2")
    else:
        mute_base_path = os.path.join(current_directory, "logs", "mute")

    options = []
    sids = []
    for name in names:
        sid = name.split("_")[0]
        if sid not in sids:
            sids.append(sid)
        options.append(
            f"{os.path.join(gt_wavs_dir, name)}.wav|{os.path.join(feature_dir, name)}.npy|{os.path.join(f0_dir, name)}.wav.npy|{os.path.join(f0nsf_dir, name)}.wav.npy|{sid}"
        )

    if include_mutes > 0:
        mute_audio_path = os.path.join(
            mute_base_path, "sliced_audios", f"mute{sample_rate}.wav"
        )
        mute_feature_path = os.path.join(model_path, "mute.npy")
        if not os.path.exists(mute_feature_path):
            mute_feature_path = os.path.join(mute_base_path, "extracted", "mute.npy")
        mute_f0_path = os.path.join(mute_base_path, "f0", "mute.wav.npy")
        mute_f0nsf_path = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")

        # adding x files per sid
        for sid in sids * include_mutes:
            options.append(
                f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}"
            )

    file_path = os.path.join(model_path, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data.update(
        {
            "speakers_id": len(sids),
        }
    )
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    shuffle(options)

    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))
