import os
import shutil
from random import shuffle
from rvc.configs.config import Config
import json

config = Config()
current_directory = os.getcwd()


def generate_config(sample_rate: int, model_path: str):
    # Check model_info.json for hidden_channels to determine which config to use
    model_info_path = os.path.join(model_path, "model_info.json")
    hidden_channels = 192  # default
    text_enc_hidden_dim = 768  # default

    if os.path.exists(model_info_path):
        try:
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
            hidden_channels = model_info.get("hidden_channels", 192)
            text_enc_hidden_dim = model_info.get("text_enc_hidden_dim", 768)
        except Exception as e:
            print(f"Warning: Could not read model_info.json: {e}")

    # Select appropriate config file based on hidden_channels
    if hidden_channels == 768:
        config_file = f"{sample_rate}-768.json"
        print(f"[Generate Config] Using high-capacity config: {config_file}")
    else:
        config_file = f"{sample_rate}.json"
        print(f"[Generate Config] Using standard config: {config_file}")

    config_path = os.path.join("rvc", "configs", config_file)
    config_save_path = os.path.join(model_path, "config.json")

    # Check if we need to update existing config.json
    need_update = True
    if os.path.exists(config_save_path):
        try:
            with open(config_save_path, "r") as f:
                existing_config = json.load(f)
            existing_hidden = existing_config.get("model", {}).get("hidden_channels", 192)

            # Check if hidden_channels matches
            if existing_hidden == hidden_channels:
                print(f"[Generate Config] Existing config.json already has correct hidden_channels={hidden_channels}")
                need_update = False
            else:
                print(f"[Generate Config] Updating config.json: hidden_channels {existing_hidden} → {hidden_channels}")
        except Exception as e:
            print(f"Warning: Could not read existing config.json: {e}")

    if need_update:
        if os.path.exists(config_path):
            shutil.copyfile(config_path, config_save_path)
            print(f"Copied {config_file} to config.json")

            # Update text_enc_hidden_dim in config.json
            try:
                with open(config_save_path, "r") as f:
                    config = json.load(f)
                config["model"]["text_enc_hidden_dim"] = text_enc_hidden_dim

                with open(config_save_path, "w") as f:
                    json.dump(config, f, indent=4)

                print(f"Updated config.json with text_enc_hidden_dim={text_enc_hidden_dim}")
            except Exception as e:
                print(f"Warning: Could not update text_enc_hidden_dim in config.json: {e}")
        else:
            print(f"Error: Config file {config_path} not found!")


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

    if embedder_name == "spin":
        mute_base_path = os.path.join(current_directory, "logs", "mute_spin")
    elif embedder_name == "spin-v2":
        mute_base_path = os.path.join(current_directory, "logs", "mute_spin-v2")
    elif embedder_name == "japanese-hubert-large":
        mute_base_path = os.path.join(current_directory, "logs", "mute_japanese_hubert_large")
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
        mute_feature_path = os.path.join(mute_base_path, f"extracted", "mute.npy")
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
