import os
import shutil
from random import shuffle
from rvc.configs.config import Config
import json

config = Config()
current_directory = os.getcwd()


def generate_config(sample_rate: int, model_path: str):
    config_path = os.path.join("rvc", "configs", f"{sample_rate}.json")
    config_save_path = os.path.join(model_path, "config.json")

    # Check if we need to update the config (new or sample_rate changed)
    need_update = True
    if os.path.exists(config_save_path):
        try:
            with open(config_save_path, "r") as f:
                existing_config = json.load(f)
            existing_sr = existing_config.get("data", {}).get("sample_rate", 0)
            if existing_sr == sample_rate:
                need_update = False
                print(f"[DEBUG] config.json already exists with correct sample_rate={sample_rate}")
            else:
                print(f"[DEBUG] config.json sample_rate mismatch: existing={existing_sr}, requested={sample_rate}")
                print(f"[DEBUG] Updating config.json to use {sample_rate}Hz settings")
        except Exception as e:
            print(f"[DEBUG] Could not read existing config.json: {e}")

    if need_update:
        if not os.path.exists(config_path):
            print(f"[ERROR] Config file not found: {config_path}")
            return
        shutil.copyfile(config_path, config_save_path)
        print(f"[DEBUG] Copied {config_path} to {config_save_path}")

        # Update text_enc_hidden_dim from model_info.json
        model_info_path = os.path.join(model_path, "model_info.json")
        if os.path.exists(model_info_path):
            try:
                with open(model_info_path, "r") as f:
                    model_info = json.load(f)
                text_enc_hidden_dim = model_info.get("text_enc_hidden_dim", 768)

                # Load and update config.json
                with open(config_save_path, "r") as f:
                    config = json.load(f)
                config["model"]["text_enc_hidden_dim"] = text_enc_hidden_dim

                with open(config_save_path, "w") as f:
                    json.dump(config, f, indent=2)

                print(f"Updated config.json with text_enc_hidden_dim={text_enc_hidden_dim}")
            except Exception as e:
                print(f"Warning: Could not update text_enc_hidden_dim in config.json: {e}")


def generate_filelist(model_path: str, sample_rate: int, include_mutes: int = 2):
    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(model_path, f"extracted")

    f0_dir, f0nsf_dir = None, None
    f0_dir = os.path.join(model_path, "f0")
    f0nsf_dir = os.path.join(model_path, "f0_voiced")

    # Debug: Check if directories exist and count files
    print(f"[DEBUG] Checking directories in {model_path}")
    for dir_name, dir_path in [
        ("sliced_audios", gt_wavs_dir),
        ("extracted", feature_dir),
        ("f0", f0_dir),
        ("f0_voiced", f0nsf_dir),
    ]:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"[DEBUG]   {dir_name}: {len(files)} files")
            if len(files) > 0 and len(files) <= 5:
                print(f"[DEBUG]     Files: {files}")
        else:
            print(f"[DEBUG]   {dir_name}: DIRECTORY NOT FOUND")

    gt_wavs_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir)) if os.path.exists(gt_wavs_dir) else set()
    feature_files = set(name.split(".")[0] for name in os.listdir(feature_dir)) if os.path.exists(feature_dir) else set()
    f0_files = set(name.split(".")[0] for name in os.listdir(f0_dir)) if os.path.exists(f0_dir) else set()
    f0nsf_files = set(name.split(".")[0] for name in os.listdir(f0nsf_dir)) if os.path.exists(f0nsf_dir) else set()

    print(f"[DEBUG] Unique names - sliced_audios: {len(gt_wavs_files)}, extracted: {len(feature_files)}, f0: {len(f0_files)}, f0_voiced: {len(f0nsf_files)}")

    names = gt_wavs_files & feature_files & f0_files & f0nsf_files
    print(f"[DEBUG] Common names (intersection): {len(names)}")
    if len(names) == 0 and len(gt_wavs_files) > 0:
        print(f"[DEBUG] WARNING: No common files found! This may indicate extraction failed.")
        # Show sample names from each set for debugging
        if gt_wavs_files:
            print(f"[DEBUG]   Sample from sliced_audios: {list(gt_wavs_files)[:3]}")
        if feature_files:
            print(f"[DEBUG]   Sample from extracted: {list(feature_files)[:3]}")
        if f0_files:
            print(f"[DEBUG]   Sample from f0: {list(f0_files)[:3]}")
        if f0nsf_files:
            print(f"[DEBUG]   Sample from f0_voiced: {list(f0nsf_files)[:3]}")

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
