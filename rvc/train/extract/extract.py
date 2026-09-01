import os
import sys
import glob
import time
import tqdm
import torch
import torchcrepe
import numpy as np
import concurrent.futures
import multiprocessing as mp
import json

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

# Zluda hijack
import rvc.lib.zluda

from rvc.lib.utils import (
    EMBEDDER_FEATURE_SCALE,
    apply_embedder_feature_scale,
    apply_embedder_input_normalization,
    load_audio_16k,
    load_embedding,
)
from rvc.train.extract.preparing_files import generate_config, generate_filelist
from rvc.lib.predictors.crepe_models import (
    CREPE_METHOD_TO_MODEL,
    MANGIO_CREPE_METHOD_TO_MODEL,
    resolve_crepe_model,
)
from rvc.lib.predictors.f0 import CREPE, FCPE, RMVPE, MANGIO_CREPE
from rvc.configs.config import Config

# Load config
config = Config()
mp.set_start_method("spawn", force=True)


class FeatureInput:
    def __init__(self, f0_method="rmvpe", device="cpu"):
        self.hop_size = 160  # default
        self.sample_rate = 16000  # default
        self.f0_bin = 256
        self.f0_max = 1680.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        if f0_method in CREPE_METHOD_TO_MODEL:
            self.model = CREPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        elif f0_method in MANGIO_CREPE_METHOD_TO_MODEL:
            self.model = MANGIO_CREPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        elif f0_method == "rmvpe":
            self.model = RMVPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        elif f0_method == "fcpe":
            self.model = FCPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        self.f0_method = f0_method

    def compute_f0(self, x, p_len=None):
        if self.f0_method in CREPE_METHOD_TO_MODEL:
            f0 = self.model.get_f0(
                x,
                self.f0_min,
                self.f0_max,
                p_len,
                resolve_crepe_model(self.f0_method),
            )
        elif self.f0_method in MANGIO_CREPE_METHOD_TO_MODEL:
            f0 = self.model.get_f0(
                x,
                self.f0_min,
                self.f0_max,
                p_len,
                resolve_crepe_model(self.f0_method),
            )
        elif self.f0_method == "rmvpe":
            f0 = self.model.get_f0(x, filter_radius=0.03)
        elif self.f0_method == "fcpe":
            f0 = self.model.get_f0(x, p_len, filter_radius=0.006)
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
        f0_mel = np.clip(
            (f0_mel - self.f0_mel_min)
            * (self.f0_bin - 2)
            / (self.f0_mel_max - self.f0_mel_min)
            + 1,
            1,
            self.f0_bin - 1,
        )
        return np.rint(f0_mel).astype(int)

    def process_file(self, file_info):
        inp_path, opt_path_coarse, opt_path_full, _ = file_info
        if os.path.exists(opt_path_coarse) and os.path.exists(opt_path_full):
            return

        try:
            np_arr = load_audio_16k(inp_path)
            feature_pit = self.compute_f0(np_arr)
            np.save(opt_path_full, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path_coarse, coarse_pit, allow_pickle=False)
        except Exception as error:
            print(
                f"An error occurred extracting file {inp_path} on {self.device}: {error}"
            )


def process_files(files, f0_method, device, threads):
    fe = FeatureInput(f0_method=f0_method, device=device)
    with tqdm.tqdm(total=len(files), leave=True) as pbar:
        for file_info in files:
            fe.process_file(file_info)
            pbar.update(1)


def run_pitch_extraction(files, devices, f0_method, threads):
    devices_str = ", ".join(devices)
    print(f"Starting pitch extraction on {devices_str} using {f0_method}...")
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        tasks = [
            executor.submit(
                process_files,
                files[i :: len(devices)],
                f0_method,
                devices[i],
                threads // len(devices),
            )
            for i in range(len(devices))
        ]
        concurrent.futures.wait(tasks)

    print(f"Pitch extraction completed in {time.time() - start_time:.2f} seconds.")


def process_file_embedding(
    files,
    embedder_model,
    embedder_model_custom,
    device_num,
    device,
    n_threads,
    overwrite=False,
):
    model = load_embedding(embedder_model, embedder_model_custom).to(device).float()
    model.eval()
    n_threads = max(1, n_threads)

    def worker(file_info):
        wav_file_path, _, _, out_file_path = file_info
        if not overwrite and os.path.exists(out_file_path):
            return
        feats = torch.from_numpy(load_audio_16k(wav_file_path)).to(device).float()
        feats = feats.view(1, -1)
        feats = apply_embedder_input_normalization(model, feats)
        with torch.no_grad():
            result = model(feats)["last_hidden_state"]
        result = apply_embedder_feature_scale(model, result)
        feats_out = result.squeeze(0).float().cpu().numpy()
        if not np.isnan(feats_out).any():
            np.save(out_file_path, feats_out, allow_pickle=False)
        else:
            print(f"{wav_file_path} produced NaN values; skipping.")

    with tqdm.tqdm(total=len(files), leave=True, position=device_num) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, f) for f in files]
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


def resolve_feature_reuse(data, chosen_embedder_model, feature_scale):
    """Decide whether the features already in the folder may be reused.

    Features and the index are embedder specific, so they can only be reused when the
    previously recorded embedder is known to be the same one. The feature scale is part of
    that identity: it changes the stored features while the embedder name stays put. A
    folder that recorded an embedder but no scale was extracted before scaling existed,
    which is exactly a scale of 1.0. A folder that recorded nothing is treated as unknown
    and left alone, the same conservative fallback preparing_files.py uses.

    Returns (rebuild, reason).
    """
    previous_embedder_model = data.get("embedder_model")
    if previous_embedder_model is None:
        return False, None
    if previous_embedder_model != chosen_embedder_model:
        return True, (
            f"Embedder changed from '{previous_embedder_model}' to "
            f"'{chosen_embedder_model}'"
        )
    previous_feature_scale = data.get("embedder_feature_scale", 1.0)
    if previous_feature_scale != feature_scale:
        return True, (
            f"Feature scale of '{chosen_embedder_model}' changed from "
            f"{previous_feature_scale} to {feature_scale}"
        )
    return False, None


def run_embedding_extraction(
    files, devices, embedder_model, embedder_model_custom, threads, overwrite=False
):
    devices_str = ", ".join(devices)
    print(
        f"Starting embedding extraction with {num_processes} cores on {devices_str}..."
    )
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        tasks = [
            executor.submit(
                process_file_embedding,
                files[i :: len(devices)],
                embedder_model,
                embedder_model_custom,
                i,
                devices[i],
                threads // len(devices),
                overwrite,
            )
            for i in range(len(devices))
        ]
        concurrent.futures.wait(tasks)

    print(f"Embedding extraction completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    exp_dir = sys.argv[1]
    f0_method = sys.argv[2]
    num_processes = int(sys.argv[3])
    gpus = sys.argv[4]
    sample_rate = sys.argv[5]
    embedder_model = sys.argv[6]
    embedder_model_custom = sys.argv[7] if len(sys.argv) > 7 else None
    include_mutes = int(sys.argv[8]) if len(sys.argv) > 8 else 2

    wav_path = os.path.join(exp_dir, "sliced_audios_16k")
    os.makedirs(os.path.join(exp_dir, "f0"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "f0_voiced"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "extracted"), exist_ok=True)

    chosen_embedder_model = (
        embedder_model_custom if embedder_model == "custom" else embedder_model
    )
    feature_scale = EMBEDDER_FEATURE_SCALE.get(embedder_model, 1.0)
    file_path = os.path.join(exp_dir, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    rebuild_features, rebuild_reason = resolve_feature_reuse(
        data, chosen_embedder_model, feature_scale
    )
    data["embedder_model"] = chosen_embedder_model
    data["embedder_feature_scale"] = feature_scale
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    if rebuild_features:
        print(f"{rebuild_reason}: re-extracting every feature.")
        index_path = os.path.join(exp_dir, f"{os.path.basename(exp_dir)}.index")
        if os.path.exists(index_path):
            os.remove(index_path)
            print(f"Removed the outdated index file {index_path}")

    files = []
    for file in glob.glob(os.path.join(wav_path, "*.wav")):
        file_name = os.path.basename(file)
        file_info = [
            file,
            os.path.join(exp_dir, "f0", file_name + ".npy"),
            os.path.join(exp_dir, "f0_voiced", file_name + ".npy"),
            os.path.join(exp_dir, "extracted", file_name.replace("wav", "npy")),
        ]
        files.append(file_info)

    devices = ["cpu"] if gpus == "-" else [f"cuda:{idx}" for idx in gpus.split("-")]

    run_pitch_extraction(files, devices, f0_method, num_processes)

    run_embedding_extraction(
        files,
        devices,
        embedder_model,
        embedder_model_custom,
        num_processes,
        rebuild_features,
    )

    generate_config(sample_rate, exp_dir)
    generate_filelist(exp_dir, sample_rate, include_mutes)
