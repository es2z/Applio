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

from rvc.lib.utils import load_audio_16k, load_embedding
from rvc.train.extract.preparing_files import generate_config, generate_filelist
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
        if f0_method in ("crepe", "crepe-tiny"):
            self.model = CREPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        elif f0_method == "mangio-crepe":
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
        if self.f0_method == "crepe":
            f0 = self.model.get_f0(x, self.f0_min, self.f0_max, p_len, "full")
        elif self.f0_method == "crepe-tiny":
            f0 = self.model.get_f0(x, self.f0_min, self.f0_max, p_len, "tiny")
        elif self.f0_method == "mangio-crepe":
            f0 = self.model.get_f0(x, self.f0_min, self.f0_max, p_len)
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
            return True  # Already exists

        try:
            np_arr = load_audio_16k(inp_path)
            if np_arr is None or len(np_arr) == 0:
                print(f"[ERROR] Failed to load audio or empty audio: {inp_path}")
                return False
            feature_pit = self.compute_f0(np_arr)
            if feature_pit is None:
                print(f"[ERROR] compute_f0 returned None for {inp_path}")
                return False
            np.save(opt_path_full, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path_coarse, coarse_pit, allow_pickle=False)
            return True
        except Exception as error:
            print(
                f"An error occurred extracting file {inp_path} on {self.device}: {error}"
            )
            import traceback
            traceback.print_exc()
            return False


def process_files(files, f0_method, device, threads):
    print(f"[DEBUG] process_files started: device={device}, f0_method={f0_method}, num_files={len(files)}")
    try:
        fe = FeatureInput(f0_method=f0_method, device=device)
        print(f"[DEBUG] FeatureInput created successfully on {device}")
    except Exception as e:
        print(f"[ERROR] Failed to create FeatureInput on {device}: {e}")
        import traceback
        traceback.print_exc()
        return

    success_count = 0
    error_count = 0
    with tqdm.tqdm(total=len(files), leave=True) as pbar:
        for file_info in files:
            result = fe.process_file(file_info)
            if result:
                success_count += 1
            else:
                error_count += 1
            pbar.update(1)
    print(f"[DEBUG] process_files completed on {device}: success={success_count}, errors={error_count}")


def run_pitch_extraction(files, devices, f0_method, threads):
    devices_str = ", ".join(devices)
    print(f"[DEBUG] run_pitch_extraction called with {len(files)} files on devices: {devices_str}")
    print(f"Starting pitch extraction on {devices_str} using {f0_method}...")

    if len(files) == 0:
        print("[ERROR] No files to process for pitch extraction!")
        return

    # Show first few file paths for debugging
    print(f"[DEBUG] First 3 files to process:")
    for i, f in enumerate(files[:3]):
        print(f"[DEBUG]   {i}: input={f[0]}, f0_out={f[1]}, f0nsf_out={f[2]}")

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        tasks = []
        for i in range(len(devices)):
            files_for_device = files[i :: len(devices)]
            print(f"[DEBUG] Submitting {len(files_for_device)} files to device {devices[i]}")
            tasks.append(
                executor.submit(
                    process_files,
                    files_for_device,
                    f0_method,
                    devices[i],
                    threads // len(devices),
                )
            )
        # Wait for all tasks and check for exceptions
        for i, task in enumerate(concurrent.futures.as_completed(tasks)):
            try:
                task.result()
                print(f"[DEBUG] Task {i} completed successfully")
            except Exception as e:
                print(f"[ERROR] Task {i} failed with exception: {e}")
                import traceback
                traceback.print_exc()

    # Verify output files were created
    f0_dir = os.path.dirname(files[0][1]) if files else None
    if f0_dir and os.path.exists(f0_dir):
        f0_files = os.listdir(f0_dir)
        print(f"[DEBUG] After extraction, f0 directory has {len(f0_files)} files")
    else:
        print(f"[DEBUG] f0 directory does not exist: {f0_dir}")

    print(f"Pitch extraction completed in {time.time() - start_time:.2f} seconds.")


def process_file_embedding(
    files, embedder_model, embedder_model_custom, device_num, device, n_threads
):
    model = load_embedding(embedder_model, embedder_model_custom).to(device).float()
    model.eval()
    n_threads = max(1, n_threads)

    def worker(file_info):
        wav_file_path, _, _, out_file_path = file_info
        try:
            if os.path.exists(out_file_path):
                return
            feats = torch.from_numpy(load_audio_16k(wav_file_path)).to(device).float()
            feats = feats.view(1, -1)
            with torch.no_grad():
                result = model(feats)["last_hidden_state"]
            feats_out = result.squeeze(0).float().cpu().numpy()
            if not np.isnan(feats_out).any():
                np.save(out_file_path, feats_out, allow_pickle=False)
            else:
                print(f"{wav_file_path} produced NaN values; skipping.")
        except Exception as e:
            print(f"Error processing {wav_file_path}: {e}")
            import traceback
            traceback.print_exc()

    with tqdm.tqdm(total=len(files), leave=True, position=device_num) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, f) for f in files]
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


def run_embedding_extraction(
    files, devices, embedder_model, embedder_model_custom, threads
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

    print(f"[DEBUG] ========== EXTRACTION STARTED ==========")
    print(f"[DEBUG] exp_dir: {exp_dir}")
    print(f"[DEBUG] f0_method: {f0_method}")
    print(f"[DEBUG] sample_rate: {sample_rate}")
    print(f"[DEBUG] gpus: {gpus}")
    print(f"[DEBUG] embedder_model: {embedder_model}")

    wav_path = os.path.join(exp_dir, "sliced_audios_16k")
    os.makedirs(os.path.join(exp_dir, "f0"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "f0_voiced"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "extracted"), exist_ok=True)

    # Debug: Check sliced_audios_16k directory
    if os.path.exists(wav_path):
        wav_files = glob.glob(os.path.join(wav_path, "*.wav"))
        print(f"[DEBUG] sliced_audios_16k directory exists with {len(wav_files)} .wav files")
        if len(wav_files) > 0 and len(wav_files) <= 5:
            print(f"[DEBUG] Files: {[os.path.basename(f) for f in wav_files]}")
        elif len(wav_files) > 5:
            print(f"[DEBUG] First 3 files: {[os.path.basename(f) for f in wav_files[:3]]}")
    else:
        print(f"[ERROR] sliced_audios_16k directory does NOT exist: {wav_path}")

    chosen_embedder_model = (
        embedder_model_custom if embedder_model == "custom" else embedder_model
    )
    file_path = os.path.join(exp_dir, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data["embedder_model"] = chosen_embedder_model

    # Save text_enc_hidden_dim based on embedder model
    from rvc.lib.utils import get_embedder_dim
    text_enc_dim = get_embedder_dim(embedder_model)
    data["text_enc_hidden_dim"] = text_enc_dim

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

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
        files, devices, embedder_model, embedder_model_custom, num_processes
    )

    generate_config(sample_rate, exp_dir)
    generate_filelist(exp_dir, sample_rate, include_mutes)
