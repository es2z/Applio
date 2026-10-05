"""Evaluation sets, and the in-training evaluation that ranks checkpoints.

An evaluation set is a folder of the user's own audio under assets/eval_sets/<name>/,
shared between runs so an A/B and a pretrain followed by its fine-tune are scored on the
same clips:

    trained/[<speaker>/]*.wav   audio that is in the training data   -> mel L1, noise
    heldout/[<speaker>/]*.wav   the same speakers, not trained on    -> mel L1, noise
    convert/*.wav               other people                         -> noise, F0 tracking

A run turns each clip into what the model was trained on - the same preprocessing
(rvc/train/preprocess/preprocess.py: load_audio, the 48 Hz high-pass, "pre"
normalisation, soxr_vhq to 16 kHz) and the same extraction (the embedder, output layer,
F0 method and coarse range recorded in the run's model_info.json) - once, into
logs/<model>/evaluation/cache/, and train.py scores each evaluated epoch from that cache.

Speakers: a run with one speaker scores everything as speaker 0. A run with several
uses the numbered subfolder as the speaker and files directly in trained/ or heldout/ as
speaker 0, and skips (with a warning) a number the run has no speaker for.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EVAL_SETS_DIR = REPO / "assets" / "eval_sets"
GROUPS = ("trained", "heldout", "convert")
GROUP_LETTERS = {"trained": "A", "heldout": "B", "convert": "C"}
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
INVALID_NAME_CHARACTERS = '\\/:*?"<>|'
# train.py's data loader caps a sample at 900 frames (9 s); longer clips are cut to that.
MAX_SECONDS = 9.0
MAX_FILES_PER_GROUP = 10
# Clips taken from the run's own filelist when trained/ is empty.
AUTO_TRAINED_COUNT = 5
# Bumped when the cached arrays change meaning, so old caches are rebuilt.
CACHE_VERSION = 1
EVALUATION_DIR = "evaluation"
SET_FILE = "set.json"
TRANSPOSE_FILE = "transpose.json"

README_TEXT = """評価セット: {name}
学習中の評価（ベストの選定、TensorBoard の eval/...）に使う音声を入れるフォルダです。
音声は wav / flac、1本 3〜9 秒程度（9 秒を超える分は切り捨て）、各フォルダ 5〜10 本が目安です。

trained\\<話者番号>\\  学習データに入っている音声（元音声のコピー）。声への適合を測ります。
heldout\\<話者番号>\\  学習データに入れていない、同じ話者の音声。初めての入力への強さを測ります。
                     ※ 事前学習・本番学習のどちらのデータにも入れないでください。
convert\\             別人の声。実際の変換に近い使い方で、ノイズと音程の追従を測ります。
                     音の高さは、変換先の話者の学習データの F0 の中央値に自動で合わせます。
                     ファイルごとに半音で指定するときは convert\\transpose.json に
                     {{"ファイル名.wav": 12}} のように書きます。

<話者番号> は学習データのフォルダ番号（0, 1, 2 ...）と同じにしてください。
番号フォルダを作らずに trained\\ や heldout\\ の直下に置いた音声は、話者 0 として扱います。
話者が1人だけの学習では、番号に関係なくすべて話者 0 として評価します。
空のフォルダは評価に使いません。trained\\ が空のときは、学習データから自動で数本選びます。
"""


# Evaluation sets


def eval_set_path(name):
    return EVAL_SETS_DIR / str(name)


def list_eval_sets():
    """The folders directly under assets/eval_sets/, sorted."""
    if not EVAL_SETS_DIR.is_dir():
        return []
    return sorted(p.name for p in EVAL_SETS_DIR.iterdir() if p.is_dir())


def validate_set_name(name):
    name = str(name or "").strip()
    if not name:
        raise ValueError("Enter a name for the evaluation set.")
    bad = sorted({c for c in name if c in INVALID_NAME_CHARACTERS})
    if bad:
        raise ValueError(f"The name cannot contain {' '.join(bad)}.")
    if name in (".", "..") or name.endswith((".", " ")):
        raise ValueError("The name cannot end with a dot or a space.")
    return name


def create_eval_set(name):
    """Create assets/eval_sets/<name>/ with its three folders and readme.txt."""
    name = validate_set_name(name)
    path = eval_set_path(name)
    if path.exists():
        raise ValueError(f"An evaluation set named '{name}' already exists.")
    for group in GROUPS:
        (path / group).mkdir(parents=True)
    (path / "readme.txt").write_text(README_TEXT.format(name=name), encoding="utf-8")
    return path


def open_folder(path):
    """Open a folder in the file manager of the machine running Applio."""
    path = str(path)
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 - a local folder the user asked to open
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _audio_files(directory):
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ) if directory.is_dir() else []


def collect_eval_files(set_dir, n_speakers=None):
    """The clips of a set with the speaker each one is scored as.

    Returns (entries, warnings); entries are dicts of group, path, sid, name. n_speakers
    None (no run chosen yet) takes the folder numbers as they are.
    """
    set_dir = Path(set_dir)
    entries, warnings = [], []
    for group in GROUPS:
        base = set_dir / group
        found = []
        if not base.is_dir():
            continue
        for path in _audio_files(base):
            found.append((path, 0))
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            if group == "convert":
                warnings.append(f"{group}/{sub.name}/: convert/ takes files directly; skipped.")
                continue
            if not sub.name.isdigit():
                warnings.append(f"{group}/{sub.name}/: not a speaker number; skipped.")
                continue
            sid = int(sub.name)
            if n_speakers is not None and n_speakers > 1 and sid >= n_speakers:
                warnings.append(
                    f"{group}/{sub.name}/: this run has no speaker {sid} "
                    f"(speakers 0..{n_speakers - 1}); skipped."
                )
                continue
            found += [(path, sid) for path in _audio_files(sub)]
        if len(found) > MAX_FILES_PER_GROUP:
            warnings.append(
                f"{group}/: {len(found)} files, only the first {MAX_FILES_PER_GROUP} are used."
            )
            found = found[:MAX_FILES_PER_GROUP]
        for path, sid in found:
            if n_speakers == 1:
                sid = 0
            entries.append({
                "group": group,
                "path": str(path),
                "sid": sid,
                "name": str(path.relative_to(set_dir)).replace("\\", "/"),
            })
    return entries, warnings


def summarize_eval_set(name, n_speakers=None):
    """One line saying what a set holds, for the Training tab."""
    if not name:
        return ""
    path = eval_set_path(name)
    if not path.is_dir():
        return f"'{name}' does not exist."
    entries, warnings = collect_eval_files(path, n_speakers)
    parts = []
    for group in GROUPS:
        group_entries = [e for e in entries if e["group"] == group]
        if group == "convert":
            parts.append(f"{group}: {len(group_entries)}")
            continue
        by_speaker = {}
        for entry in group_entries:
            by_speaker[entry["sid"]] = by_speaker.get(entry["sid"], 0) + 1
        detail = ", ".join(f"speaker {sid}: {n}" for sid, n in sorted(by_speaker.items()))
        parts.append(f"{group}: {len(group_entries)}" + (f" ({detail})" if detail else ""))
    text = " / ".join(parts)
    if warnings:
        text += "\n" + "\n".join(f"! {w}" for w in warnings)
    return text


def _feature_input_class():
    """rvc.train.extract.extract.FeatureInput, imported without its side effect.

    extract.py calls mp.set_start_method("spawn", force=True) at import. Inside a
    training process on Linux that would also switch the DataLoader's workers from fork
    to spawn, so the previous start method is put back.
    """
    import multiprocessing

    previous = multiprocessing.get_start_method(allow_none=True)
    from rvc.train.extract.extract import FeatureInput

    if previous is not None and multiprocessing.get_start_method(allow_none=True) != previous:
        multiprocessing.set_start_method(previous, force=True)
    return FeatureInput


# What a run's clips are extracted with


def extraction_spec(model_dir):
    """The preprocessing and extraction a run's training data went through.

    Raises ValueError when the run does not record enough to reproduce it.
    """
    model_dir = Path(model_dir)
    with open(model_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    info_path = model_dir / "model_info.json"
    if not info_path.is_file():
        raise ValueError(f"{info_path} is missing; extract the features first.")
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    from rvc.lib.predictors.f0_quantization import recorded_coarse_max

    method = ((info.get("pitch_extraction_run") or {}).get("specification") or {}).get("method")
    if not method:
        raise ValueError("model_info.json records no F0 method; re-run the extraction.")
    embedder = info.get("embedder_model")
    if not embedder or embedder == "custom":
        raise ValueError(
            "This run's embedder is not recorded (or is a custom one, whose path is not); "
            "evaluation cannot reproduce its features."
        )
    return {
        "cache_version": CACHE_VERSION,
        "sample_rate": int(config["data"]["sample_rate"]),
        "embedder_model": embedder,
        "embedder_output_layer": info.get("embedder_output_layer"),
        "f0_method": method,
        "f0_coarse_max": float(recorded_coarse_max(info)),
        "n_speakers": int(info.get("speakers_id") or 1),
    }


def preprocess_like_training(path, sample_rate):
    """(audio at sample_rate, audio at 16 kHz) as preprocess.py would have made them.

    Returns None for a file preprocess would have filtered out (peak above 2.5).
    """
    import librosa
    from scipy import signal

    from rvc.lib.utils import load_audio
    from rvc.train.preprocess.preprocess import (
        ALPHA,
        HIGH_PASS_CUTOFF,
        MAX_AMPLITUDE,
        RES_TYPE,
    )

    audio = load_audio(str(path), sample_rate)
    b, a = signal.butter(N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=sample_rate)
    audio = signal.lfilter(b, a, audio)
    peak = np.abs(audio).max()
    if peak > 2.5 or peak == 0:
        return None
    audio = (audio / peak * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio
    audio = audio[: int(MAX_SECONDS * sample_rate)].astype(np.float32)
    audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000, res_type=RES_TYPE)
    return audio, audio_16k.astype(np.float32)


def _cache_key(path, spec):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    digest.update(json.dumps(spec, sort_keys=True).encode())
    return digest.hexdigest()[:32]


def prepare_eval_cache(model_dir, set_name, device, log=print):
    """Extract every clip of a set the way this run's data was, reusing what is cached.

    Writes evaluation/set.json and returns its content, or None when the set has nothing
    usable. The embedder and the F0 model are only loaded when a clip is not cached yet.
    """
    model_dir = Path(model_dir)
    set_dir = eval_set_path(set_name)
    if not set_dir.is_dir():
        raise ValueError(f"Evaluation set '{set_name}' does not exist ({set_dir}).")
    spec = extraction_spec(model_dir)
    entries, warnings = collect_eval_files(set_dir, spec["n_speakers"])
    for warning in warnings:
        log(f"Evaluation set '{set_name}': {warning}")
    cache_dir = model_dir / EVALUATION_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for entry in entries:
        entry["cache"] = str(cache_dir / f"{_cache_key(entry['path'], spec)}.npz")
        if not os.path.isfile(entry["cache"]):
            pending.append(entry)

    if pending:
        import torch

        from rvc.lib.utils import embedder_forward, load_embedding

        FeatureInput = _feature_input_class()

        log(f"Evaluation set '{set_name}': extracting {len(pending)} clip(s) on {device}")
        embedder = f0_input = None
        try:
            embedder = load_embedding(
                spec["embedder_model"], None, spec["embedder_output_layer"]
            ).to(device).float().eval()
            f0_input = FeatureInput(
                f0_method=spec["f0_method"], device=device, coarse_max=spec["f0_coarse_max"]
            )
            for entry in pending:
                prepared = preprocess_like_training(entry["path"], spec["sample_rate"])
                if prepared is None:
                    log(f"Evaluation set '{set_name}': {entry['name']} is silent or clipped; skipped.")
                    entry["skip"] = True
                    continue
                audio, audio_16k = prepared
                f0 = np.asarray(f0_input.compute_f0(audio_16k), dtype=np.float32)
                coarse = np.asarray(f0_input.coarse_f0(f0), dtype=np.int64)
                with torch.no_grad():
                    feats = torch.from_numpy(audio_16k).to(device).float().view(1, -1)
                    phone = embedder_forward(embedder, feats).squeeze(0).float().cpu().numpy()
                np.savez(entry["cache"], audio=audio, phone=phone, f0=f0, coarse=coarse)
        finally:
            del embedder, f0_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    entries = [e for e in entries if not e.pop("skip", False)]
    transpose = {}
    transpose_path = set_dir / "convert" / TRANSPOSE_FILE
    if transpose_path.is_file():
        with open(transpose_path, encoding="utf-8") as f:
            transpose = {str(k): float(v) for k, v in json.load(f).items()}
    manifest = {
        "eval_set": str(set_name),
        "spec": spec,
        "entries": entries,
        "transpose": transpose,
        "warnings": warnings,
    }
    with open(model_dir / EVALUATION_DIR / SET_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def pick_auto_trained(filelist_path, count=AUTO_TRAINED_COUNT):
    """Training clips to score as "trained" when the set has none, chosen the same way
    every time for the same filelist: evenly spaced through each speaker's sorted
    entries, speakers taken in turn."""
    by_speaker = {}
    with open(filelist_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 5 or "mute" in parts[0]:
                continue
            by_speaker.setdefault(int(parts[4]), []).append(parts)
    speakers = sorted(by_speaker)
    if not speakers:
        return []
    picks = []
    per_speaker = max(1, -(-count // len(speakers)))
    for sid in speakers:
        items = sorted(by_speaker[sid])
        step = max(1, len(items) // (per_speaker + 1))
        picks += [items[min(len(items) - 1, step * (i + 1))] for i in range(per_speaker)]
    return [
        {"group": "trained", "name": os.path.basename(p[0]), "sid": int(p[4]),
         "wav": p[0], "phone": p[1], "coarse": p[2], "f0": p[3]}
        for p in picks[:count]
    ]


# Metrics


# The band the noise metric looks at. It stops at 12 kHz because speech has next to no
# energy above that, and a spectral flatness over near-empty bins measures the noise floor
# rather than anything audible: with the band running to Nyquist, writing the original
# of a real 48 kHz training clip as 16-bit raised its flatness 1.43x (4-12 kHz: 1.005x),
# and the ranking of a run's checkpoints moved with that floor.
HF_BAND = (4000, 12000)


def voiced_hf_flatness(audio, sample_rate, f0):
    """Median spectral flatness in HF_BAND over the frames f0 calls voiced.

    High means the harmonics have turned into noise there. f0 is at 100 frames/s.
    """
    from scipy import signal

    hop = sample_rate // 100
    nperseg = 1024 if sample_rate <= 48000 else 2048
    if len(audio) < nperseg:
        return None
    f, t, Z = signal.stft(audio, sample_rate, nperseg=nperseg, noverlap=nperseg - hop)
    power = np.abs(Z) ** 2 + 1e-20
    band = (f >= HF_BAND[0]) & (f < HF_BAND[1])
    flatness = np.exp(np.mean(np.log(power[band]), 0)) / np.mean(power[band], 0)
    index = np.clip(np.round(t * 100).astype(int), 0, len(f0) - 1)
    voiced = np.asarray(f0)[index] > 0
    if voiced.sum() < 5:
        return None
    return float(np.median(flatness[voiced]))


def f0_tracking(output_f0, target_f0):
    """(median cent error over frames voiced in both, voiced/unvoiced mismatch rate)."""
    n = min(len(output_f0), len(target_f0))
    out, target = np.asarray(output_f0[:n]), np.asarray(target_f0[:n])
    both = (out > 0) & (target > 0)
    cents = float(np.median(np.abs(1200 * np.log2(out[both] / target[both])))) if both.sum() >= 5 else None
    mismatch = float(np.mean((out > 0) != (target > 0))) if n else None
    return cents, mismatch


def _training_f0_median(filelist_path, sid, limit=200):
    values = []
    with open(filelist_path, encoding="utf-8") as f:
        lines = [line.strip().split("|") for line in f]
    lines = [p for p in lines if len(p) >= 5 and int(p[4]) == sid and "mute" not in p[0]]
    for parts in lines[:: max(1, len(lines) // limit)][:limit]:
        try:
            f0 = np.load(parts[3])
        except OSError:
            continue
        values.append(f0[f0 > 0])
    values = np.concatenate(values) if values else np.array([])
    return float(np.median(values)) if values.size else None


class Evaluator:
    """Scores a generator on the cached clips of an evaluation set.

    Every clip is generated with fixed seeds inside a forked RNG, so a score depends only
    on the weights and the training run's own random stream is left exactly where it was.
    """

    def __init__(self, manifest, auto_trained, config, device, filelist_path, seeds=(0, 1)):
        self.config = config
        self.device = device
        self.seeds = seeds
        self.spec = manifest["spec"]
        self.items = []
        for entry in manifest["entries"]:
            data = np.load(entry["cache"])
            self.items.append(self._item(entry["group"], entry["name"], entry["sid"],
                                         data["phone"], data["f0"], data["coarse"], data["audio"]))
        if not any(item["group"] == "trained" for item in self.items):
            for entry in auto_trained:
                import soundfile as sf

                audio, _ = sf.read(entry["wav"], dtype="float32")
                self.items.append(self._item("trained", entry["name"], entry["sid"],
                                             np.load(entry["phone"]), np.load(entry["f0"]),
                                             np.load(entry["coarse"]), audio))
        self._prepare_convert(manifest.get("transpose") or {}, filelist_path)
        self._f0_input = None

    def _item(self, group, name, sid, phone, f0, coarse, audio):
        phone = np.repeat(phone, 2, axis=0)
        n = min(len(phone), len(f0), len(coarse), 900)
        return {"group": group, "name": name, "sid": int(sid), "phone": phone[:n],
                "f0": np.asarray(f0[:n], dtype=np.float32),
                "coarse": np.asarray(coarse[:n], dtype=np.int64),
                "audio": np.asarray(audio, dtype=np.float32), "n": n}

    def _prepare_convert(self, transpose, filelist_path):
        from rvc.lib.predictors.f0_quantization import quantize_f0

        converts = [item for item in self.items if item["group"] == "convert"]
        self.items = [item for item in self.items if item["group"] != "convert"]
        targets = sorted({item["sid"] for item in self.items}) or [0]
        medians = {sid: _training_f0_median(filelist_path, sid) for sid in targets}
        for item in converts:
            source_f0 = item["f0"]
            voiced = source_f0[source_f0 > 0]
            for sid in targets:
                name = item["name"].split("/")[-1]
                if name in transpose:
                    ratio = 2 ** (transpose[name] / 12)
                elif medians.get(sid) and voiced.size:
                    ratio = medians[sid] / float(np.median(voiced))
                else:
                    ratio = 1.0
                f0 = (source_f0 * ratio).astype(np.float32)
                self.items.append({
                    **item,
                    "sid": sid,
                    "name": f"{item['name']}->speaker{sid}",
                    "f0": f0,
                    "source_f0": source_f0,
                    "coarse": quantize_f0(f0, maximum=self.spec["f0_coarse_max"]),
                })

    @property
    def groups(self):
        return sorted({item["group"] for item in self.items}, key=GROUPS.index)

    def _f0_extractor(self):
        if self._f0_input is None:
            FeatureInput = _feature_input_class()
            self._f0_input = FeatureInput(
                f0_method=self.spec["f0_method"], device=str(self.device),
                coarse_max=self.spec["f0_coarse_max"],
            )
        return self._f0_input

    def _mel(self, audio):
        import torch

        from rvc.train.mel_processing import mel_spectrogram_torch

        data = self.config.data
        return mel_spectrogram_torch(
            torch.from_numpy(audio).float().to(self.device).unsqueeze(0),
            data.filter_length, data.n_mel_channels, data.sample_rate,
            data.hop_length, data.win_length, data.mel_fmin, data.mel_fmax,
        )

    def evaluate(self, net_g, use_amp, dtype):
        """Returns (scores, audio samples). scores are means over clips and seeds."""
        import librosa
        import torch

        sample_rate = self.config.data.sample_rate
        device = torch.device(self.device)
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        per_metric = {}
        samples = {}
        was_training = net_g.training
        net_g.eval()
        try:
            for item in self.items:
                letter = GROUP_LETTERS[item["group"]]
                for seed in self.seeds:
                    with torch.random.fork_rng(devices=fork_devices):
                        torch.manual_seed(seed)
                        with torch.no_grad(), torch.amp.autocast(
                            device_type=device.type, enabled=use_amp, dtype=dtype
                        ):
                            out, *_ = net_g.infer(
                                torch.from_numpy(item["phone"]).float().to(device).unsqueeze(0),
                                torch.tensor([item["n"]], device=device),
                                torch.from_numpy(item["coarse"]).long().to(device).unsqueeze(0),
                                torch.from_numpy(item["f0"]).float().to(device).unsqueeze(0),
                                torch.tensor([item["sid"]], device=device),
                            )
                    generated = out[0, 0].float().cpu().numpy()
                    if seed == self.seeds[0] and item["group"] not in samples:
                        samples[item["group"]] = (item["name"], generated)
                    reference = item["audio"]
                    if item["group"] != "convert":
                        n = min(len(generated), len(reference))
                        mel_gen, mel_ref = self._mel(generated[:n]), self._mel(reference[:n])
                        frames = min(mel_gen.shape[-1], mel_ref.shape[-1])
                        mel = float(torch.mean(torch.abs(mel_gen[..., :frames] - mel_ref[..., :frames])))
                        per_metric.setdefault(f"{letter}_mel", []).append(mel)
                        ref_f0 = item["f0"]
                    else:
                        ref_f0 = item["source_f0"]
                    flat_gen = voiced_hf_flatness(generated, sample_rate, item["f0"])
                    flat_ref = voiced_hf_flatness(reference, sample_rate, ref_f0)
                    if flat_gen is not None and flat_ref:
                        per_metric.setdefault(f"{letter}_noise", []).append(flat_gen / flat_ref)
                    if item["group"] == "convert":
                        audio_16k = librosa.resample(generated, orig_sr=sample_rate, target_sr=16000, res_type="soxr_hq")
                        out_f0 = self._f0_extractor().compute_f0(audio_16k)
                        cents, mismatch = f0_tracking(out_f0, item["f0"])
                        if cents is not None:
                            per_metric.setdefault("C_f0_cents", []).append(cents)
                        if mismatch is not None:
                            per_metric.setdefault("C_vuv_mismatch", []).append(mismatch)
        finally:
            if was_training:
                net_g.train()
        scores = {key: float(np.mean(values)) for key, values in per_metric.items() if values}
        return scores, samples
