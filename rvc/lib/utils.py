import os
import sys
import soxr
import librosa
import soundfile as sf
import numpy as np
import re
import unicodedata
import wget
import torch
from torch import nn

import logging
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import warnings

# Remove this to see warnings about transformers models
warnings.filterwarnings("ignore")

logging.getLogger("fairseq").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

now_dir = os.getcwd()
sys.path.append(now_dir)

base_path = os.path.join(now_dir, "rvc", "models", "formant", "stftpitchshift")
stft = base_path + ".exe" if sys.platform == "win32" else base_path

JAPANESE_HUBERT_BASE_K2 = "japanese-hubert-base-k2"
JAPANESE_HUBERT_BASE_K2_REPO = "reazon-research/japanese-hubert-base-k2"
# Pinned so an upstream update cannot silently replace the weights under a model that
# was already trained, and so a cached load needs no network round trip.
JAPANESE_HUBERT_BASE_K2_REVISION = "a9f26026165f8b80256f0aeecee53dedf81abce1"

# k2's final LayerNorm gain is roughly 10x smaller than every other embedder here, so its
# last_hidden_state lands at about a tenth of the magnitude: 0.64 per frame against 6.49
# for japanese-hubert-base and 9.31 for contentvec. RVC v2's TextEncoder adds
# emb_phone(feature) straight onto a scale free emb_pitch embedding, so without this the
# content term ends up about 5.7x under weighted, and emb_phone never catches up because
# its gradient scales with the input magnitude as well.
EMBEDDER_FEATURE_SCALE = {JAPANESE_HUBERT_BASE_K2: 10.0}


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def load_audio_16k(file):
    # this is used by f0 and feature extractions that load preprocessed 16k files, so there's no need to resample
    try:
        audio, sr = librosa.load(file, sr=16000)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq"
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio_infer(
    file,
    sample_rate,
    **kwargs,
):
    formant_shifting = kwargs.get("formant_shifting", False)
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File not found: {file}")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq"
            )
        if formant_shifting:
            formant_qfrency = kwargs.get("formant_qfrency", 0.8)
            formant_timbre = kwargs.get("formant_timbre", 0.8)

            from stftpitchshift import StftPitchShift

            pitchshifter = StftPitchShift(1024, 32, sample_rate)
            audio = pitchshifter.shiftpitch(
                audio,
                factors=1,
                quefrency=formant_qfrency * 1e-3,
                distortion=formant_timbre,
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")
    return np.array(audio).flatten()


def format_title(title):
    formatted_title = unicodedata.normalize("NFC", title)
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title, flags=re.UNICODE)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title


def _attach_input_preprocessing(model, repo_id, cache_dir, revision=None):
    """Record the embedder's official feature extractor settings on the model.

    Only used for repos that ship a preprocessor_config.json. Every other embedder keeps
    input_do_normalize = False, so its raw waveform is passed through unchanged.
    """
    do_normalize, sampling_rate = True, 16000
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            repo_id, cache_dir=cache_dir, revision=revision
        )
        do_normalize = bool(feature_extractor.do_normalize)
        sampling_rate = int(feature_extractor.sampling_rate)
    except Exception as error:
        print(
            f"Could not read the feature extractor config of {repo_id} ({error}), "
            f"falling back to do_normalize={do_normalize}, sampling_rate={sampling_rate}."
        )
    if sampling_rate != 16000:
        print(
            f"Warning: {repo_id} expects {sampling_rate} Hz audio, but features are "
            "always extracted at 16000 Hz."
        )
    model.input_do_normalize = do_normalize
    model.input_sampling_rate = sampling_rate
    return model


def apply_embedder_input_normalization(model, feats):
    """Zero-mean / unit-variance the waveform when the embedder asks for it.

    Equivalent to Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm for a single unbatched
    array, which is what every embedder call site passes. Embedders whose official config
    does not set do_normalize are returned untouched.

    For a feat_extract_norm="group" model with conv_bias=False, which is every embedder
    here including k2, the GroupNorm right after the first bias free conv already cancels
    any scalar gain, so this is close to a no-op (measured: under 0.5% feature change). It
    is kept because it is what the official preprocessor_config.json asks for, and a
    future feat_extract_norm="layer" embedder would genuinely need it.
    """
    if not getattr(model, "input_do_normalize", False):
        return feats
    feats = feats.float()
    return (feats - feats.mean(dim=-1, keepdim=True)) / torch.sqrt(
        feats.var(dim=-1, unbiased=False, keepdim=True) + 1e-7
    )


def apply_embedder_feature_scale(model, feats):
    """Bring the embedder's hidden states into the range RVC v2 was designed for.

    Only embedders listed in EMBEDDER_FEATURE_SCALE are touched; every other one carries
    feature_scale = 1.0 and is handed back as the very same object.
    """
    scale = getattr(model, "feature_scale", 1.0)
    return feats if scale == 1.0 else feats * scale


def warn_on_feature_scale_mismatch(model, checkpoint):
    """Warn when a checkpoint was trained against a different embedder feature scale.

    A checkpoint that records an embedder but no scale was trained before scaling existed,
    which is exactly a scale of 1.0. Feeding such a model features at another scale gives
    garbage rather than an error, so say so instead of failing silently.
    """
    if not isinstance(checkpoint, dict) or checkpoint.get("embedder_model") is None:
        return
    recorded = checkpoint.get("embedder_feature_scale", 1.0)
    current = getattr(model, "feature_scale", 1.0)
    if recorded != current:
        print(
            f"Warning: this checkpoint was trained on '{checkpoint['embedder_model']}' "
            f"features scaled by {recorded}, but they are now scaled by {current}. "
            "Re-extract the features and retrain, or the output will be wrong."
        )


def load_embedding(embedder_model, custom_embedder=None):
    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")
    embedding_list = {
        "contentvec": os.path.join(embedder_root, "contentvec"),
        "spin": os.path.join(embedder_root, "spin"),
        "spin-v2": os.path.join(embedder_root, "spin-v2"),
        "chinese-hubert-base": os.path.join(embedder_root, "chinese_hubert_base"),
        "japanese-hubert-base": os.path.join(embedder_root, "japanese_hubert_base"),
        JAPANESE_HUBERT_BASE_K2: os.path.join(embedder_root, "japanese_hubert_base_k2"),
        "korean-hubert-base": os.path.join(embedder_root, "korean_hubert_base"),
    }

    online_embedders = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/pytorch_model.bin",
        "spin": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin/pytorch_model.bin",
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/pytorch_model.bin",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/pytorch_model.bin",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/pytorch_model.bin",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean_hubert_base/pytorch_model.bin",
    }

    config_files = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/config.json",
        "spin": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin/config.json",
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/config.json",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/config.json",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/config.json",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean_hubert_base/config.json",
    }

    if embedder_model == "custom":
        if os.path.exists(custom_embedder):
            model_path = custom_embedder
        else:
            print(f"Custom embedder not found: {custom_embedder}, using contentvec")
            model_path = embedding_list["contentvec"]
    else:
        model_path = embedding_list[embedder_model]
        if embedder_model == JAPANESE_HUBERT_BASE_K2:
            os.makedirs(model_path, exist_ok=True)
            models = HubertModelWithFinalProj.from_pretrained(
                JAPANESE_HUBERT_BASE_K2_REPO,
                cache_dir=model_path,
                revision=JAPANESE_HUBERT_BASE_K2_REVISION,
                use_safetensors=True,
            )
            models.feature_scale = EMBEDDER_FEATURE_SCALE.get(embedder_model, 1.0)
            return _attach_input_preprocessing(
                models,
                JAPANESE_HUBERT_BASE_K2_REPO,
                model_path,
                JAPANESE_HUBERT_BASE_K2_REVISION,
            )

        bin_file = os.path.join(model_path, "pytorch_model.bin")
        json_file = os.path.join(model_path, "config.json")
        os.makedirs(model_path, exist_ok=True)
        if not os.path.exists(bin_file):
            url = online_embedders[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=bin_file)
        if not os.path.exists(json_file):
            url = config_files[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=json_file)

    models = HubertModelWithFinalProj.from_pretrained(model_path)
    # These embedders ship no preprocessor_config.json and have always been fed the
    # raw waveform, so their input pipeline stays exactly as it was.
    models.input_do_normalize = False
    models.input_sampling_rate = 16000
    models.feature_scale = EMBEDDER_FEATURE_SCALE.get(embedder_model, 1.0)
    return models
