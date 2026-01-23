import os
import sys
import soxr
import librosa
import soundfile as sf
import numpy as np
import re
import unicodedata
import wget
from torch import nn

import logging
from transformers import HubertModel
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


def load_embedding(embedder_model, custom_embedder=None):
    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")
    embedding_list = {
        "contentvec": os.path.join(embedder_root, "contentvec"),
        "spin": os.path.join(embedder_root, "spin"),
        "spin-v2": os.path.join(embedder_root, "spin-v2"),
        "chinese-hubert-base": os.path.join(embedder_root, "chinese_hubert_base"),
        "japanese-hubert-base": os.path.join(embedder_root, "japanese_hubert_base"),
        "japanese-hubert-large": os.path.join(embedder_root, "japanese_hubert_large"),
        "korean-hubert-base": os.path.join(embedder_root, "korean_hubert_base"),
    }

    online_embedders = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/pytorch_model.bin",
        "spin": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin/pytorch_model.bin",
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/pytorch_model.bin",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/pytorch_model.bin",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/pytorch_model.bin",
        "japanese-hubert-large": "https://huggingface.co/rinna/japanese-hubert-large/resolve/main/pytorch_model.bin",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean_hubert_base/pytorch_model.bin",
    }

    config_files = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/config.json",
        "spin": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin/config.json",
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/config.json",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/config.json",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/config.json",
        "japanese-hubert-large": "https://huggingface.co/rinna/japanese-hubert-large/resolve/main/config.json",
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
    return models


def get_embedder_dim(embedder_model: str) -> int:
    """
    Returns the output dimension for a given embedder model.

    Args:
        embedder_model (str): Name of the embedder model

    Returns:
        int: Output dimension (768 or 1024)
    """
    embedder_dims = {
        "contentvec": 768,
        "spin": 768,
        "spin-v2": 768,
        "chinese-hubert-base": 768,
        "japanese-hubert-base": 768,
        "japanese-hubert-large": 1024,
        "korean-hubert-base": 768,
    }
    return embedder_dims.get(embedder_model, 768)


def detect_vocoder_from_checkpoint(checkpoint_path: str) -> str:
    """
    Detects the vocoder type from a checkpoint file by analyzing its weight keys.

    This is useful for automatically determining the correct vocoder type when
    loading custom pretrained models, to avoid architecture mismatches.

    Args:
        checkpoint_path (str): Path to the checkpoint file (.pth)

    Returns:
        str: Detected vocoder type - one of:
            - "BigVGAN": Has Snake activation keys (dec.resblocks.X.activations.Y.act.alpha)
            - "RefineGAN": Has RefineGAN-specific keys (dec.resblocks.X.conv_blocks)
            - "MRF HiFi-GAN": Has MRF-specific keys (dec.mrfs.X with conv1/conv2)
            - "HiFi-GAN": Default/fallback for standard weight_g/weight_v patterns
    """
    import torch

    try:
        # Load only the model keys without loading all tensors to memory
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        if isinstance(ckpt, dict) and "model" in ckpt:
            model_keys = list(ckpt["model"].keys())
        elif isinstance(ckpt, dict) and "weight" in ckpt:
            # Exported model format
            model_keys = list(ckpt["weight"].keys())
        else:
            model_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else []

        del ckpt  # Free memory

        # Check for BigVGAN-specific keys (Snake/SnakeBeta activations)
        # BigVGAN uses: dec.resblocks.X.activations.Y.act.alpha/beta
        for key in model_keys:
            if "activations" in key and ".act.alpha" in key:
                return "BigVGAN"

        # Check for RefineGAN-specific keys
        # RefineGAN uses: dec.resblocks.X.conv_blocks.Y.conv.weight
        for key in model_keys:
            if "dec.resblocks" in key and "conv_blocks" in key:
                return "RefineGAN"

        # Check for MRF HiFi-GAN specific keys
        # MRF HiFi-GAN uses: dec.mrfs (ModuleList of MRFBlock) with layers.X.conv1/conv2
        # NOT to be confused with BigVGAN which uses dec.resblocks with convs1/convs2
        has_mrfs = any("dec.mrfs" in key for key in model_keys)
        has_upsamples = any("dec.upsamples" in key for key in model_keys)
        if has_mrfs and has_upsamples:
            return "MRF HiFi-GAN"

        # HiFi-GAN NSF uses dec.ups and dec.resblocks (with standard ResBlock)
        # Standard ResBlock has convs1/convs2 but no activations (uses LeakyReLU)
        has_ups = any("dec.ups" in key and "weight" in key for key in model_keys)
        has_noise_convs = any("dec.noise_convs" in key for key in model_keys)
        has_resblocks = any("dec.resblocks" in key for key in model_keys)
        if has_ups and has_noise_convs and has_resblocks:
            return "HiFi-GAN"

        # Default to HiFi-GAN for standard weight_g/weight_v patterns
        return "HiFi-GAN"

    except Exception as e:
        print(f"Warning: Could not detect vocoder from checkpoint {checkpoint_path}: {e}")
        return "HiFi-GAN"  # Default fallback


def validate_vocoder_checkpoint_match(checkpoint_path: str, selected_vocoder: str) -> tuple:
    """
    Validates that the selected vocoder matches the checkpoint's architecture.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        selected_vocoder (str): The vocoder type selected in the UI

    Returns:
        tuple: (is_valid: bool, detected_vocoder: str, message: str)
    """
    detected = detect_vocoder_from_checkpoint(checkpoint_path)

    if detected == selected_vocoder:
        return True, detected, f"Vocoder match: {selected_vocoder}"
    else:
        return False, detected, (
            f"WARNING: Vocoder mismatch detected!\n"
            f"  - Checkpoint appears to be: {detected}\n"
            f"  - Selected vocoder: {selected_vocoder}\n"
            f"  This may cause errors during training. "
            f"Consider selecting '{detected}' vocoder in the UI."
        )
