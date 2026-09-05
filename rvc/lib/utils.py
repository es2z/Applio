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
JAPANESE_HUBERT_LARGE = "japanese-hubert-large"

APPLIO_EMBEDDER_URL = (
    "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders"
)

# One entry per embedder. "dir" is the folder under rvc/models/embedders; it doubles as
# the folder name on IAHispano/Applio for the legacy pytorch_model.bin embedders.
# "repo"/"revision" switch an entry to the transformers loader instead, pinned to a full
# commit SHA so an upstream update cannot silently replace the weights under a model that
# was already trained, and so a cached load needs no network round trip.
# "feature_scale" defaults to 1.0; see apply_embedder_feature_scale for what it is for.
EMBEDDERS = {
    "contentvec": {"dir": "contentvec"},
    "spin": {"dir": "spin"},
    "spin-v2": {"dir": "spin-v2"},
    "chinese-hubert-base": {"dir": "chinese_hubert_base"},
    "japanese-hubert-base": {"dir": "japanese_hubert_base"},
    "korean-hubert-base": {"dir": "korean_hubert_base"},
    JAPANESE_HUBERT_BASE_K2: {
        "dir": "japanese_hubert_base_k2",
        "repo": "reazon-research/japanese-hubert-base-k2",
        "revision": "a9f26026165f8b80256f0aeecee53dedf81abce1",
        # k2's hidden states land about a tenth of everyone else's: 0.58 per frame against
        # 6.35 for japanese-hubert-base and 9.82 for contentvec, because its final
        # LayerNorm gain is that much smaller. RVC v2's TextEncoder adds emb_phone(feature)
        # straight onto a scale free emb_pitch embedding, so without this the content term
        # ends up heavily under weighted, and emb_phone never catches up because its
        # gradient scales with the input magnitude as well.
        "feature_scale": 10.0,
    },
    JAPANESE_HUBERT_LARGE: {
        "dir": "japanese_hubert_large",
        # Mirror of rinna/japanese-hubert-large, which the HF API serves only to logged in
        # clients. Apache-2.0, 24 layers, hidden size 1024.
        "repo": "yky-h/japanese-hubert-large",
        "revision": "bccd07ba8a9f025576d53ca84669c540c7ef204e",
        # Measured at 5.95 per frame, right next to japanese-hubert-base's 6.35, so it
        # needs no correction. The 1024 wide features are handled by text_enc_hidden_dim.
    },
}

JAPANESE_HUBERT_BASE_K2_REPO = EMBEDDERS[JAPANESE_HUBERT_BASE_K2]["repo"]
JAPANESE_HUBERT_BASE_K2_REVISION = EMBEDDERS[JAPANESE_HUBERT_BASE_K2]["revision"]
JAPANESE_HUBERT_LARGE_REPO = EMBEDDERS[JAPANESE_HUBERT_LARGE]["repo"]
JAPANESE_HUBERT_LARGE_REVISION = EMBEDDERS[JAPANESE_HUBERT_LARGE]["revision"]

EMBEDDER_FEATURE_SCALE = {
    name: spec["feature_scale"]
    for name, spec in EMBEDDERS.items()
    if "feature_scale" in spec
}

# Floor on the standard deviation used by the do_normalize step, as a fraction of full
# scale. Without it, zero-mean/unit-variance normalisation lifts a near silent window to
# speech level: measured, -60 dBFS room tone comes out at 0.95 RMS, a gain of about 60 dB.
# The embedder then reads that amplified noise as speech and the decoder synthesises
# voiced rubbish over the silence. Speech itself sits far above the floor and is
# untouched. Set to 0.0 for the literal Wav2Vec2FeatureExtractor behaviour.
EMBEDDER_INPUT_STD_FLOOR = 0.01


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


def _attach_input_preprocessing(
    model, repo_id, cache_dir, revision=None, default_do_normalize=True
):
    """Record the embedder's official feature extractor settings on the model.

    Only used for embedders that ship a preprocessor_config.json. Every other one keeps
    input_do_normalize = False, so its raw waveform is passed through unchanged.

    default_do_normalize is what to assume when the config cannot be read at all. For the
    known repos that is True, which is what they document. For a custom folder it is
    False, so an unreadable config leaves custom behaving exactly as it always has rather
    than silently switching its features to a normalised waveform.
    """
    do_normalize, sampling_rate = default_do_normalize, 16000
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
    array, which is what every embedder call site passes, plus the standard deviation
    floor described at EMBEDDER_INPUT_STD_FLOOR. Embedders whose official config does not
    set do_normalize are returned untouched.

    How much this matters depends entirely on the embedder's feature encoder. For
    feat_extract_norm="group" with conv_bias=False - contentvec, japanese-hubert-base, k2 -
    the GroupNorm after the first bias free conv already cancels any scalar gain, so it is
    close to a no-op (measured: 0.29%, 4.99% and 0.21% feature change). For
    feat_extract_norm="layer" with conv_bias=True - japanese-hubert-large - nothing cancels
    the gain and skipping this changes the features by 59%.
    """
    if not getattr(model, "input_do_normalize", False):
        return feats
    feats = feats.float()
    std = torch.sqrt(feats.var(dim=-1, unbiased=False, keepdim=True) + 1e-7)
    if EMBEDDER_INPUT_STD_FLOOR > 0:
        std = std.clamp(min=EMBEDDER_INPUT_STD_FLOOR)
    return (feats - feats.mean(dim=-1, keepdim=True)) / std


def apply_embedder_feature_scale(model, feats):
    """Bring the embedder's hidden states into the range RVC v2 was designed for.

    Only embedders carrying a feature_scale in EMBEDDERS are touched; every other one
    carries 1.0 and is handed back as the very same object.
    """
    scale = getattr(model, "feature_scale", 1.0)
    return feats if scale == 1.0 else feats * scale


def embedder_forward(model, feats):
    """Run the embedder the one way every call site should run it.

    Training extraction, offline inference and realtime all need the same input
    normalisation, layer choice and feature scaling, so they all come through here.

    output_layer is None for every embedder by default, which is plain last_hidden_state.
    Picking an earlier layer is only meaningful for a deep embedder like
    japanese-hubert-large, where phonetic content peaks before the top while speaker
    identity is strongest near the bottom. For a do_stable_layer_norm model the
    intermediate hidden states are raw pre-norm residuals - measured on
    japanese-hubert-large they run from 69 per frame at layer 0 to 538 at layer 23,
    against 5.9 for last_hidden_state - so the encoder's final LayerNorm is applied to
    bring a chosen intermediate layer back into the same range.
    """
    feats = apply_embedder_input_normalization(model, feats)
    # Realtime can run the embedder in bf16/fp16 while the rest of the pipeline stays
    # float32, so match whatever the weights are rather than assuming.
    parameter = next(model.parameters(), None) if isinstance(model, nn.Module) else None
    if parameter is not None and feats.dtype != parameter.dtype:
        feats = feats.to(parameter.dtype)
    layer = getattr(model, "output_layer", None)
    n_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if layer is None or n_layers is None or layer >= n_layers:
        out = model(feats)["last_hidden_state"]
    else:
        out = model(feats, output_hidden_states=True)["hidden_states"][layer]
        if getattr(model.config, "do_stable_layer_norm", False):
            out = model.encoder.layer_norm(out)
    return apply_embedder_feature_scale(model, out)


def checkpoint_text_enc_hidden_dim(checkpoint):
    """Width of the features a checkpoint's TextEncoder expects.

    enc_p.emb_phone is Linear(feature_dim, hidden_channels), so its weight carries the
    answer for every checkpoint ever saved and needs no metadata migration. That covers
    v1 too, where final_proj has already narrowed the features to 256. The recorded key
    and the old version based guess are only fallbacks for a checkpoint that somehow has
    no emb_phone weight.
    """
    weight = checkpoint.get("weight", {}).get("enc_p.emb_phone.weight")
    if weight is not None:
        return weight.shape[1]
    recorded = checkpoint.get("text_enc_hidden_dim")
    if recorded is not None:
        return recorded
    return 768 if checkpoint.get("version", "v1") == "v2" else 256


def embedder_identity(model, embedder_name=None):
    """The properties of an embedder that its extracted features depend on.

    Any of these changing means every .npy, the FAISS index and enc_p.emb_phone are all
    stale together, so this travels with the model info, the resume checkpoints and the
    exported .pth.
    """
    return {
        "embedder_model": embedder_name,
        "embedder_feature_scale": getattr(model, "feature_scale", 1.0),
        "embedder_output_layer": getattr(model, "output_layer", None),
        "embedder_dim": getattr(model, "embed_dim", None),
        # Only meaningful when the waveform is normalised at all. Measured on
        # japanese-hubert-large, changing the floor moves a quiet window's features by
        # over 50%, so it belongs with the rest of the identity.
        "embedder_input_std_floor": (
            EMBEDDER_INPUT_STD_FLOOR
            if getattr(model, "input_do_normalize", False)
            else None
        ),
    }


def describe_embedder_output_layer(layer):
    return "last" if layer is None else str(layer)


def describe_embedder_mismatch(recorded, current, subject="it"):
    """Return a human readable reason two embedder identities differ, or None.

    A record that names no embedder at all predates any of this being tracked and cannot
    be compared, so it never reports a mismatch. Within a record that does name one, a
    missing feature scale means 1.0 and a missing output layer means the last layer,
    because that is what they defaulted to before they were written down. A missing
    feature dimension is genuinely unknown and is skipped.
    """
    if recorded.get("embedder_model") is None:
        return None

    reasons = []
    if current.get("embedder_model") != recorded["embedder_model"]:
        reasons.append(
            f"embedder '{recorded['embedder_model']}' -> "
            f"'{current.get('embedder_model')}'"
        )

    was_scale = recorded.get("embedder_feature_scale", 1.0)
    now_scale = current.get("embedder_feature_scale", 1.0)
    if was_scale != now_scale:
        reasons.append(f"feature scale {was_scale} -> {now_scale}")

    was_layer = recorded.get("embedder_output_layer")
    now_layer = current.get("embedder_output_layer")
    if was_layer != now_layer:
        reasons.append(
            f"output layer {describe_embedder_output_layer(was_layer)} -> "
            f"{describe_embedder_output_layer(now_layer)}"
        )

    was_dim, now_dim = recorded.get("embedder_dim"), current.get("embedder_dim")
    if was_dim is not None and now_dim is not None and was_dim != now_dim:
        reasons.append(f"feature dimension {was_dim} -> {now_dim}")

    was_floor = recorded.get("embedder_input_std_floor")
    now_floor = current.get("embedder_input_std_floor")
    if was_floor is not None and now_floor is not None and was_floor != now_floor:
        reasons.append(f"input std floor {was_floor} -> {now_floor}")

    if not reasons:
        return None
    return f"{subject} was built with a different embedder ({'; '.join(reasons)})"


def warn_on_feature_scale_mismatch(model, checkpoint):
    """Warn when a checkpoint was trained against different embedder settings.

    A checkpoint that records an embedder but no scale was trained before scaling existed,
    which is exactly a scale of 1.0. Feeding such a model features from a differently
    configured embedder gives garbage rather than an error, so say so instead of failing
    silently.
    """
    if not isinstance(checkpoint, dict) or checkpoint.get("embedder_model") is None:
        return
    recorded = {
        "embedder_model": checkpoint["embedder_model"],
        "embedder_feature_scale": checkpoint.get("embedder_feature_scale", 1.0),
        "embedder_output_layer": checkpoint.get("embedder_output_layer"),
        "embedder_dim": checkpoint.get("text_enc_hidden_dim"),
        "embedder_input_std_floor": checkpoint.get("embedder_input_std_floor"),
    }
    current = embedder_identity(model, checkpoint["embedder_model"])
    reason = describe_embedder_mismatch(recorded, current, "this checkpoint")
    if reason:
        print(
            f"Warning: {reason}. Re-extract the features and retrain, or the output "
            "will be wrong."
        )


def _finalize_embedder(model, spec, output_layer=None):
    """Attach everything the rest of the pipeline reads off an embedder."""
    config = getattr(model, "config", None)
    model.embed_dim = getattr(config, "hidden_size", None)
    model.feature_scale = spec.get("feature_scale", 1.0)
    n_layers = getattr(config, "num_hidden_layers", None)
    if output_layer is not None and n_layers is not None and output_layer >= n_layers:
        output_layer = None
    model.output_layer = output_layer
    if isinstance(model, nn.Module):
        # layerdrop and spec augment are gated on module.training, and leaving them live
        # would randomly drop whole transformer layers during extraction.
        model.eval()
    return model


def load_embedding(embedder_model, custom_embedder=None, output_layer=None):
    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")

    if embedder_model == "custom":
        if custom_embedder and os.path.exists(custom_embedder):
            model_path = custom_embedder
        else:
            print(f"Custom embedder not found: {custom_embedder}, using contentvec")
            model_path = os.path.join(embedder_root, EMBEDDERS["contentvec"]["dir"])
        model = HubertModelWithFinalProj.from_pretrained(model_path)
        # A custom folder may or may not ship a preprocessor_config.json. Read it when it
        # is there, so a feat_extract_norm="layer" embedder is handled correctly, and fall
        # back to the raw waveform when it is not, which is how custom has always behaved.
        if os.path.isfile(os.path.join(model_path, "preprocessor_config.json")):
            _attach_input_preprocessing(
                model, model_path, None, default_do_normalize=False
            )
        else:
            model.input_do_normalize = False
            model.input_sampling_rate = 16000
        return _finalize_embedder(model, {}, output_layer)

    spec = EMBEDDERS[embedder_model]
    model_path = os.path.join(embedder_root, spec["dir"])
    os.makedirs(model_path, exist_ok=True)

    if "repo" in spec:
        model = HubertModelWithFinalProj.from_pretrained(
            spec["repo"],
            cache_dir=model_path,
            revision=spec["revision"],
            use_safetensors=True,
        )
        _attach_input_preprocessing(model, spec["repo"], model_path, spec["revision"])
        return _finalize_embedder(model, spec, output_layer)

    bin_file = os.path.join(model_path, "pytorch_model.bin")
    json_file = os.path.join(model_path, "config.json")
    for target, filename in ((bin_file, "pytorch_model.bin"), (json_file, "config.json")):
        if not os.path.exists(target):
            url = f"{APPLIO_EMBEDDER_URL}/{spec['dir']}/{filename}"
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=target)

    model = HubertModelWithFinalProj.from_pretrained(model_path)
    # These embedders ship no preprocessor_config.json and have always been fed the
    # raw waveform, so their input pipeline stays exactly as it was.
    model.input_do_normalize = False
    model.input_sampling_rate = 16000
    return _finalize_embedder(model, spec, output_layer)
