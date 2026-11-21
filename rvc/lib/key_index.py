"""Key/value retrieval indexes: search with one embedder, retrieve the model's features.

An ordinary RVC index holds a model's training features and is searched with the
source's features from the same embedder, so the match is pulled by whatever of the
source speaker's timbre those features still carry. A key/value index is searched with
another embedder's features instead - the keys, e.g. SPIN v2, which place the same sound
from different speakers closer together (ITA same-sentence / different-sentence distance
ratio 0.307 at kushinada layer 22 against 0.226 for SPIN) - and returns the model's own
training features, row for row - the values. The model still only ever sees features
from the embedder it was trained on, so SPIN's lost spectral detail never reaches it
(logs/_spin/RESULTS.md 16 and 18).

On disk it is an ordinary FAISS file over the keys, which is what the index dropdowns
list, next to two files the dropdowns do not list:
    <name>.index            FAISS IVF over the keys, ids = rows of the values
    <name>.index.kv.json    {"key_embedder", "key_output_layer", "values", "rows", ...}
    <name>.values.npy       the values, float32, one row per key
An index without the .kv.json is an ordinary one and loads exactly as before.

Every key index of a model folder has the same values, so a later one points at the
file an earlier one wrote instead of writing its own. The key embedder is a registered
name or, for an experiment that should not be registered, a folder written by
tools/spin/export.py ("key_embedder_path", relative to the repository).
"""

import json
import os

import numpy as np

SIDECAR_SUFFIX = ".kv.json"
DEFAULT_KEY_EMBEDDER = "kushinada-hubert-large-spin-v2"
# Short names for the file name: <model>_key-<short>.index
KEY_EMBEDDER_SHORT = {
    "kushinada-hubert-large-spin-v2": "spin-v2",
    "kushinada-hubert-large-spin": "spin-v1",
}

_key_models = {}


def sidecar_path(index_path):
    return index_path + SIDECAR_SUFFIX


def read_key_spec(index_path):
    """The key/value description next to an index, or None for an ordinary index."""
    path = sidecar_path(index_path)
    if not index_path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    spec["values_path"] = os.path.join(os.path.dirname(index_path), spec["values"])
    return spec


def load_index(index_path):
    """(index, big_npy, key_spec). big_npy is what a search's ids index into: the
    index's own vectors for an ordinary index, the values for a key/value one."""
    import faiss

    index = faiss.read_index(index_path)
    spec = read_key_spec(index_path)
    if spec is None:
        return index, index.reconstruct_n(0, index.ntotal), None
    values = np.load(spec["values_path"])
    if len(values) != index.ntotal:
        raise ValueError(
            f"{spec['values_path']} has {len(values)} rows but {index_path} holds "
            f"{index.ntotal} keys; rebuild the key index."
        )
    return index, values, spec


def load_key_embedder(spec, device, dtype=None):
    """The embedder that makes a key index's keys, cached per (embedder, layer, device,
    dtype) so offline conversions do not reload it every time."""
    import torch

    path = spec.get("key_embedder_path")
    key = (path or spec["key_embedder"], spec.get("key_output_layer"), str(device), str(dtype))
    model = _key_models.get(key)
    if model is None:
        model = _load_key_model(spec["key_embedder"], path, spec.get("key_output_layer"))
        model = model.to(device=device, dtype=dtype or torch.float32).eval()
        _key_models[key] = model
    return model


def _load_key_model(key_embedder, path, output_layer):
    from rvc.lib.utils import load_embedding

    if path is None:
        return load_embedding(key_embedder, None, output_layer)
    # load_embedding falls back to contentvec for a custom folder it cannot find.
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise FileNotFoundError(f"Key embedder folder not found: {path}")
    return load_embedding("custom", path, output_layer)


def key_index_path(model_folder, key_embedder=DEFAULT_KEY_EMBEDDER):
    name = os.path.basename(os.path.normpath(model_folder))
    short = KEY_EMBEDDER_SHORT.get(key_embedder, key_embedder)
    return os.path.join(model_folder, f"{name}_key-{short}.index")


def _matching_values(model_folder, values):
    """The name of a values file in the folder that holds exactly these rows, or None."""
    for name in sorted(os.listdir(model_folder)):
        if name.endswith(".values.npy"):
            existing = np.load(os.path.join(model_folder, name), mmap_mode="r")
            if existing.shape == values.shape and np.array_equal(existing, values):
                return name
    return None


def build_key_index(model_folder, key_embedder=DEFAULT_KEY_EMBEDDER, key_output_layer=None,
                    device="cuda", key_embedder_path=None):
    """Build <model>_key-<short>.index (+ .kv.json, .values.npy) in a model folder.
    With key_embedder_path, key_embedder is only the name the files and logs go by.

    Keys come from the folder's own training slices (sliced_audios_16k), through the key
    embedder exactly as training extraction runs an embedder; values are the folder's
    extracted features (extracted/*.npy), the files extract_index.py builds the ordinary
    index from, in the same order. The FAISS index is built as extract_index.py builds
    one (IVF, n_ivf = min(16 sqrt(N), N // 39), Flat, nprobe 1), without KMeans, so each
    id is one training frame. Only reads the folder's existing files.
    """
    import faiss
    import torch

    from rvc.lib.utils import embedder_forward, load_audio_16k

    feature_dir = os.path.join(model_folder, "extracted")
    names = sorted(n for n in os.listdir(feature_dir) if n.endswith(".npy"))
    if not names:
        raise FileNotFoundError(f"No extracted features in {feature_dir}")
    model = _load_key_model(key_embedder, key_embedder_path, key_output_layer).to(device).eval()
    keys, values, trimmed = [], [], 0
    with torch.no_grad():
        for name in names:
            value = np.load(os.path.join(feature_dir, name))
            wav = os.path.join(model_folder, "sliced_audios_16k", name[:-4] + ".wav")
            x = torch.from_numpy(load_audio_16k(wav)).to(device).float().view(1, -1)
            key = embedder_forward(model, x)[0].float().cpu().numpy()
            n = min(len(key), len(value))
            trimmed += abs(len(key) - len(value))
            keys.append(key[:n])
            values.append(value[:n].astype(np.float32))
    del model
    keys, values = np.concatenate(keys), np.concatenate(values)
    n_ivf = min(int(16 * np.sqrt(len(keys))), len(keys) // 39)
    index = faiss.index_factory(keys.shape[1], f"IVF{n_ivf},Flat")
    faiss.extract_index_ivf(index).nprobe = 1
    index.train(keys[np.random.default_rng(0).permutation(len(keys))])
    for i in range(0, len(keys), 8192):
        index.add(keys[i:i + 8192])

    path = key_index_path(model_folder, key_embedder)
    faiss.write_index(index, path)
    values_name = _matching_values(model_folder, values)
    if values_name is None:
        values_name = os.path.basename(path)[: -len(".index")] + ".values.npy"
        np.save(os.path.join(model_folder, values_name), values)
    spec = {
        "key_embedder": key_embedder,
        "key_embedder_path": key_embedder_path,
        "key_output_layer": key_output_layer,
        "values": values_name,
        "rows": int(len(keys)),
        "files": len(names),
        "frames_trimmed": int(trimmed),
        "n_ivf": n_ivf,
    }
    with open(sidecar_path(path), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    return path, spec
