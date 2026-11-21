"""rvc/lib/key_index.py: an index searched with one embedder that retrieves another's."""

import json
from types import SimpleNamespace

import faiss
import numpy as np
import pytest
import torch

from rvc.lib import key_index


def _write_kv(folder, keys, values, name="m_key-spin-v2"):
    path = str(folder / f"{name}.index")
    index = faiss.IndexFlatL2(keys.shape[1])
    index.add(keys)
    faiss.write_index(index, path)
    np.save(folder / f"{name}.values.npy", values)
    (folder / f"{name}.index.kv.json").write_text(
        json.dumps({"key_embedder": "kushinada-hubert-large-spin-v2",
                    "key_output_layer": None, "values": f"{name}.values.npy",
                    "rows": len(keys)}))
    return path


def test_load_index_reads_values_for_a_key_index_and_vectors_otherwise(tmp_path):
    rng = np.random.default_rng(0)
    keys = rng.normal(size=(50, 6)).astype(np.float32)
    values = rng.normal(size=(50, 4)).astype(np.float32)
    path = _write_kv(tmp_path, keys, values)
    index, big_npy, spec = key_index.load_index(path)
    np.testing.assert_array_equal(big_npy, values)
    assert spec["key_embedder"] == "kushinada-hubert-large-spin-v2" and index.ntotal == 50

    plain = str(tmp_path / "m.index")
    flat = faiss.IndexFlatL2(4)
    flat.add(values)
    faiss.write_index(flat, plain)
    _, big_npy, spec = key_index.load_index(plain)
    assert spec is None
    np.testing.assert_array_equal(big_npy, values)

    np.save(tmp_path / "m_key-spin-v2.values.npy", values[:40])
    with pytest.raises(ValueError, match="rebuild"):
        key_index.load_index(path)


def _expected(query, index, big_npy, feats, rate):
    score, ix = index.search(query, k=8)
    weight = np.square(1 / score)
    weight /= weight.sum(axis=1, keepdims=True)
    return rate * np.sum(big_npy[ix] * weight[..., None], axis=1) + (1 - rate) * feats


def _retrieval_case():
    rng = np.random.default_rng(1)
    keys = rng.normal(size=(200, 6)).astype(np.float32)
    values = rng.normal(size=(200, 4)).astype(np.float32)
    index = faiss.IndexFlatL2(6)
    index.add(keys)
    feats = rng.normal(size=(1, 30, 4)).astype(np.float32)
    query = keys[rng.integers(0, 200, 30)] + 0.01 * rng.normal(size=(30, 6)).astype(np.float32)
    return index, values, feats, query


def test_offline_retrieval_searches_with_the_keys_and_returns_values():
    from rvc.infer.pipeline import Pipeline

    pipeline = Pipeline(40000, SimpleNamespace(x_pad=1, x_query=1, x_center=2, x_max=3, device="cpu"))
    index, values, feats, query = _retrieval_case()
    out = pipeline._retrieve_speaker_embeddings(
        torch.from_numpy(feats), index, values, 0.75, torch.from_numpy(query)[None]
    )
    np.testing.assert_allclose(out[0].numpy(), _expected(query, index, values, feats[0], 0.75), rtol=1e-5)

    # Without keys it is the ordinary retrieval: the features themselves search.
    plain = faiss.IndexFlatL2(4)
    plain.add(values)
    out = pipeline._retrieve_speaker_embeddings(torch.from_numpy(feats), plain, values, 0.5)
    np.testing.assert_allclose(out[0].numpy(), _expected(feats[0], plain, values, feats[0], 0.5), rtol=1e-5)

    # Keys that do not line up with the features leave them alone.
    short = torch.from_numpy(query[:20])[None]
    assert torch.equal(pipeline._retrieve_speaker_embeddings(torch.from_numpy(feats), index, values, 0.75, short),
                       torch.from_numpy(feats))


def test_realtime_retrieval_searches_with_the_keys_from_skip_head_on():
    from rvc.realtime.pipeline import Realtime_Pipeline

    index, values, feats, query = _retrieval_case()
    skip_head = 10  # 100 Hz frames; the retrieval starts at skip_head // 2
    out = Realtime_Pipeline._retrieve_speaker_embeddings(
        SimpleNamespace(device="cpu"), skip_head, torch.from_numpy(feats.copy()), index,
        values, 0.6, torch.from_numpy(query)[None],
    )[0].numpy()
    np.testing.assert_array_equal(out[:5], feats[0][:5])
    np.testing.assert_allclose(out[5:], _expected(query[5:], index, values, feats[0][5:], 0.6), rtol=1e-5)


def test_build_key_index_pairs_keys_and_values_row_for_row(tmp_path, monkeypatch):
    import rvc.lib.utils as utils

    folder = tmp_path / "m"
    (folder / "extracted").mkdir(parents=True)
    (folder / "sliced_audios_16k").mkdir()
    rng = np.random.default_rng(2)
    audio, values = {}, []
    for i in range(6):
        frames = 60 + 7 * i
        audio[f"0_{i}_0.wav"] = rng.normal(size=frames * 320).astype(np.float32)
        value = rng.normal(size=(frames, 4)).astype(np.float32)
        np.save(folder / "extracted" / f"0_{i}_0.npy", value)
        (folder / "sliced_audios_16k" / f"0_{i}_0.wav").write_bytes(b"")
    names = sorted(audio)
    values = [np.load(folder / "extracted" / n.replace(".wav", ".npy")) for n in names]

    class FakeEmbedder(torch.nn.Module):
        def forward(self, x):  # one 3-wide key per 320 samples: a function of the audio
            return x.view(1, -1, 320)[..., :3]

    monkeypatch.setattr(utils, "load_embedding", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr(utils, "embedder_forward", lambda model, x: model(x))
    monkeypatch.setattr(utils, "load_audio_16k", lambda path: audio[path.replace("\\", "/").rsplit("/", 1)[1]])

    path, spec = key_index.build_key_index(str(folder), device="cpu")
    assert path.endswith("m_key-spin-v2.index") and spec["rows"] == sum(len(v) for v in values)
    index, big_npy, loaded = key_index.load_index(path)
    np.testing.assert_array_equal(big_npy, np.concatenate(values))
    assert loaded["key_embedder"] == key_index.DEFAULT_KEY_EMBEDDER
    # An exact key finds its own row, whose value is that frame's training feature.
    faiss.extract_index_ivf(index).nprobe = index.nlist
    row = 70
    first = names[0]
    key = audio[first].reshape(-1, 320)[:, :3]
    flat_keys = np.concatenate([audio[n].reshape(-1, 320)[:, :3] for n in names])
    _, ix = index.search(flat_keys[row:row + 1], k=1)
    assert ix[0, 0] == row
    assert key.shape[0] == len(values[0])


def test_a_second_key_index_shares_the_values_and_reads_its_embedder_from_a_folder(tmp_path, monkeypatch):
    import rvc.lib.utils as utils

    folder = tmp_path / "m"
    (folder / "extracted").mkdir(parents=True)
    (folder / "sliced_audios_16k").mkdir()
    rng = np.random.default_rng(3)
    audio = {}
    for i in range(6):
        frames = 60 + 7 * i
        audio[f"0_{i}_0.wav"] = rng.normal(size=frames * 320).astype(np.float32)
        np.save(folder / "extracted" / f"0_{i}_0.npy", rng.normal(size=(frames, 4)).astype(np.float32))
        (folder / "sliced_audios_16k" / f"0_{i}_0.wav").write_bytes(b"")

    class FakeEmbedder(torch.nn.Module):
        def forward(self, x):
            return x.view(1, -1, 320)[..., :3]

    loads = []

    def load(name, path=None, layer=None):
        loads.append((name, path))
        return FakeEmbedder()

    monkeypatch.setattr(utils, "load_embedding", load)
    monkeypatch.setattr(utils, "embedder_forward", lambda model, x: model(x))
    monkeypatch.setattr(utils, "load_audio_16k", lambda path: audio[path.replace("\\", "/").rsplit("/", 1)[1]])

    _, first_spec = key_index.build_key_index(str(folder), device="cpu")
    with pytest.raises(FileNotFoundError, match="not found"):
        key_index.build_key_index(str(folder), "spin-x", device="cpu", key_embedder_path=str(tmp_path / "gone"))

    embedder = tmp_path / "embedder"
    embedder.mkdir()
    (embedder / "config.json").write_text("{}")
    second, spec = key_index.build_key_index(str(folder), "spin-x", device="cpu", key_embedder_path=str(embedder))
    assert second.endswith("m_key-spin-x.index")
    assert spec["values"] == first_spec["values"]
    assert len(list(folder.glob("*.values.npy"))) == 1
    assert loads[-1] == ("custom", str(embedder))
    _, big_npy, loaded = key_index.load_index(second)
    assert len(big_npy) == spec["rows"]

    # Two folders under one name stay apart: the cache goes by the folder.
    key_index._key_models.clear()
    key_index.load_key_embedder(loaded, "cpu")
    key_index.load_key_embedder(dict(loaded), "cpu")
    assert len(key_index._key_models) == 1
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text("{}")
    key_index.load_key_embedder(dict(loaded, key_embedder_path=str(other)), "cpu")
    assert len(key_index._key_models) == 2
    key_index._key_models.clear()
