"""The opt-in exact index search (rvc/realtime/index_search.py, option `index_search`).

Runs on the CPU with synthetic vectors: clustered like speech features are (many frames
around a limited number of phonetic centres), so an IVF index really does file close
neighbours into different cells.
"""

import types
import unittest

import faiss
import numpy as np
import torch

from rvc.realtime.index_search import ExactIndex
from rvc.realtime.options import RealtimeOptions

DIM = 64


def clustered(count, centres, seed, spread=0.6):
    rng = np.random.default_rng(seed)
    centre = rng.standard_normal((centres, DIM)).astype(np.float32)
    rows = centre[rng.integers(0, centres, count)]
    return (rows + spread * rng.standard_normal((count, DIM))).astype(np.float32)


def legacy_blend(index, big_npy, queries, k=8):
    """The FAISS path of Realtime_Pipeline._retrieve_speaker_embeddings, verbatim."""
    score, ix = index.search(queries, k=k)
    weight = np.square(1 / score)
    weight /= weight.sum(axis=1, keepdims=True)
    return np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)


class ExactIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vectors = clustered(6000, 40, seed=0)
        cls.queries = clustered(200, 40, seed=1)
        cls.flat = faiss.IndexFlatL2(DIM)
        cls.flat.add(cls.vectors)

    def test_it_finds_what_a_full_faiss_search_finds(self):
        distances, indices = ExactIndex(self.vectors, "cpu", 8).search(
            torch.from_numpy(self.queries)
        )
        flat_distances, flat_indices = self.flat.search(self.queries, 8)
        np.testing.assert_array_equal(indices.numpy(), flat_indices)
        np.testing.assert_allclose(
            distances.numpy(), flat_distances, rtol=1e-3, atol=1e-3
        )

    def test_the_blend_is_the_legacy_formula_over_those_neighbours(self):
        exact = ExactIndex(self.vectors, "cpu", 8).retrieve(
            torch.from_numpy(self.queries)
        )
        expected = legacy_blend(self.flat, self.vectors, self.queries)
        np.testing.assert_allclose(exact.numpy(), expected, rtol=1e-3, atol=1e-3)

    def test_an_ivf_index_at_one_probe_misses_neighbours_it_keeps(self):
        # What Applio writes: IVF{n},Flat with nprobe 1.
        ivf = faiss.index_factory(DIM, "IVF60,Flat")
        ivf.train(self.vectors)
        ivf.add(self.vectors)
        ivf.nprobe = 1
        _, approximate = ivf.search(self.queries, 8)
        _, exact = ExactIndex(self.vectors, "cpu", 8).search(
            torch.from_numpy(self.queries)
        )
        nearest_found = np.mean(approximate[:, 0] == exact[:, 0].numpy())
        self.assertLess(nearest_found, 1.0)
        # Visiting every cell is the exact search.
        ivf.nprobe = 60
        _, every_cell = ivf.search(self.queries, 8)
        np.testing.assert_array_equal(every_cell, exact.numpy())

    def test_chunked_queries_give_the_same_result(self):
        whole = ExactIndex(self.vectors, "cpu", 8)
        pieces = ExactIndex(self.vectors, "cpu", 8)
        pieces.chunk = 7
        queries = torch.from_numpy(self.queries)
        np.testing.assert_array_equal(
            whole.search(queries)[1].numpy(), pieces.search(queries)[1].numpy()
        )

    def test_a_query_on_an_index_vector_returns_that_vector(self):
        # FAISS's weights are 1 / 0 here and the blend becomes NaN.
        result = ExactIndex(self.vectors, "cpu", 8).retrieve(
            torch.from_numpy(self.vectors[:5].copy())
        )
        self.assertTrue(torch.isfinite(result).all())
        np.testing.assert_allclose(result.numpy(), self.vectors[:5], atol=1e-3)

    def test_top_k_is_capped_by_the_index_size(self):
        self.assertEqual(ExactIndex(self.vectors[:3], "cpu", 8).top_k, 3)
        with self.assertRaises(ValueError):
            ExactIndex(np.zeros((0, DIM), dtype=np.float32), "cpu")

    def test_half_precision_queries_come_back_in_their_own_dtype(self):
        result = ExactIndex(self.vectors, "cpu", 4).retrieve(
            torch.from_numpy(self.queries).to(torch.bfloat16)
        )
        self.assertEqual(result.dtype, torch.bfloat16)


class OptionTests(unittest.TestCase):
    def test_the_defaults_are_the_legacy_search(self):
        options = RealtimeOptions()
        self.assertEqual(options.index_search, "faiss")
        self.assertEqual(options.index_top_k, 8)
        self.assertEqual(options.changed(), [])

    def test_values_are_validated(self):
        options = RealtimeOptions.from_dict(
            {"index_search": "exact", "index_top_k": 16.0}
        )
        self.assertEqual((options.index_search, options.index_top_k), ("exact", 16))
        self.assertIsInstance(options.index_top_k, int)
        fallback = RealtimeOptions.from_dict(
            {"index_search": "gpu", "index_top_k": "many"}
        )
        self.assertEqual((fallback.index_search, fallback.index_top_k), ("faiss", 8))
        self.assertEqual(RealtimeOptions.from_dict({"index_top_k": 0}).index_top_k, 1)
        self.assertEqual(RealtimeOptions.from_dict({"index_top_k": 999}).index_top_k, 64)


class PipelineTests(unittest.TestCase):
    """Realtime_Pipeline._retrieve_speaker_embeddings, on a stand-in for the pipeline."""

    def blend(self, options, index, vectors, feats, index_rate=0.75, skip_head=8):
        from rvc.realtime.pipeline import Realtime_Pipeline

        stand_in = types.SimpleNamespace(
            options=options, exact_index=None, device="cpu"
        )
        return Realtime_Pipeline._retrieve_speaker_embeddings(
            stand_in, skip_head, feats.clone(), index, vectors, index_rate
        )

    def setUp(self):
        self.vectors = clustered(3000, 30, seed=2)
        self.index = faiss.IndexFlatL2(DIM)
        self.index.add(self.vectors)
        self.feats = torch.from_numpy(clustered(50, 30, seed=3)).unsqueeze(0)

    def test_default_options_are_the_legacy_blend(self):
        result = self.blend(RealtimeOptions(), self.index, self.vectors, self.feats)
        expected = self.feats.clone()
        tail = expected[0][4:].numpy()
        expected[0][4:] = torch.from_numpy(
            legacy_blend(self.index, self.vectors, tail) * 0.75 + 0.25 * tail
        )
        self.assertTrue(torch.equal(result, expected))

    def test_exact_matches_faiss_when_faiss_is_exhaustive(self):
        faiss_result = self.blend(
            RealtimeOptions(), self.index, self.vectors, self.feats
        )
        exact_result = self.blend(
            RealtimeOptions(index_search="exact"), self.index, self.vectors, self.feats
        )
        # Frames ahead of skip_head // 2 are left alone by both.
        self.assertTrue(torch.equal(exact_result[0][:4], self.feats[0][:4]))
        np.testing.assert_allclose(
            exact_result.numpy(), faiss_result.numpy(), rtol=1e-3, atol=1e-3
        )

    def test_top_k_reaches_both_paths(self):
        for search in ("faiss", "exact"):
            one = self.blend(
                RealtimeOptions(index_search=search, index_top_k=1),
                self.index, self.vectors, self.feats, index_rate=1.0, skip_head=0,
            )
            _, nearest = self.index.search(self.feats[0].numpy(), 1)
            np.testing.assert_allclose(
                one[0].numpy(), self.vectors[nearest[:, 0]], atol=1e-4
            )

    def test_a_mismatched_index_is_still_skipped(self):
        wide = torch.zeros(1, 10, DIM + 1)
        result = self.blend(
            RealtimeOptions(index_search="exact"), self.index, self.vectors, wide
        )
        self.assertTrue(torch.equal(result, wide))


if __name__ == "__main__":
    unittest.main()
