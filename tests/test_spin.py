"""SPIN fine-tuning (tools/spin/): on a tiny random embedder shaped like kushinada.

The real model is 1.2 GB and hand installed, so these build a 4-layer HubertModel with the
same architecture flags (feat_extract_norm "layer", conv_bias, do_stable_layer_norm) and
check the claims the training relies on.
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch
from transformers import HubertConfig

from unittest.mock import patch

from rvc.lib import utils as embedder_utils
from rvc.lib.utils import HubertModelWithFinalProj, _attach_input_preprocessing, embedder_forward
from tools.spin.data import SpinBatches, build_cache
from tools.spin.export import load_spin_checkpoint
from tools.spin.model import SpinHead, compute_sinkhorn, load_truncated, set_trainable, warmup_decay
from tools.spin.perturb import perturb_speaker
from tools.spin.train import trained_state


def _tiny_embedder(folder):
    config = HubertConfig(
        hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64,
        conv_dim=(16,) * 7, conv_stride=(5, 2, 2, 2, 2, 2, 2), conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=True, feat_extract_norm="layer", do_stable_layer_norm=True,
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4, classifier_proj_size=8,
        hidden_dropout=0.1, attention_dropout=0.1, layerdrop=0.1,
    )
    torch.manual_seed(0)
    HubertModelWithFinalProj(config).save_pretrained(folder)
    with open(os.path.join(folder, "preprocessor_config.json"), "w") as f:
        json.dump({"feature_extractor_type": "Wav2Vec2FeatureExtractor", "do_normalize": True,
                   "feature_size": 1, "padding_side": "right", "padding_value": 0.0,
                   "return_attention_mask": True, "sampling_rate": 16000}, f)


def _voice(seconds, sr=16000, f0=150.0):
    t = np.arange(int(seconds * sr)) / sr
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))) / sr
    wave = sum(np.sin(k * phase) / k for k in range(1, 12))
    return (0.1 * wave).astype(np.float32)


class SpinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.model_dir = os.path.join(cls.tmp, "embedder")
        _tiny_embedder(cls.model_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cut_model_is_the_full_model_at_that_layer(self):
        full = HubertModelWithFinalProj.from_pretrained(self.model_dir)
        _attach_input_preprocessing(full, self.model_dir, None, default_do_normalize=True)
        full.output_layer = 3
        full.eval()
        cut = load_truncated(self.model_dir, 3).eval()
        self.assertEqual(len(cut.encoder.layers), 3)
        self.assertTrue(cut.input_do_normalize)
        x = torch.from_numpy(_voice(1.0)).unsqueeze(0)
        with torch.no_grad():
            torch.testing.assert_close(embedder_forward(cut, x), embedder_forward(full, x),
                                       rtol=1e-5, atol=1e-5)

    def test_only_the_top_layers_learn(self):
        model = load_truncated(self.model_dir, 3)
        params = set_trainable(model, 2)
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.encoder.layers[0].training)
        self.assertTrue(model.encoder.layers[1].training and model.encoder.layers[2].training)
        x = torch.from_numpy(_voice(1.0)).unsqueeze(0)
        embedder_forward(model, x).square().mean().backward()
        for name, p in model.named_parameters():
            top = name.startswith(("encoder.layers.1.", "encoder.layers.2."))
            self.assertEqual(p.grad is not None, top, name)
        self.assertEqual(sum(p.numel() for p in params),
                         sum(p.numel() for n, p in model.named_parameters()
                             if n.startswith(("encoder.layers.1.", "encoder.layers.2."))))

    def test_sinkhorn_gives_each_frame_a_distribution(self):
        # The scores are cosines (both sides L2-normalised), so they lie in [-1, 1].
        q = compute_sinkhorn(torch.rand(50, 8) * 2 - 1, 0.02, 3)
        self.assertTrue(bool((q >= 0).all()))
        torch.testing.assert_close(q.sum(1), torch.ones(50), rtol=1e-4, atol=1e-4)

    def test_head_loss_trains_the_projection_and_codebook(self):
        head = SpinHead(dim=32, proj_dim=16, num_vars=8)
        loss, stats = head(torch.randn(40, 32), torch.randn(40, 32))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(head.pred.weight.grad)
        self.assertIsNotNone(head.codebook.weight.grad)
        self.assertTrue(0 <= stats["acc"] <= 1)

    def test_perturbation_keeps_the_length(self):
        rng = np.random.default_rng(0)
        wav = _voice(1.5)
        out, failed = perturb_speaker(wav, 16000, 75, 400, rng)
        self.assertFalse(failed)
        self.assertEqual(out.shape, wav.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(np.abs(out - wav).mean(), 1e-3)
        silent, failed = perturb_speaker(np.zeros(16000, np.float32), 16000, 75, 400, rng)
        self.assertTrue(failed)
        self.assertEqual(silent.shape, (16000,))

    def test_schedule_is_spins(self):
        self.assertAlmostEqual(warmup_decay(0), 0.01)
        self.assertAlmostEqual(warmup_decay(2500), 1.0)
        self.assertAlmostEqual(warmup_decay(4999), 0.01, places=6)

    def test_batches_are_equal_length_pairs_and_reproducible(self):
        dataset = os.path.join(self.tmp, "dataset")
        for speaker, f0 in (("0", 120.0), ("1", 240.0)):
            os.makedirs(os.path.join(dataset, speaker))
            for i, seconds in enumerate((2.6, 3.0, 4.0, 1.0)):
                wav = _voice(seconds, 48000, f0)
                sf.write(os.path.join(dataset, speaker, f"{i}.wav"), wav, 48000, subtype="PCM_24")
        cache = os.path.join(self.tmp, "cache")
        build_cache(dataset, cache)
        a = SpinBatches(cache, steps=3, batch_seconds=6.0)
        b = SpinBatches(cache, steps=3, batch_seconds=6.0)
        self.assertEqual(a.dropped, 2)  # the 1 s files
        originals, perturbed, _ = a[0]
        self.assertEqual(originals.shape, perturbed.shape)
        self.assertGreaterEqual(originals.shape[0] * originals.shape[1], 6 * 16000)
        torch.testing.assert_close(originals, b[0][0])

    def test_crop_sampler_draws_seconds_and_honours_speaker_weights(self):
        dataset = os.path.join(self.tmp, "crop_dataset")
        # Speaker 0: two long files. Speaker 1: many short ones, a little more in total.
        for speaker, lengths in (("0", (12.0,) * 2), ("1", (2.0,) * 16)):
            os.makedirs(os.path.join(dataset, speaker))
            for i, seconds in enumerate(lengths):
                wav = _voice(seconds, 16000, 150.0)
                sf.write(os.path.join(dataset, speaker, f"{i}.wav"), wav, 16000)
        cache = os.path.join(self.tmp, "crop_cache")
        build_cache(dataset, cache)

        def share(weights, sampler="crop"):
            data = SpinBatches(cache, steps=400, batch_seconds=4.0, min_seconds=1.5,
                               sampler=sampler, speaker_weights=weights,
                               short_max_seconds=1.9)
            seconds = {"0": 0, "1": 0}
            for k in range(400):
                chosen, length = data.batch_members(k)
                self.assertTrue(all(data.lengths[i] >= length for i in chosen))
                for i in chosen:
                    seconds[data.items[i]["speaker"]] += length
            return seconds["0"] / (seconds["0"] + seconds["1"])

        # Drawn by file, the long files get a ninth of the draws; drawn by seconds they are
        # 43% of the audio, short crops are cut out of them too, and only they are long
        # enough for the long crops.
        self.assertLess(share({}, "length"), 0.3)
        self.assertGreater(share({}), 0.4)
        self.assertGreater(share({"0": 3.0}), share({}) + 0.15)
        data = SpinBatches(cache, steps=3, batch_seconds=8.0, min_seconds=1.5, sampler="crop")
        torch.testing.assert_close(data[1][0], SpinBatches(
            cache, steps=3, batch_seconds=8.0, min_seconds=1.5, sampler="crop")[1][0])

    def test_export_round_trip_carries_the_trained_layers(self):
        model = load_truncated(self.model_dir, 3)
        with torch.no_grad():
            model.encoder.layers[2].feed_forward.output_dense.weight.add_(1.0)
        ckpt = os.path.join(self.tmp, "ckpt.pt")
        torch.save({"step": 1, "layers": 3, "train_layers": 2, "model": self.model_dir,
                    "state": trained_state(model, 2)}, ckpt)
        loaded, _ = load_spin_checkpoint(ckpt)
        torch.testing.assert_close(loaded.encoder.layers[2].feed_forward.output_dense.weight,
                                   model.encoder.layers[2].feed_forward.output_dense.weight)
        torch.testing.assert_close(loaded.encoder.layers[0].feed_forward.output_dense.weight,
                                   model.encoder.layers[0].feed_forward.output_dense.weight)


class SpinEmbedderRegistryTest(unittest.TestCase):
    NAME = "kushinada-hubert-large-spin"

    def test_it_is_a_local_embedder_with_its_own_install_hint(self):
        spec = embedder_utils.EMBEDDERS[self.NAME]
        self.assertTrue(spec["local"])
        self.assertNotIn("repo", spec)
        self.assertNotIn(self.NAME, embedder_utils.EMBEDDER_FEATURE_SCALE)
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir, True)
        with patch.object(embedder_utils, "now_dir", temp_dir):
            with self.assertRaises(FileNotFoundError) as raised:
                embedder_utils.load_embedding(self.NAME)
        message = str(raised.exception)
        self.assertIn("tools/spin/export.py", message)
        self.assertIn(spec["dir"], message)
        self.assertNotIn("license", message)

    def test_every_embedder_list_offers_it(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for path in ("core.py", "tabs/train/train.py", "tabs/inference/inference.py",
                     "tabs/realtime/realtime.py", "tabs/tts/tts.py"):
            with open(os.path.join(root, path), encoding="utf-8") as f:
                text = f.read()
            self.assertEqual(text.count('"kushinada-hubert-large",'),
                             text.count(f'"{self.NAME}",'), path)


class SpinV2EmbedderRegistryTest(SpinEmbedderRegistryTest):
    NAME = "kushinada-hubert-large-spin-v2"


class SpinV21kEmbedderRegistryTest(SpinEmbedderRegistryTest):
    NAME = "kushinada-hubert-large-spin-v2-1k"


class SpinV22kEmbedderRegistryTest(SpinEmbedderRegistryTest):
    NAME = "kushinada-hubert-large-spin-v2-2k"


if __name__ == "__main__":
    unittest.main()
