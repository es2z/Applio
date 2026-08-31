import argparse
import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

import core
from rvc.lib import utils as embedder_utils


EMBEDDER_NAME = "japanese-hubert-base-k2"
EMBEDDER_REPO = "reazon-research/japanese-hubert-base-k2"
EMBEDDER_REVISION = "a9f26026165f8b80256f0aeecee53dedf81abce1"


def _i18n_label(node):
    if not isinstance(node, ast.Call) or not node.args:
        return None
    value = node.args[0]
    return value.value if isinstance(value, ast.Constant) else None


def _embedder_choice_lists(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    choice_lists = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "Radio":
            continue

        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        if _i18n_label(keywords.get("label")) != "Embedder Model":
            continue

        choices = keywords.get("choices")
        if not isinstance(choices, ast.List):
            continue
        choice_lists.append(
            [
                item.value
                for item in choices.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
        )
    return choice_lists


class EmbedderLoaderTest(unittest.TestCase):
    def test_k2_uses_transformers_safetensors_cache(self):
        model = SimpleNamespace()
        extractor = SimpleNamespace(do_normalize=True, sampling_rate=16000)
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            embedder_utils, "now_dir", temp_dir
        ), patch.object(
            embedder_utils.HubertModelWithFinalProj,
            "from_pretrained",
            return_value=model,
        ) as from_pretrained, patch.object(
            embedder_utils.Wav2Vec2FeatureExtractor,
            "from_pretrained",
            return_value=extractor,
        ) as feature_extractor:
            result = embedder_utils.load_embedding(EMBEDDER_NAME)

            expected_cache = os.path.join(
                temp_dir, "rvc", "models", "embedders", "japanese_hubert_base_k2"
            )
            self.assertIs(result, model)
            self.assertTrue(os.path.isdir(expected_cache))
            from_pretrained.assert_called_once_with(
                EMBEDDER_REPO,
                cache_dir=expected_cache,
                revision=EMBEDDER_REVISION,
                use_safetensors=True,
            )
            feature_extractor.assert_called_once_with(
                EMBEDDER_REPO, cache_dir=expected_cache, revision=EMBEDDER_REVISION
            )

    def test_the_pinned_revision_is_a_full_commit_sha(self):
        # A full 40-character SHA is what lets huggingface_hub serve a cached load
        # without a network round trip.
        self.assertEqual(embedder_utils.JAPANESE_HUBERT_BASE_K2_REVISION, EMBEDDER_REVISION)
        self.assertEqual(len(EMBEDDER_REVISION), 40)
        int(EMBEDDER_REVISION, 16)

    def test_k2_takes_do_normalize_from_the_official_config(self):
        model = SimpleNamespace()
        extractor = SimpleNamespace(do_normalize=True, sampling_rate=16000)
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            embedder_utils, "now_dir", temp_dir
        ), patch.object(
            embedder_utils.HubertModelWithFinalProj,
            "from_pretrained",
            return_value=model,
        ), patch.object(
            embedder_utils.Wav2Vec2FeatureExtractor,
            "from_pretrained",
            return_value=extractor,
        ):
            result = embedder_utils.load_embedding(EMBEDDER_NAME)

        self.assertTrue(result.input_do_normalize)
        self.assertEqual(result.input_sampling_rate, 16000)

    def test_k2_falls_back_to_the_documented_official_values(self):
        model = SimpleNamespace()
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            embedder_utils, "now_dir", temp_dir
        ), patch.object(
            embedder_utils.HubertModelWithFinalProj,
            "from_pretrained",
            return_value=model,
        ), patch.object(
            embedder_utils.Wav2Vec2FeatureExtractor,
            "from_pretrained",
            side_effect=OSError("offline"),
        ):
            result = embedder_utils.load_embedding(EMBEDDER_NAME)

        self.assertTrue(result.input_do_normalize)
        self.assertEqual(result.input_sampling_rate, 16000)

    def test_existing_japanese_hubert_keeps_legacy_local_loader(self):
        model = SimpleNamespace()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "rvc/models/embedders/japanese_hubert_base"
            model_dir.mkdir(parents=True)
            (model_dir / "pytorch_model.bin").write_bytes(b"")
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            with patch.object(
                embedder_utils, "now_dir", temp_dir
            ), patch.object(
                embedder_utils.HubertModelWithFinalProj,
                "from_pretrained",
                return_value=model,
            ) as from_pretrained, patch.object(
                embedder_utils.Wav2Vec2FeatureExtractor, "from_pretrained"
            ) as feature_extractor, patch.object(
                embedder_utils.wget, "download"
            ) as download:
                result = embedder_utils.load_embedding("japanese-hubert-base")

            self.assertIs(result, model)
            from_pretrained.assert_called_once_with(str(model_dir))
            feature_extractor.assert_not_called()
            download.assert_not_called()
            self.assertFalse(result.input_do_normalize)


class EmbedderInputNormalizationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.audio = (rng.standard_normal(16000 * 3) * 0.13 + 0.02).astype(np.float32)
        self.feats = torch.from_numpy(self.audio).view(1, -1)

    def test_matches_the_official_feature_extractor(self):
        # These values are the preprocessor_config.json of japanese-hubert-base-k2.
        extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )
        expected = extractor(self.audio, return_tensors="pt", sampling_rate=16000)[
            "input_values"
        ]

        model = SimpleNamespace(input_do_normalize=True)
        actual = embedder_utils.apply_embedder_input_normalization(model, self.feats)

        self.assertEqual(actual.shape, expected.shape)
        self.assertLess((actual - expected).abs().max().item(), 1e-5)

    def test_is_a_noop_for_embedders_without_the_flag(self):
        for model in (SimpleNamespace(input_do_normalize=False), SimpleNamespace()):
            with self.subTest(model=model):
                actual = embedder_utils.apply_embedder_input_normalization(
                    model, self.feats
                )
                self.assertIs(actual, self.feats)


class EmbedderInterfaceTest(unittest.TestCase):
    def test_cli_modes_accept_k2(self):
        with patch.object(
            argparse.ArgumentParser,
            "parse_args",
            autospec=True,
            side_effect=lambda parser, *args, **kwargs: parser,
        ):
            parser = core.parse_arguments()

        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        for mode in ("infer", "batch_infer", "tts", "extract"):
            with self.subTest(mode=mode):
                action = next(
                    item
                    for item in subparsers.choices[mode]._actions
                    if item.dest == "embedder_model"
                )
                self.assertIn(EMBEDDER_NAME, action.choices)

    def test_all_embedder_radios_include_k2(self):
        root = Path(__file__).resolve().parents[1]
        expected_counts = {
            "tabs/train/train.py": 1,
            "tabs/inference/inference.py": 2,
            "tabs/realtime/realtime.py": 1,
            "tabs/tts/tts.py": 1,
        }
        for relative_path, expected_count in expected_counts.items():
            with self.subTest(path=relative_path):
                choice_lists = _embedder_choice_lists(root / relative_path)
                self.assertEqual(len(choice_lists), expected_count)
                for choices in choice_lists:
                    self.assertIn(EMBEDDER_NAME, choices)


if __name__ == "__main__":
    unittest.main()
