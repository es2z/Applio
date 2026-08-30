import argparse
import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core
from rvc.lib import utils as embedder_utils


EMBEDDER_NAME = "japanese-hubert-base-phoneme-ctc-v4"
EMBEDDER_REPO = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4"


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
    def test_phoneme_ctc_v4_uses_transformers_safetensors_cache(self):
        sentinel = object()
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            embedder_utils, "now_dir", temp_dir
        ), patch.object(
            embedder_utils.HubertModelWithFinalProj,
            "from_pretrained",
            return_value=sentinel,
        ) as from_pretrained:
            result = embedder_utils.load_embedding(EMBEDDER_NAME)

            expected_cache = os.path.join(
                temp_dir,
                "rvc",
                "models",
                "embedders",
                "japanese_hubert_base_phoneme_ctc_v4",
            )
            self.assertIs(result, sentinel)
            self.assertTrue(os.path.isdir(expected_cache))
            from_pretrained.assert_called_once_with(
                EMBEDDER_REPO,
                cache_dir=expected_cache,
                use_safetensors=True,
            )

    def test_existing_japanese_hubert_keeps_legacy_local_loader(self):
        sentinel = object()
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
                return_value=sentinel,
            ) as from_pretrained, patch.object(
                embedder_utils.wget, "download"
            ) as download:
                result = embedder_utils.load_embedding("japanese-hubert-base")

            self.assertIs(result, sentinel)
            from_pretrained.assert_called_once_with(str(model_dir))
            download.assert_not_called()


class EmbedderInterfaceTest(unittest.TestCase):
    def test_cli_modes_accept_phoneme_ctc_v4(self):
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

    def test_all_embedder_radios_include_phoneme_ctc_v4(self):
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
