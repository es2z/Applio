import argparse
import ast
import json
import os
import shutil
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
from rvc.train.extract.extract import resolve_feature_reuse
from rvc.train.extract.preparing_files import (
    apply_train_settings,
    generate_config,
    read_train_settings,
)
from rvc.train.utils import assert_resumable, load_pretrained


EMBEDDER_NAME = "japanese-hubert-base-k2"
EMBEDDER_REPO = "reazon-research/japanese-hubert-base-k2"
EMBEDDER_REVISION = "a9f26026165f8b80256f0aeecee53dedf81abce1"
EMBEDDER_FEATURE_SCALE = 10.0

# Per-frame norm of last_hidden_state, measured on the same speech with each embedder.
# k2 sits an order of magnitude below the rest because its final LayerNorm gain is that
# much smaller, and RVC v2 adds emb_phone(feature) straight onto a scale free emb_pitch.
K2_FEATURE_NORM = 0.644
JAPANESE_HUBERT_BASE_FEATURE_NORM = 6.494
CONTENTVEC_FEATURE_NORM = 9.31


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

    def test_k2_carries_the_feature_scale(self):
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

        self.assertEqual(result.feature_scale, EMBEDDER_FEATURE_SCALE)

    def test_custom_embedders_keep_a_neutral_feature_scale(self):
        model = SimpleNamespace()
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            embedder_utils.HubertModelWithFinalProj,
            "from_pretrained",
            return_value=model,
        ):
            result = embedder_utils.load_embedding("custom", temp_dir)

        self.assertEqual(result.feature_scale, 1.0)

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
            self.assertEqual(result.feature_scale, 1.0)


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


class EmbedderFeatureScaleTest(unittest.TestCase):
    def setUp(self):
        self.feats = torch.arange(12, dtype=torch.float32).view(1, 4, 3)

    def test_is_a_noop_for_embedders_without_a_scale(self):
        for model in (SimpleNamespace(feature_scale=1.0), SimpleNamespace()):
            with self.subTest(model=model):
                actual = embedder_utils.apply_embedder_feature_scale(model, self.feats)
                self.assertIs(actual, self.feats)

    def test_multiplies_the_hidden_states(self):
        model = SimpleNamespace(feature_scale=EMBEDDER_FEATURE_SCALE)
        actual = embedder_utils.apply_embedder_feature_scale(model, self.feats)
        self.assertTrue(torch.equal(actual, self.feats * EMBEDDER_FEATURE_SCALE))

    def test_the_constant_lands_k2_beside_the_other_embedders(self):
        scaled = K2_FEATURE_NORM * embedder_utils.EMBEDDER_FEATURE_SCALE[EMBEDDER_NAME]
        self.assertGreaterEqual(scaled, JAPANESE_HUBERT_BASE_FEATURE_NORM * 0.9)
        self.assertLessEqual(scaled, CONTENTVEC_FEATURE_NORM)

    def test_only_k2_is_scaled(self):
        self.assertEqual(
            set(embedder_utils.EMBEDDER_FEATURE_SCALE), {EMBEDDER_NAME}
        )


class CheckpointFeatureScaleWarningTest(unittest.TestCase):
    def _warn(self, checkpoint, feature_scale):
        model = SimpleNamespace(feature_scale=feature_scale)
        with patch("builtins.print") as printed:
            embedder_utils.warn_on_feature_scale_mismatch(model, checkpoint)
        return [call.args[0] for call in printed.call_args_list]

    def test_a_checkpoint_trained_before_scaling_is_flagged(self):
        messages = self._warn({"embedder_model": EMBEDDER_NAME}, EMBEDDER_FEATURE_SCALE)
        self.assertEqual(len(messages), 1)
        self.assertIn("1.0", messages[0])
        self.assertIn("10.0", messages[0])

    def test_a_matching_checkpoint_is_quiet(self):
        checkpoint = {
            "embedder_model": EMBEDDER_NAME,
            "embedder_feature_scale": EMBEDDER_FEATURE_SCALE,
        }
        self.assertEqual(self._warn(checkpoint, EMBEDDER_FEATURE_SCALE), [])

    def test_existing_unscaled_checkpoints_are_quiet(self):
        self.assertEqual(self._warn({"embedder_model": "contentvec"}, 1.0), [])

    def test_a_checkpoint_without_metadata_is_quiet(self):
        for checkpoint in ({}, None, {"embedder_model": None}):
            with self.subTest(checkpoint=checkpoint):
                self.assertEqual(self._warn(checkpoint, EMBEDDER_FEATURE_SCALE), [])


class FeatureReuseGuardTest(unittest.TestCase):
    def test_an_unrecorded_folder_is_left_alone(self):
        rebuild, reason = resolve_feature_reuse({}, EMBEDDER_NAME, EMBEDDER_FEATURE_SCALE)
        self.assertFalse(rebuild)
        self.assertIsNone(reason)

    def test_a_matching_embedder_and_scale_is_reused(self):
        data = {
            "embedder_model": EMBEDDER_NAME,
            "embedder_feature_scale": EMBEDDER_FEATURE_SCALE,
        }
        rebuild, _ = resolve_feature_reuse(data, EMBEDDER_NAME, EMBEDDER_FEATURE_SCALE)
        self.assertFalse(rebuild)

    def test_a_different_embedder_forces_a_rebuild(self):
        data = {"embedder_model": "japanese-hubert-base"}
        rebuild, reason = resolve_feature_reuse(data, EMBEDDER_NAME, EMBEDDER_FEATURE_SCALE)
        self.assertTrue(rebuild)
        self.assertIn("japanese-hubert-base", reason)

    def test_features_extracted_before_scaling_force_a_rebuild(self):
        # No recorded scale means the folder predates scaling, which is a scale of 1.0.
        data = {"embedder_model": EMBEDDER_NAME}
        rebuild, reason = resolve_feature_reuse(data, EMBEDDER_NAME, EMBEDDER_FEATURE_SCALE)
        self.assertTrue(rebuild)
        self.assertIn("1.0", reason)

    def test_an_unscaled_embedder_without_a_record_is_reused(self):
        data = {"embedder_model": "japanese-hubert-base"}
        rebuild, _ = resolve_feature_reuse(data, "japanese-hubert-base", 1.0)
        self.assertFalse(rebuild)


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
                self.assertIn(LARGE_NAME, action.choices)

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
                    self.assertIn(LARGE_NAME, choices)


LARGE_NAME = "japanese-hubert-large"
LARGE_REPO = "yky-h/japanese-hubert-large"
LARGE_REVISION = "bccd07ba8a9f025576d53ca84669c540c7ef204e"

# Measured on logs/reference/reference.wav, per-frame L2 norm of last_hidden_state.
LARGE_FEATURE_NORM = 5.949


class _FakeConfig:
    def __init__(self, hidden_size=1024, num_hidden_layers=24, do_stable_layer_norm=True):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.do_stable_layer_norm = do_stable_layer_norm


class _FakeEncoder(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # A deliberately non-identity norm, so a test can tell whether it was applied.
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        with torch.no_grad():
            self.layer_norm.weight.fill_(3.0)
            self.layer_norm.bias.fill_(0.5)


class _FakeEmbedder(torch.nn.Module):
    """A stand-in that records how it was called and returns identifiable layers."""

    def __init__(self, hidden_size=1024, num_hidden_layers=24, do_stable_layer_norm=True):
        super().__init__()
        self.config = _FakeConfig(hidden_size, num_hidden_layers, do_stable_layer_norm)
        self.encoder = _FakeEncoder(hidden_size)
        self.input_do_normalize = False
        self.feature_scale = 1.0
        self.output_layer = None
        self.embed_dim = hidden_size
        self.seen_dtype = None

    def forward(self, feats, output_hidden_states=False):
        self.seen_dtype = feats.dtype
        frames, width = 4, self.config.hidden_size
        last = torch.full((1, frames, width), 99.0)
        if not output_hidden_states:
            return {"last_hidden_state": last}
        hidden = [
            torch.full((1, frames, width), float(i))
            for i in range(self.config.num_hidden_layers)
        ]
        return {"last_hidden_state": last, "hidden_states": hidden + [last]}


class LargeEmbedderRegistryTest(unittest.TestCase):
    def test_the_registry_pins_a_full_commit_sha(self):
        spec = embedder_utils.EMBEDDERS[LARGE_NAME]
        self.assertEqual(spec["repo"], LARGE_REPO)
        self.assertEqual(spec["revision"], LARGE_REVISION)
        self.assertEqual(len(spec["revision"]), 40)
        int(spec["revision"], 16)

    def test_large_carries_no_feature_scale(self):
        # Measured at 5.95 per frame against japanese-hubert-base's 6.35, so unlike k2 it
        # needs no correction to sit where RVC v2 expects its features.
        self.assertNotIn(LARGE_NAME, embedder_utils.EMBEDDER_FEATURE_SCALE)
        self.assertAlmostEqual(
            LARGE_FEATURE_NORM / JAPANESE_HUBERT_BASE_FEATURE_NORM, 1.0, delta=0.15
        )

    def test_large_loads_through_transformers_with_the_official_preprocessing(self):
        model = SimpleNamespace()
        extractor = SimpleNamespace(do_normalize=True, sampling_rate=16000)
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            embedder_utils, "now_dir", temp_dir
        ), patch.object(
            embedder_utils.HubertModelWithFinalProj, "from_pretrained", return_value=model
        ) as from_pretrained, patch.object(
            embedder_utils.Wav2Vec2FeatureExtractor,
            "from_pretrained",
            return_value=extractor,
        ):
            result = embedder_utils.load_embedding(LARGE_NAME)

            expected_cache = os.path.join(
                temp_dir, "rvc", "models", "embedders", "japanese_hubert_large"
            )
            from_pretrained.assert_called_once_with(
                LARGE_REPO,
                cache_dir=expected_cache,
                revision=LARGE_REVISION,
                use_safetensors=True,
            )
        # feat_extract_norm="layer" with conv_bias=True means nothing downstream cancels
        # the input gain, so this flag is load bearing rather than cosmetic here.
        self.assertTrue(result.input_do_normalize)
        self.assertEqual(result.feature_scale, 1.0)

    def test_an_out_of_range_output_layer_falls_back_to_the_last(self):
        model = _FakeEmbedder(num_hidden_layers=24)
        embedder_utils._finalize_embedder(model, {}, output_layer=99)
        self.assertIsNone(model.output_layer)

    def test_loading_leaves_the_embedder_in_eval_mode(self):
        # layerdrop 0.1 over 24 layers would silently drop a couple of them per call.
        model = _FakeEmbedder()
        model.train()
        embedder_utils._finalize_embedder(model, {})
        self.assertFalse(model.training)


class EmbedderInputStdFloorTest(unittest.TestCase):
    def _normalise(self, rms, floor):
        model = SimpleNamespace(input_do_normalize=True)
        noise = torch.randn(1, 16000, generator=torch.Generator().manual_seed(0)) * rms
        with patch.object(embedder_utils, "EMBEDDER_INPUT_STD_FLOOR", floor):
            return float(embedder_utils.apply_embedder_input_normalization(model, noise).std())

    def test_speech_level_audio_still_reaches_unit_variance(self):
        self.assertAlmostEqual(self._normalise(0.15, 0.01), 1.0, places=3)

    def test_room_tone_is_not_lifted_to_speech_level(self):
        # Without the floor, -60 dBFS room tone comes out at 0.95 RMS, a gain of about
        # 60 dB, and the embedder reads the amplified noise as speech.
        self.assertGreater(self._normalise(0.001, 0.0), 0.9)
        self.assertLess(self._normalise(0.001, 0.01), 0.15)

    def test_a_zero_floor_is_the_literal_official_behaviour(self):
        model = SimpleNamespace(input_do_normalize=True)
        audio = torch.randn(1, 4000, generator=torch.Generator().manual_seed(1)) * 0.02
        expected = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True
        ).zero_mean_unit_var_norm([audio[0].numpy()], attention_mask=None)[0]
        with patch.object(embedder_utils, "EMBEDDER_INPUT_STD_FLOOR", 0.0):
            got = embedder_utils.apply_embedder_input_normalization(model, audio)
        np.testing.assert_allclose(got[0].numpy(), expected, rtol=1e-5, atol=1e-6)


class EmbedderForwardTest(unittest.TestCase):
    def test_the_last_layer_uses_last_hidden_state(self):
        model = _FakeEmbedder()
        out = embedder_utils.embedder_forward(model, torch.zeros(1, 320))
        self.assertTrue(torch.equal(out, torch.full((1, 4, 1024), 99.0)))

    def test_an_intermediate_layer_is_renormalised(self):
        # Raw pre-norm residuals run from 69 per frame at layer 0 to 538 at layer 23 on
        # the real model, so handing one straight to emb_phone would be off by orders of
        # magnitude against the last layer it was trained beside.
        model = _FakeEmbedder()
        model.output_layer = 18
        out = embedder_utils.embedder_forward(model, torch.zeros(1, 320))
        expected = model.encoder.layer_norm(torch.full((1, 4, 1024), 18.0))
        self.assertTrue(torch.allclose(out, expected))

    def test_a_post_norm_embedder_takes_the_layer_untouched(self):
        model = _FakeEmbedder(hidden_size=768, num_hidden_layers=12, do_stable_layer_norm=False)
        model.output_layer = 9
        out = embedder_utils.embedder_forward(model, torch.zeros(1, 320))
        self.assertTrue(torch.equal(out, torch.full((1, 4, 768), 9.0)))

    def test_the_feature_scale_is_applied_after_the_layer_choice(self):
        model = _FakeEmbedder()
        model.feature_scale = 10.0
        out = embedder_utils.embedder_forward(model, torch.zeros(1, 320))
        self.assertTrue(torch.equal(out, torch.full((1, 4, 1024), 990.0)))

    def test_the_input_is_cast_to_the_embedder_dtype(self):
        model = _FakeEmbedder().to(torch.bfloat16)
        embedder_utils.embedder_forward(model, torch.zeros(1, 320, dtype=torch.float32))
        self.assertEqual(model.seen_dtype, torch.bfloat16)


class CheckpointFeatureWidthTest(unittest.TestCase):
    def test_the_width_comes_from_the_projection_weights(self):
        for width in (256, 768, 1024):
            checkpoint = {"weight": {"enc_p.emb_phone.weight": torch.zeros(192, width)}}
            self.assertEqual(
                embedder_utils.checkpoint_text_enc_hidden_dim(checkpoint), width
            )

    def test_the_weights_win_over_a_stale_recorded_value(self):
        checkpoint = {
            "weight": {"enc_p.emb_phone.weight": torch.zeros(192, 1024)},
            "text_enc_hidden_dim": 768,
        }
        self.assertEqual(embedder_utils.checkpoint_text_enc_hidden_dim(checkpoint), 1024)

    def test_a_checkpoint_without_the_projection_falls_back_to_the_version(self):
        self.assertEqual(
            embedder_utils.checkpoint_text_enc_hidden_dim({"weight": {}, "version": "v2"}),
            768,
        )
        self.assertEqual(
            embedder_utils.checkpoint_text_enc_hidden_dim({"weight": {}, "version": "v1"}),
            256,
        )


class EmbedderMismatchTest(unittest.TestCase):
    def test_an_unrecorded_identity_cannot_be_compared(self):
        self.assertIsNone(
            embedder_utils.describe_embedder_mismatch({}, {"embedder_model": LARGE_NAME})
        )

    def test_a_changed_output_layer_is_reported_by_name(self):
        reason = embedder_utils.describe_embedder_mismatch(
            {"embedder_model": LARGE_NAME, "embedder_output_layer": None},
            {"embedder_model": LARGE_NAME, "embedder_output_layer": 18},
        )
        self.assertIn("output layer last -> 18", reason)

    def test_a_changed_width_is_reported(self):
        reason = embedder_utils.describe_embedder_mismatch(
            {"embedder_model": LARGE_NAME, "embedder_dim": 768},
            {"embedder_model": LARGE_NAME, "embedder_dim": 1024},
        )
        self.assertIn("feature dimension 768 -> 1024", reason)

    def test_an_untracked_width_is_not_a_mismatch(self):
        self.assertIsNone(
            embedder_utils.describe_embedder_mismatch(
                {"embedder_model": LARGE_NAME},
                {"embedder_model": LARGE_NAME, "embedder_dim": 1024},
            )
        )


    def test_a_changed_input_std_floor_is_reported(self):
        # It changes the stored features for any do_normalize embedder by over 50% on a
        # quiet window, so it has to invalidate them like the scale does.
        reason = embedder_utils.describe_embedder_mismatch(
            {"embedder_model": LARGE_NAME, "embedder_input_std_floor": 0.01},
            {"embedder_model": LARGE_NAME, "embedder_input_std_floor": 0.0},
        )
        self.assertIn("input std floor 0.01 -> 0.0", reason)

    def test_an_embedder_that_does_not_normalise_records_no_floor(self):
        model = SimpleNamespace(input_do_normalize=False)
        self.assertIsNone(
            embedder_utils.embedder_identity(model)["embedder_input_std_floor"]
        )

    def test_a_normalising_embedder_records_the_floor(self):
        model = SimpleNamespace(input_do_normalize=True)
        self.assertEqual(
            embedder_utils.embedder_identity(model)["embedder_input_std_floor"],
            embedder_utils.EMBEDDER_INPUT_STD_FLOOR,
        )


class GenerateConfigTest(unittest.TestCase):
    def _run(self, temp_dir, dim, seed=None):
        model_dir = Path(temp_dir) / "logs" / "run"
        model_dir.mkdir(parents=True, exist_ok=True)
        if seed is not None:
            (model_dir / "config.json").write_text(json.dumps(seed), encoding="utf-8")
        cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            (Path(temp_dir) / "rvc" / "configs").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                Path(cwd) / "rvc" / "configs" / "40000.json",
                Path(temp_dir) / "rvc" / "configs" / "40000.json",
            )
            generate_config(40000, str(model_dir), dim)
        finally:
            os.chdir(cwd)
        return json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    def test_a_fresh_run_gets_the_measured_width(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._run(temp_dir, 1024)
        self.assertEqual(config["model"]["text_enc_hidden_dim"], 1024)

    def test_hand_edited_hyperparameters_survive_a_re_extract(self):
        # logs/naru_20260831_1/config.json really does carry hand-tuned values, so only
        # the one key that has to follow the features may be rewritten.
        seed = {
            "train": {"learning_rate": 7e-05, "c_mel": 50},
            "model": {"text_enc_hidden_dim": 768, "gin_channels": 256},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._run(temp_dir, 1024, seed)
        self.assertEqual(config["train"]["learning_rate"], 7e-05)
        self.assertEqual(config["train"]["c_mel"], 50)
        self.assertEqual(config["model"]["gin_channels"], 256)
        self.assertEqual(config["model"]["text_enc_hidden_dim"], 1024)

    def test_no_measured_width_leaves_the_config_alone(self):
        seed = {"model": {"text_enc_hidden_dim": 768}}
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._run(temp_dir, None, seed)
        self.assertEqual(config["model"]["text_enc_hidden_dim"], 768)


class _TinyNet(torch.nn.Module):
    def __init__(self, feature_dim=768, decoder_out=32):
        super().__init__()
        self.enc_p = torch.nn.Module()
        self.enc_p.emb_phone = torch.nn.Linear(feature_dim, 192)
        self.dec = torch.nn.Linear(192, decoder_out)


class LoadPretrainedTest(unittest.TestCase):
    def _checkpoint(self, temp_dir, net):
        path = os.path.join(temp_dir, "pretrain.pth")
        torch.save({"model": net.state_dict()}, path)
        return path

    def test_only_the_embedder_projection_may_change_shape(self):
        pretrain = _TinyNet(feature_dim=768)
        target = _TinyNet(feature_dim=1024)
        before = target.enc_p.emb_phone.weight.detach().clone()
        with tempfile.TemporaryDirectory() as temp_dir:
            load_pretrained(target, self._checkpoint(temp_dir, pretrain), "G", verbose=False)
        # The decoder is inherited, which is the whole point of warm starting.
        self.assertTrue(torch.equal(target.dec.weight, pretrain.dec.weight))
        # The projection is left as it was initialised, not silently half-loaded.
        self.assertTrue(torch.equal(target.enc_p.emb_phone.weight, before))

    def test_a_matching_pretrain_is_loaded_whole(self):
        pretrain = _TinyNet(feature_dim=768)
        target = _TinyNet(feature_dim=768)
        with tempfile.TemporaryDirectory() as temp_dir:
            load_pretrained(target, self._checkpoint(temp_dir, pretrain), "G", verbose=False)
        self.assertTrue(
            torch.equal(target.enc_p.emb_phone.weight, pretrain.enc_p.emb_phone.weight)
        )

    def test_any_other_shape_mismatch_still_stops_the_run(self):
        # A different sample rate or vocoder shows up here, and must not be warm started.
        pretrain = _TinyNet(decoder_out=32)
        target = _TinyNet(decoder_out=64)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._checkpoint(temp_dir, pretrain)
            with self.assertRaises(SystemExit):
                load_pretrained(target, path, "G", verbose=False)


class ResumeGuardTest(unittest.TestCase):
    def _experiment(self, temp_dir, recorded):
        checkpoint = {"model": {}, "iteration": 1}
        checkpoint.update(recorded)
        torch.save(checkpoint, os.path.join(temp_dir, "G_100.pth"))
        return temp_dir

    def test_resuming_onto_different_features_is_refused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._experiment(temp_dir, {"embedder_model": "japanese-hubert-base"})
            with self.assertRaises(SystemExit):
                assert_resumable(temp_dir, {"embedder_model": LARGE_NAME})

    def test_a_changed_feature_scale_alone_is_refused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._experiment(
                temp_dir,
                {"embedder_model": EMBEDDER_NAME, "embedder_feature_scale": 1.0},
            )
            with self.assertRaises(SystemExit):
                assert_resumable(
                    temp_dir,
                    {"embedder_model": EMBEDDER_NAME, "embedder_feature_scale": 10.0},
                )

    def test_a_matching_checkpoint_resumes(self):
        identity = {
            "embedder_model": LARGE_NAME,
            "embedder_feature_scale": 1.0,
            "embedder_output_layer": None,
            "embedder_dim": 1024,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            self._experiment(temp_dir, identity)
            assert_resumable(temp_dir, identity)

    def test_a_checkpoint_from_before_this_was_recorded_still_resumes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._experiment(temp_dir, {})
            assert_resumable(temp_dir, {"embedder_model": LARGE_NAME})

    def test_a_fresh_folder_resumes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            assert_resumable(temp_dir, {"embedder_model": LARGE_NAME})



class RealtimeWiringTest(unittest.TestCase):
    """The realtime constructors are chained by hand, so guard how they are called.

    Realtime, VoiceChanger and AudioCallbacks each forward a long list of settings to the
    next one down. While those calls were positional, inserting embedder_precision into a
    signature silently shifted every argument after it: vad_frame_ms received sid's 0 and
    the run died with "VAD frame duration must be 10, 20, or 30 ms". Keyword arguments
    make that impossible, so require them.
    """

    CHAINED_CALLS = {"create_pipeline", "Realtime", "VoiceChanger"}

    def test_the_realtime_chain_is_wired_by_keyword(self):
        root = Path(__file__).resolve().parents[1] / "rvc" / "realtime"
        checked = set()
        for path in sorted(root.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                name = node.func.id
                if name not in self.CHAINED_CALLS:
                    continue
                checked.add(name)
                with self.subTest(call=name, file=path.name):
                    self.assertEqual(
                        [ast.dump(arg) for arg in node.args],
                        [],
                        f"{path.name}: {name}() must be called with keyword arguments",
                    )
        self.assertEqual(checked, self.CHAINED_CALLS)

    def test_every_link_accepts_the_embedder_precision(self):
        import inspect
        from rvc.realtime.callbacks import AudioCallbacks
        from rvc.realtime.core import Realtime, VoiceChanger
        from rvc.realtime.pipeline import create_pipeline

        for target in (AudioCallbacks.__init__, VoiceChanger.__init__,
                       Realtime.__init__, create_pipeline):
            with self.subTest(target=target.__qualname__):
                parameter = inspect.signature(target).parameters["embedder_precision"]
                self.assertEqual(parameter.default, "fp32")



class TrainSettingsTest(unittest.TestCase):
    """learning_rate and c_mel live only in logs/<model>/config.json.

    The Training tab reads them from there and writes them back, so the invariant that
    matters is that handing back what was read cannot disturb a hand-tuned file.
    """

    STOCK = {"learning_rate": 0.0001, "c_mel": 45}

    def _run_dir(self, temp_dir, config=None):
        model_dir = Path(temp_dir) / "run"
        model_dir.mkdir(parents=True, exist_ok=True)
        if config is not None:
            (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return str(model_dir)

    def test_an_unextracted_run_shows_the_stock_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                read_train_settings(self._run_dir(temp_dir), 48000), self.STOCK
            )

    def test_an_existing_run_shows_its_own_values(self):
        seed = {"train": {"learning_rate": 7e-05, "c_mel": 50}}
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                read_train_settings(self._run_dir(temp_dir, seed), 48000),
                {"learning_rate": 7e-05, "c_mel": 50},
            )

    def test_writing_back_what_was_read_is_a_no_op(self):
        seed = {"train": {"learning_rate": 7e-05, "c_mel": 50, "c_kl": 1.0}}
        with tempfile.TemporaryDirectory() as temp_dir:
            run = self._run_dir(temp_dir, seed)
            path = Path(run) / "config.json"
            before = path.read_bytes()
            apply_train_settings(run, **read_train_settings(run, 48000))
            self.assertEqual(path.read_bytes(), before)

    def test_only_the_changed_key_is_touched(self):
        seed = {
            "train": {"learning_rate": 7e-05, "c_mel": 50, "c_kl": 1.0, "seed": 1234},
            "model": {"text_enc_hidden_dim": 1024},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            run = self._run_dir(temp_dir, seed)
            apply_train_settings(run, learning_rate=2e-05)
            after = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(after["train"]["learning_rate"], 2e-05)
        self.assertEqual(after["train"]["c_mel"], 50)
        self.assertEqual(after["train"]["c_kl"], 1.0)
        self.assertEqual(after["model"], seed["model"])

    def test_none_leaves_a_setting_alone(self):
        seed = {"train": {"learning_rate": 7e-05, "c_mel": 50}}
        with tempfile.TemporaryDirectory() as temp_dir:
            run = self._run_dir(temp_dir, seed)
            apply_train_settings(run, learning_rate=None, c_mel=None)
            after = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(after["train"], seed["train"])

    def test_a_run_without_a_config_is_left_alone(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run = self._run_dir(temp_dir)
            apply_train_settings(run, learning_rate=2e-05)
            self.assertFalse((Path(run) / "config.json").exists())

    def test_the_stock_config_still_carries_the_documented_defaults(self):
        # docs/ADDING_AN_EMBEDDER_MODEL.md and the UI help text both quote these.
        root = Path(__file__).resolve().parents[1] / "rvc" / "configs"
        for sample_rate in (32000, 40000, 48000):
            with self.subTest(sample_rate=sample_rate):
                train = json.loads(
                    (root / f"{sample_rate}.json").read_text(encoding="utf-8")
                )["train"]
                self.assertEqual(train["learning_rate"], 0.0001)
                self.assertEqual(train["c_mel"], 45)

    def test_the_training_tab_passes_them_last(self):
        # run_train_script takes them as its final two parameters, and gradio forwards the
        # inputs list positionally, so the order of the two has to agree.
        import inspect
        from core import run_train_script

        names = list(inspect.signature(run_train_script).parameters)
        self.assertEqual(names[-2:], ["learning_rate", "c_mel"])

        source = (
            Path(__file__).resolve().parents[1] / "tabs" / "train" / "train.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "click"
                and any(
                    isinstance(kw.value, ast.Name) and kw.value.id == "enforce_terms"
                    for kw in node.keywords
                    if kw.arg == "fn"
                )
            ):
                inputs = next(kw.value for kw in node.keywords if kw.arg == "inputs")
                tail = [element.id for element in inputs.elts[-2:]]
                self.assertEqual(tail, ["learning_rate", "c_mel"])
                return
        self.fail("could not find the train button wiring")


if __name__ == "__main__":
    unittest.main()
