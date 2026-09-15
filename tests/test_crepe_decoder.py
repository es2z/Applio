import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torchcrepe

from rvc.lib.predictors import crepe_decoder


class CrepeDecoderSettingTests(unittest.TestCase):
    def _config(self, directory, payload):
        config = Path(directory) / "config.json"
        config.write_text(json.dumps(payload), encoding="utf-8")
        return config

    def test_default_is_what_mangio_crepe_has_always_used(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, {})
            with patch.object(crepe_decoder, "CONFIG_PATH", str(config)):
                self.assertEqual(crepe_decoder.load_decoder(), "viterbi")
                self.assertIs(crepe_decoder.resolve_decoder(), torchcrepe.decode.viterbi)

    def test_round_trip_without_losing_other_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, {"theme": "keep me"})
            with patch.object(crepe_decoder, "CONFIG_PATH", str(config)):
                for name in crepe_decoder.DECODERS:
                    crepe_decoder.save_decoder(name)
                    self.assertEqual(crepe_decoder.load_decoder(), name)
                    self.assertIs(
                        crepe_decoder.resolve_decoder(), getattr(torchcrepe.decode, name)
                    )
                self.assertEqual(json.loads(config.read_text())["theme"], "keep me")

    def test_unknown_names_fall_back_rather_than_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, {"mangio_crepe_decoder": "nonsense"})
            with patch.object(crepe_decoder, "CONFIG_PATH", str(config)):
                self.assertEqual(crepe_decoder.load_decoder(), "viterbi")
                crepe_decoder.save_decoder("also nonsense")
                self.assertEqual(
                    json.loads(config.read_text())["mangio_crepe_decoder"], "viterbi"
                )
        self.assertIs(
            crepe_decoder.resolve_decoder("nonsense"), torchcrepe.decode.viterbi
        )

    def test_the_setting_applies_to_mangio_crepe_only(self):
        for method in ("mangio-crepe", "mangio-crepe-full", "mangio-crepe-full-speech"):
            self.assertTrue(crepe_decoder.uses_mangio_crepe(method), method)
        for method in ("crepe", "crepe-full", "crepe-full-speech", "rmvpe", "fcpe", "swift"):
            self.assertFalse(crepe_decoder.uses_mangio_crepe(method), method)

    def test_every_offered_decoder_exists_in_torchcrepe(self):
        for name in crepe_decoder.DECODERS:
            self.assertTrue(hasattr(torchcrepe.decode, name), name)
        for name in crepe_decoder.REPEATABLE_DECODERS:
            self.assertIn(name, crepe_decoder.DECODERS)


class MangioCrepeUsesTheSettingTests(unittest.TestCase):
    def test_get_f0_passes_the_configured_decoder(self):
        from rvc.lib.predictors import f0 as f0_module

        mangio = f0_module.MANGIO_CREPE(device="cpu")
        with patch.object(
            f0_module, "resolve_decoder", return_value=torchcrepe.decode.weighted_argmax
        ), patch.object(f0_module, "get_torch_compile_settings", return_value=(False, "default")), patch.object(
            f0_module.torchcrepe, "predict", side_effect=RuntimeError("stop here")
        ) as predict:
            with self.assertRaisesRegex(RuntimeError, "stop here"):
                mangio.get_f0(__import__("numpy").ones(16000, dtype="float32"), 50.0, 1680.0, 100)
        self.assertIs(
            predict.call_args.kwargs["decoder"], torchcrepe.decode.weighted_argmax
        )


if __name__ == "__main__":
    unittest.main()
