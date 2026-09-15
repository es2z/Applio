import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rvc.train.extract import compile_extract


class ExtractCompileSettingsTests(unittest.TestCase):
    def test_mode_is_shared_and_flag_round_trips_without_losing_other_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(
                json.dumps({"torch_compile_mode": "max-autotune", "theme": "keep me"}),
                encoding="utf-8",
            )
            with patch.object(compile_extract, "CONFIG_PATH", str(config)):
                self.assertEqual(compile_extract.load_settings(), (False, "max-autotune"))
                compile_extract.save_enabled(True)
                self.assertEqual(compile_extract.load_settings(), (True, "max-autotune"))
                self.assertTrue(compile_extract.load_enabled())
                stored = json.loads(config.read_text(encoding="utf-8"))
                self.assertEqual(stored["theme"], "keep me")
                self.assertTrue(stored["training_compile_extraction"])

    def test_unknown_mode_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"torch_compile_mode": "nonsense"}), encoding="utf-8")
            with patch.object(compile_extract, "CONFIG_PATH", str(config)):
                self.assertEqual(compile_extract.load_settings()[1], "default")


class CompiledExtractorTests(unittest.TestCase):
    def test_disabled_returns_the_eager_callable_itself(self):
        eager = lambda x: x + 1
        with patch.object(compile_extract, "load_settings", return_value=(False, "default")):
            with patch.object(torch, "compile") as compiler:
                self.assertIs(compile_extract.compiled_extractor("E", eager, "cuda"), eager)
            compiler.assert_not_called()

    def test_compilation_failure_falls_back_to_the_eager_callable(self):
        eager = lambda x: x + 1
        with patch.object(compile_extract, "load_settings", return_value=(True, "default")), patch.object(
            torch.cuda, "is_available", return_value=True
        ), patch("torch.utils._triton.has_triton", return_value=True), patch.object(
            torch, "compile", side_effect=RuntimeError("no compiler")
        ):
            self.assertIs(compile_extract.compiled_extractor("E", eager, "cuda"), eager)

    def test_extraction_compiles_dynamically_and_without_cuda_graphs(self):
        """Every clip has its own length, so fixed shapes and CUDA graphs are wrong here."""
        eager = lambda x: x
        with patch.object(compile_extract, "load_settings", return_value=(True, "reduce-overhead")), patch.object(
            torch.cuda, "is_available", return_value=True
        ), patch("torch.utils._triton.has_triton", return_value=True), patch.object(
            torch, "compile", side_effect=lambda fn, **kwargs: fn
        ) as compiler:
            compile_extract.compiled_extractor("E", eager, "cuda")
        self.assertTrue(compiler.call_args.kwargs["dynamic"])
        self.assertFalse(compiler.call_args.kwargs["options"]["triton.cudagraphs"])


class CompileF0PredictorTests(unittest.TestCase):
    def _predictor(self, inner_name, inner):
        return SimpleNamespace(model=SimpleNamespace(**{inner_name: inner}))

    def test_rmvpe_and_fcpe_swap_the_attribute_that_is_actually_called(self):
        marker = object()
        with patch.object(compile_extract, "compiled_extractor", return_value=marker):
            rmvpe = self._predictor("model", lambda x: x)
            compile_extract.compile_f0_predictor(rmvpe, "rmvpe", "cuda")
            self.assertIs(rmvpe.model.model, marker)

            fcpe = self._predictor("infer", lambda x: x)
            compile_extract.compile_f0_predictor(fcpe, "fcpe", "cuda")
            self.assertIs(fcpe.model.infer, marker)

    def test_other_methods_and_missing_predictors_are_left_alone(self):
        with patch.object(compile_extract, "compiled_extractor") as compiled:
            compile_extract.compile_f0_predictor(None, "rmvpe", "cuda")
            # crepe compiles itself through get_torch_compile_settings; swift is CPU only.
            crepe = self._predictor("model", lambda x: x)
            compile_extract.compile_f0_predictor(crepe, "mangio-crepe-full", "cuda")
            compile_extract.compile_f0_predictor(crepe, "swift", "cuda")
        compiled.assert_not_called()


if __name__ == "__main__":
    unittest.main()
