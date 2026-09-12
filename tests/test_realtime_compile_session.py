import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rvc.realtime import compile_session as compile_module
from rvc.realtime.compile_session import CompiledPath, CompileSession, CompileSettings


class CompileSessionTests(unittest.TestCase):
    def test_mode_is_shared_with_crepe_and_checkbox_saves_do_not_overwrite_it(self):
        from tabs.settings.sections import torch_compile as crepe_settings

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"realtime_compile_mode": "max-autotune"}), encoding="utf-8")
            with patch.object(compile_module, "CONFIG_PATH", str(config)), patch.object(
                crepe_settings, "CONFIG_PATH", str(config)
            ):
                self.assertEqual(compile_module.load_settings().mode, crepe_settings.load_torch_compile_mode())
                for mode in compile_module.MODES:
                    crepe_settings.save_torch_compile_mode(mode)
                    compile_module.save_settings(True, True)
                    self.assertEqual(compile_module.load_settings().mode, mode)
                    self.assertEqual(crepe_settings.get_torch_compile_settings()[1], mode)

    def test_settings_change_makes_torchcrepe_reload_instead_of_leaving_a_dead_model(self):
        """A TorchCompile settings change must not strand torchcrepe on a None model.

        torchcrepe reloads on ``infer.capacity`` alone and then always calls
        ``infer.model.to(device)``, so clearing the model used to make the next
        realtime start die in the compile warmup with
        ``AttributeError: 'NoneType' object has no attribute 'to'``.
        """
        import torchcrepe

        from tabs.settings.sections import torch_compile as crepe_settings

        infer = torchcrepe.core.infer
        saved = {name: getattr(infer, name) for name in ("model", "capacity") if hasattr(infer, name)}

        def restore():
            for name in ("model", "capacity"):
                if name in saved:
                    setattr(infer, name, saved[name])
                elif hasattr(infer, name):
                    delattr(infer, name)

        self.addCleanup(restore)

        stale = Mock(name="stale")
        stale.to.return_value = stale
        infer.model = stale
        infer.capacity = "full_speech"

        crepe_settings.reset_torchcrepe_compiled_model()

        # Only the capacity is invalidated: a concurrent audio thread still sees a
        # usable module, never a None it would call .to() on.
        self.assertIs(infer.model, stale)

        fresh = Mock(name="fresh")
        fresh.to.return_value = fresh

        def load(device, capacity, compile_model, compile_mode):
            infer.capacity = capacity
            infer.model = fresh

        with patch.object(torchcrepe.load, "model", side_effect=load) as loader:
            result = torchcrepe.core.infer(torch.zeros(1, 1024), model="full_speech", device="cpu")
        loader.assert_called_once()
        self.assertIs(result, fresh.return_value)

    def test_settings_ignore_legacy_flags_and_preserve_other_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"torch_compile_rvc_enabled": True}), encoding="utf-8")
            with patch.object(compile_module, "CONFIG_PATH", str(config)):
                self.assertEqual(compile_module.load_settings(), CompileSettings())
                compile_module.save_settings(True, False)
                self.assertEqual(compile_module.load_settings(), CompileSettings(True, False))
                self.assertTrue(json.loads(config.read_text())["torch_compile_rvc_enabled"])

    def test_disabled_and_cpu_paths_do_not_compile(self):
        with patch.object(torch, "compile") as compiler:
            disabled = CompiledPath("test", lambda x: x, False, "default", "cuda")
            cpu = CompiledPath("test", lambda x: x, True, "default", "cpu")
            self.assertEqual(disabled(3), 3)
            self.assertEqual(cpu(3), 3)
            compiler.assert_not_called()
            self.assertIn("CUDA", cpu.reason)

    def test_lazy_failure_falls_back_once_and_does_not_hide_model_errors(self):
        path = CompiledPath("test", lambda x: x + 1, False, "default", "cpu")
        compiled = Mock(side_effect=RuntimeError("compiler failure"))
        path.compiled = compiled
        with patch.object(torch.compiler, "cudagraph_mark_step_begin"):
            self.assertEqual(path(2), 3)
            self.assertEqual(path(3), 4)
        compiled.assert_called_once()
        path.eager = Mock(side_effect=ValueError("invalid input"))
        with self.assertRaisesRegex(ValueError, "invalid input"):
            path(2)

    def test_outputs_survive_next_iteration(self):
        buffer = torch.zeros(3)

        def run(value):
            return buffer.fill_(value)

        path = CompiledPath("test", run, False, "default", "cpu")
        path.compiled = run
        with patch.object(torch.compiler, "cudagraph_mark_step_begin") as step:
            first = path(1)
            second = path(2)
        self.assertEqual(step.call_count, 2)
        torch.testing.assert_close(first, torch.ones(3))
        torch.testing.assert_close(second, torch.full((3,), 2.0))

    def test_independent_flags_options_and_session_release(self):
        for embedder in (False, True):
            for rvc in (False, True):
                with self.subTest(embedder=embedder, rvc=rvc), patch.object(
                    torch.cuda, "is_available", return_value=True
                ), patch("torch.utils._triton.has_triton", return_value=True), patch.object(
                    torch, "compile", side_effect=lambda fn, **kwargs: fn
                ) as compiler:
                    session = CompileSession(
                        CompileSettings(embedder, rvc, "reduce-overhead"), lambda x: x, lambda x: x, "cuda",
                    )
                    self.assertEqual(compiler.call_count, int(embedder) + int(rvc))
                    for call in compiler.call_args_list:
                        self.assertTrue(call.kwargs["options"]["triton.cudagraphs"])
                        self.assertTrue(call.kwargs["options"]["triton.cudagraph_trees"])
                    session.close()
                    self.assertFalse(session.enabled)

    def test_compile_creation_failure_is_reported(self):
        with patch.object(torch.cuda, "is_available", return_value=True), patch(
            "torch.utils._triton.has_triton", return_value=True
        ), patch.object(torch, "compile", side_effect=RuntimeError("backend missing")):
            session = CompileSession(CompileSettings(True, False), lambda x: x, lambda x: x, "cuda")
        self.assertFalse(session.enabled)
        self.assertIn("backend missing", session.status())

    def test_warmup_runs_before_audio_start_and_cleans_buffers_on_failure(self):
        from rvc.realtime import callbacks as callback_module

        for fails in (False, True):
            with self.subTest(fails=fails):
                runtime = SimpleNamespace(
                    device="cuda:0",
                    vad=object(),
                    pipeline=SimpleNamespace(compile_session=SimpleNamespace(enabled=True)),
                    inference=Mock(side_effect=ValueError("bad model") if fails else None),
                    flush_buffers=Mock(), consecutive_silence_frames=3,
                )
                vc = SimpleNamespace(vc_model=runtime, block_frame=1024)
                original_vad = runtime.vad
                with patch.object(callback_module, "VoiceChanger", return_value=vc), patch.object(
                    callback_module, "Audio"
                ) as audio, patch.object(torch.random, "fork_rng"), patch.object(torch.cuda, "synchronize"):
                    if fails:
                        with self.assertRaisesRegex(ValueError, "bad model"):
                            callback_module.AudioCallbacks()
                    else:
                        callback_module.AudioCallbacks()
                        self.assertEqual(runtime.inference.call_count, 3)
                        waveform = runtime.inference.call_args.args[0]
                        self.assertEqual(waveform.shape, (1024,))
                        self.assertGreater(abs(waveform).max(), 0)
                    audio.return_value.start.assert_not_called()
                    runtime.flush_buffers.assert_called_once()
                    self.assertEqual(runtime.consecutive_silence_frames, 0)
                    self.assertIs(runtime.vad, original_vad)


if __name__ == "__main__":
    unittest.main()
