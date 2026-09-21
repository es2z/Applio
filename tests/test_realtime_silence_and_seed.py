import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from rvc.realtime import rng
from rvc.realtime.core import Realtime, silence_blocks_to_stop


class SilenceGateTests(unittest.TestCase):
    def test_window_longer_than_a_block_needs_more_than_one_silent_block(self):
        # The reference template: 21280-sample window, 15360-sample block.
        self.assertEqual(silence_blocks_to_stop(21280, 15360), 2)
        # A window that fits in one block still needs one silent block, never zero.
        self.assertEqual(silence_blocks_to_stop(15360, 15360), 1)
        self.assertEqual(silence_blocks_to_stop(1000, 15360), 1)
        self.assertEqual(silence_blocks_to_stop(21280, 0), 21280)
        # Three blocks of context take three silent blocks to clear.
        self.assertEqual(silence_blocks_to_stop(46000, 15360), 3)

    def test_tail_survives_until_the_whole_window_is_silence(self):
        runtime = SimpleNamespace(consecutive_input_silence=0, silence_blocks_to_stop=2)
        track = Realtime.track_input_silence.__get__(runtime)
        # Speech keeps the gate open.
        self.assertFalse(track(False))
        # The first silent block is still carrying the tail, so do not gate yet.
        self.assertFalse(track(True))
        self.assertTrue(track(True))
        self.assertTrue(track(True))
        # Speech resumes: the counter resets and the gate reopens immediately.
        self.assertFalse(track(False))
        self.assertFalse(track(True))

    def test_a_non_negative_threshold_does_not_mute_everything(self):
        """10 ** (0 / 20) is 1.0, which would call every possible input silent."""
        from rvc.realtime import core

        pipeline = SimpleNamespace(device="cpu", tgt_sr=48000)
        for threshold, expected in ((-90, 10 ** (-90 / 20)), (-60, 10 ** (-60 / 20)), (0, 0.0), (3, 0.0)):
            with self.subTest(threshold=threshold), patch.object(
                core, "create_pipeline", return_value=pipeline
            ), patch.object(core.tat, "Resample"):
                runtime = core.Realtime(silent_threshold=threshold)
            self.assertAlmostEqual(runtime.input_sensitivity, expected)


class VadWarmupTests(unittest.TestCase):
    """The detector's first frames are noise; scoring them makes room tone read as speech."""

    def _processor(self):
        from rvc.realtime.utils import vad as vad_module

        return vad_module, vad_module.VADProcessor(3, 16000, 30)

    def _fake_vad(self, verdicts):
        """A detector whose frame counter restarts per instance, like the real one."""
        calls = {"n": 0, "built": 0}

        class Fake:
            def __init__(self, mode):
                calls["built"] += 1
                self.i = 0

            def is_speech(self, frame, rate):
                verdict = verdicts(self.i)
                self.i += 1
                calls["n"] += 1
                return verdict

        return Fake, calls

    def test_warmup_frames_are_not_scored(self):
        vad_module, processor = self._processor()
        chunk = np.zeros(16000 * 30 // 1000 * 32, dtype=np.float32)
        Fake, calls = self._fake_vad(lambda i: i < processor.warmup_frames)
        with patch.object(vad_module.webrtcvad, "Vad", Fake):
            self.assertEqual(processor.speech_ratio(chunk), 0.0)
            self.assertFalse(processor.is_speech(chunk))
        # Every frame is still fed to the detector, only the verdicts are dropped,
        # and each call gets its own detector.
        self.assertEqual(calls["n"], 64)
        self.assertEqual(calls["built"], 2)

    def test_one_scored_frame_is_speech(self):
        vad_module, processor = self._processor()
        chunk = np.zeros(16000 * 30 // 1000 * 32, dtype=np.float32)
        Fake, _ = self._fake_vad(lambda i: i == processor.warmup_frames)
        with patch.object(vad_module.webrtcvad, "Vad", Fake):
            self.assertTrue(processor.is_speech(chunk))
            self.assertAlmostEqual(processor.speech_ratio(chunk), 1 / (32 - processor.warmup_frames))

    def test_a_detector_error_never_gates_real_audio(self):
        vad_module, processor = self._processor()
        chunk = np.zeros(16000 * 30 // 1000 * 32, dtype=np.float32)

        class Boom:
            def __init__(self, mode):
                pass

            def is_speech(self, frame, rate):
                raise RuntimeError("Error talking to VAD")

        with patch.object(vad_module.webrtcvad, "Vad", Boom):
            self.assertEqual(processor.speech_ratio(chunk), 1.0)
            self.assertTrue(processor.is_speech(chunk))

    def test_digital_silence_is_not_speech(self):
        _, processor = self._processor()
        self.assertFalse(processor.is_speech(np.zeros(15360, dtype=np.float32)))


class RealtimeSeedTests(unittest.TestCase):
    def _config(self, directory, payload):
        config = Path(directory) / "config.json"
        config.write_text(json.dumps(payload), encoding="utf-8")
        return config

    def test_seed_round_trips_and_negative_means_random(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, {"theme": "keep me"})
            with patch.object(rng, "CONFIG_PATH", str(config)):
                self.assertIsNone(rng.load_seed())
                rng.save_seed(1234)
                self.assertEqual(rng.load_seed(), 1234)
                rng.save_seed(-1)
                self.assertIsNone(rng.load_seed())
                rng.save_seed(0)
                self.assertEqual(rng.load_seed(), 0)
                self.assertEqual(json.loads(config.read_text())["theme"], "keep me")

    def test_unparseable_seed_is_treated_as_random(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, {"realtime_seed": "nonsense"})
            with patch.object(rng, "CONFIG_PATH", str(config)):
                self.assertIsNone(rng.load_seed())

    def test_applying_a_seed_makes_the_generator_noise_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, {"realtime_seed": 7})
            with patch.object(rng, "CONFIG_PATH", str(config)):
                self.assertEqual(rng.apply_seed(), 7)
                first = torch.randn(8)
                self.assertEqual(rng.apply_seed(), 7)
                torch.testing.assert_close(torch.randn(8), first)

    def test_random_seed_leaves_the_generator_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, {"realtime_seed": -1})
            with patch.object(rng, "CONFIG_PATH", str(config)):
                torch.manual_seed(11)
                expected = torch.randn(4)
                torch.manual_seed(11)
                self.assertIsNone(rng.apply_seed())
                torch.testing.assert_close(torch.randn(4), expected)


if __name__ == "__main__":
    unittest.main()
