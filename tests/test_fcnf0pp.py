import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from rvc.lib.predictors.f0 import FCNF0PP, FCNF0PP_SPEECH
from rvc.realtime.compile_session import PennSession
from tabs.settings.sections.torch_compile import RealtimeCompileSettings


class FCNF0PPTests(unittest.TestCase):
    def _predictor(
        self, pitch, periodicity, device="cpu", predictor_class=FCNF0PP
    ):
        fake_penn = types.SimpleNamespace(
            from_audio=Mock(
                return_value=(
                    torch.from_numpy(np.asarray(pitch, np.float32)).unsqueeze(0),
                    torch.from_numpy(
                        np.asarray(periodicity, np.float32)
                    ).unsqueeze(0),
                )
            )
        )
        with patch.dict(sys.modules, {"penn": fake_penn}):
            predictor = predictor_class(device)
        return predictor, fake_penn

    def test_official_api_settings_and_sanitized_exact_length(self):
        below_threshold = max(0.0, FCNF0PP.PERIODICITY_THRESHOLD - 0.001)
        predictor, fake_penn = self._predictor(
            [np.nan, np.inf, 40.0, 220.0, 1200.0, 330.0],
            [1.0, 1.0, 1.0, below_threshold, 1.0, 0.5],
        )
        with patch("pathlib.Path.is_file", return_value=False):
            f0 = predictor.get_f0(np.zeros(8 * 160, np.float32), 50, 1100, 8)

        self.assertEqual(f0.shape, (8,))
        self.assertEqual(f0.dtype, np.float32)
        self.assertTrue(np.isfinite(f0).all())
        np.testing.assert_array_equal(
            f0, np.array([0, 0, 0, 0, 0, 330, 0, 0], np.float32)
        )
        kwargs = fake_penn.from_audio.call_args.kwargs
        self.assertEqual(kwargs["sample_rate"], 16000)
        self.assertEqual(kwargs["hopsize"], 0.01)
        self.assertEqual(kwargs["center"], "half-hop")
        self.assertEqual(kwargs["decoder"], "viterbi")
        self.assertIsNone(kwargs["interp_unvoiced_at"])
        self.assertIsNone(kwargs["checkpoint"])
        self.assertIsNone(kwargs["gpu"])

    def test_length_mismatch_is_tail_crop_without_interpolation(self):
        pitch = np.arange(100, 110, dtype=np.float32)
        predictor, _ = self._predictor(pitch, np.ones(10, np.float32))
        f0 = predictor.get_f0(np.zeros(4 * 160, np.float32), 50, 1100, 4)
        np.testing.assert_array_equal(f0, pitch[:4])

    def test_cuda_device_index_is_forwarded(self):
        predictor, fake_penn = self._predictor([220.0], [1.0], "cuda:3")
        predictor.get_f0(np.zeros(160, np.float32), 50, 1100, 1)
        self.assertEqual(fake_penn.from_audio.call_args.kwargs["gpu"], 3)

    def test_speech_variant_closes_one_frame_voicing_hole(self):
        periodicity = [0.04, 0.06, 0.06, 0.04, 0.06, 0.06, 0.04]
        predictor, _ = self._predictor(
            [220.0] * len(periodicity),
            periodicity,
            predictor_class=FCNF0PP_SPEECH,
        )
        f0 = predictor.get_f0(
            np.zeros(len(periodicity) * 160, np.float32),
            50,
            1100,
            len(periodicity),
        )
        np.testing.assert_array_equal(
            f0,
            np.array([0, 220, 220, 220, 220, 220, 0], np.float32),
        )


class PennSessionTests(unittest.TestCase):
    def test_optional_compile_activates_after_eager_model_load(self):
        eager = torch.nn.Linear(2, 2)
        compiled = Mock()
        fake_penn = types.SimpleNamespace(
            infer=types.SimpleNamespace(model=eager)
        )
        settings = RealtimeCompileSettings(
            mode="reduce-overhead", fcnf0pp_enabled=True
        )
        session = PennSession(fake_penn, "cuda:0", settings, "test")
        predict = Mock(return_value="result")

        with (
            patch(
                "rvc.realtime.compile_session.is_torch_compile_available",
                return_value=True,
            ),
            patch(
                "rvc.realtime.compile_session.activate_compile_namespace"
            ) as namespace,
            patch("rvc.realtime.compile_session.torch.compile", return_value=compiled),
        ):
            namespace.return_value = Mock()
            self.assertEqual(session.predict(predict), "result")

        self.assertIs(fake_penn.infer.model, compiled)
        self.assertTrue(session.status.active)
        self.assertEqual(session.status.backend, "inductor/reduce-overhead")

    def test_live_compile_failure_falls_back_to_eager(self):
        eager = torch.nn.Linear(2, 2)
        compiled = Mock()
        fake_penn = types.SimpleNamespace(
            infer=types.SimpleNamespace(model=compiled)
        )
        settings = RealtimeCompileSettings(fcnf0pp_enabled=True)
        session = PennSession(fake_penn, "cuda:0", settings, "test")
        session._eager_model = eager
        session._compiled_model = compiled
        session._namespace = Mock()
        session._repair_allowed = False
        session.status.active = True
        PennSession._active_key = session.key
        predict = Mock(side_effect=[RuntimeError("graph failed"), "eager result"])

        self.assertEqual(session.predict(predict), "eager result")
        self.assertIs(fake_penn.infer.model, eager)
        self.assertFalse(session.status.active)
        self.assertEqual(session.status.backend, "eager-fallback")


if __name__ == "__main__":
    unittest.main()
