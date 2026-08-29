import unittest
from unittest.mock import patch

import numpy as np
import torch

from rvc.lib.predictors.crepe_models import (
    CREPE_UI_METHODS,
    resolve_crepe_model,
)
from rvc.lib.predictors import f0 as f0_module


class CrepeModelTest(unittest.TestCase):
    def test_crepe_ui_methods_are_paired_by_capacity(self):
        self.assertEqual(
            CREPE_UI_METHODS,
            [
                "crepe-tiny",
                "mangio-crepe-tiny",
                "crepe-small",
                "mangio-crepe-small",
                "crepe-medium",
                "mangio-crepe-medium",
                "crepe-large",
                "mangio-crepe-large",
                "crepe-full",
                "mangio-crepe-full",
                "crepe-full-speech",
                "mangio-crepe-full-speech",
            ],
        )

    def test_resolve_crepe_model(self):
        cases = [
            ("crepe", "full"),
            ("mangio-crepe", "full"),
            ("crepe-full", "full"),
            ("mangio-crepe-full", "full"),
            ("crepe-full-speech", "full_speech"),
            ("mangio-crepe-full-speech", "full_speech"),
        ]
        for method, model in cases:
            with self.subTest(method=method):
                self.assertEqual(resolve_crepe_model(method), model)

    def test_speech_predictors_forward_full_speech_model(self):
        for predictor_type in (f0_module.CREPE, f0_module.MANGIO_CREPE):
            with self.subTest(predictor=predictor_type.__name__):
                captured = {}

                def fake_predict(*args, **kwargs):
                    captured["model"] = kwargs["model"]
                    pitch = torch.full((1, 10), 220.0)
                    periodicity = torch.ones_like(pitch)
                    return pitch, periodicity

                with patch.object(f0_module.torchcrepe, "predict", fake_predict), patch.object(
                    f0_module,
                    "get_torch_compile_settings",
                    return_value=(False, "default"),
                ):
                    predictor = predictor_type(
                        device="cpu", sample_rate=16000, hop_size=160
                    )
                    predictor.get_f0(
                        np.ones(1600, dtype=np.float32),
                        p_len=10,
                        model="full_speech",
                    )

                self.assertEqual(captured["model"], "full_speech")


if __name__ == "__main__":
    unittest.main()
