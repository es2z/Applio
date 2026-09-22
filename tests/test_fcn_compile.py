from unittest.mock import patch

import torch

from rvc.lib.predictors.fcn.compile import FCNForward
from rvc.lib.predictors.fcn.profiles import FCNProfile


class TinyNetwork:
    def __call__(self, x):
        return torch.full((1, 2, 486), 0.2)


def test_compilation_failure_falls_back_without_changing_model():
    model = TinyNetwork()
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.compile", side_effect=RuntimeError("backend failure")),
    ):
        forward = FCNForward(model, FCNProfile(compile_model=True), "cuda")
    torch.testing.assert_close(forward(torch.zeros(1, 1, 1001)), model(None))
    assert forward.path.compiled is None
    assert forward.model is model


def test_numeric_compilation_failure_falls_back():
    model = TinyNetwork()
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.compile", return_value=lambda x: model(x) + 0.1),
    ):
        forward = FCNForward(model, FCNProfile(compile_model=True), "cuda")
    with patch("torch.compiler.cudagraph_mark_step_begin"):
        result = forward(torch.zeros(1, 1, 1001))
    torch.testing.assert_close(result, model(None))
    assert forward.path.compiled is None
