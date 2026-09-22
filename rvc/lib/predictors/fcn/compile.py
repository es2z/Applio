"""Opt-in network compilation; fallback never resets the owning FCN stream."""

import torch

from .decoder import FCNDecoder


class FCNForward:
    def __init__(self, model, profile, device):
        from rvc.realtime.compile_session import CompiledPath

        self.model = model
        self.profile = profile
        self.path = CompiledPath(
            "FCN-993",
            model,
            True,
            profile.compile_mode,
            device,
            dynamic=True,
            cudagraphs=False,
        )
        self.validated_shapes = set()

    def __call__(self, audio):
        activation = self.path(audio)
        shape = tuple(audio.shape)
        if self.path.compiled is not None and shape not in self.validated_shapes:
            eager = self.model(audio)
            decoder = FCNDecoder()
            actual_cents, _, actual_conf = decoder(activation)
            expected_cents, _, expected_conf = decoder(eager)
            close = torch.allclose(activation, eager, atol=1e-5, rtol=1e-4)
            close = close and bool(
                torch.all((actual_cents - expected_cents).abs() <= 0.1)
            )
            close = close and torch.equal(activation.argmax(-1), eager.argmax(-1))
            if self.profile.method == "fcn-993-rvc":
                for threshold in (
                    self.profile.enter_threshold,
                    self.profile.exit_threshold,
                ):
                    close = close and torch.equal(
                        actual_conf >= threshold, expected_conf >= threshold
                    )
            if not close:
                self.path.fallback(
                    RuntimeError(
                        "FCN compiled output failed eager parity; keeping eager network"
                    )
                )
                return eager
            self.validated_shapes.add(shape)
        return activation

    def status(self):
        return self.path.status()
