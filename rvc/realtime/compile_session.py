"""Session-scoped torch.compile wrappers used by realtime inference."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import torch
import torchcrepe

from tabs.settings.sections.torch_compile import (
    RealtimeCompileSettings,
    activate_compile_namespace,
    is_torch_compile_available,
    reset_failed_compile_namespace,
)


@dataclass
class CompileSessionStatus:
    component: str
    requested: bool
    active: bool = False
    backend: str = "eager"
    error: str | None = None
    cache_rebuilt: bool = False


class CrepeSession:
    """Own the otherwise-global torchcrepe model for one realtime session.

    torchcrepe's public API stores its model on ``torchcrepe.infer``. This wrapper
    serializes that global and never resets it from a live settings callback.
    """

    _global_lock = threading.RLock()
    _active_key: tuple | None = None

    def __init__(
        self,
        device,
        capacity: str,
        settings: RealtimeCompileSettings,
        signature: str,
    ):
        self.device = device
        self.capacity = capacity
        self.settings = settings
        self.signature = f"{capacity}|{signature}"
        self.status = CompileSessionStatus(
            component="crepe", requested=settings.crepe_enabled
        )
        self._namespace = None
        self._eager_model = None
        self._repair_allowed = True

    @property
    def key(self) -> tuple:
        return (
            str(self.device),
            self.capacity,
            self.settings.crepe_enabled,
            self.settings.mode,
            self.signature,
        )

    def _load(self, *, rebuild: bool = False) -> None:
        compile_requested = self.settings.crepe_enabled
        compile_enabled = compile_requested and is_torch_compile_available()
        torchcrepe.load.model(
            self.device,
            self.capacity,
            compile_model=False,
            compile_mode=self.settings.mode,
        )
        eager_model = torchcrepe.infer.model
        self._eager_model = eager_model
        if not compile_enabled:
            self.status.active = False
            self.status.backend = "eager"
            if compile_requested:
                self.status.error = "Triton/CUDA torch.compile backend is unavailable"
            CrepeSession._active_key = self.key
            return

        self._namespace = activate_compile_namespace(
            "crepe", self.signature, self.settings
        )
        if rebuild:
            reset_failed_compile_namespace(self._namespace)
            self._namespace.mkdir(parents=True, exist_ok=True)
            self.status.cache_rebuilt = True
        torchcrepe.infer.model = torch.compile(
            eager_model,
            mode=self.settings.mode,
            dynamic=False,
        )
        self.status.active = True
        self.status.backend = f"inductor/{self.settings.mode}"
        self.status.error = None
        CrepeSession._active_key = self.key

    def finish_warmup(self) -> None:
        self._repair_allowed = False

    def _fall_back_to_eager(self, first_error, retry_error=None):
        if self._eager_model is None:
            torchcrepe.load.model(
                self.device,
                self.capacity,
                compile_model=False,
                compile_mode=self.settings.mode,
            )
            self._eager_model = torchcrepe.infer.model
        else:
            torchcrepe.infer.model = self._eager_model
        CrepeSession._active_key = self.key
        self.status.active = False
        self.status.backend = "eager-fallback"
        detail = f"compile failed: {first_error}"
        if retry_error is not None:
            detail += f"; rebuild failed: {retry_error}"
        self.status.error = detail

    def predict(self, *args, **kwargs):
        with self._global_lock:
            kwargs["compile_model"] = False
            kwargs["compile_mode"] = self.settings.mode
            try:
                if CrepeSession._active_key != self.key:
                    self._load()
                if self._namespace is not None:
                    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(self._namespace)
                return torchcrepe.predict(*args, **kwargs)
            except Exception as first_error:
                if (
                    not self.settings.crepe_enabled
                    or not is_torch_compile_available()
                ):
                    raise
                if not self._repair_allowed:
                    self._fall_back_to_eager(first_error)
                    return torchcrepe.predict(*args, **kwargs)
                try:
                    if self._namespace is None:
                        self._namespace = activate_compile_namespace(
                            "crepe", self.signature, self.settings
                        )
                    self._load(rebuild=True)
                    return torchcrepe.predict(*args, **kwargs)
                except Exception as retry_error:
                    self._fall_back_to_eager(first_error, retry_error)
                    return torchcrepe.predict(*args, **kwargs)


class PennSession:
    """Session-scoped optional compilation for PENN's cached FCNF0++ model."""

    _global_lock = threading.RLock()
    _active_key: tuple | None = None

    def __init__(self, penn, device, settings, signature):
        self.penn = penn
        self.device = device
        self.settings = settings
        self.signature = signature
        self.status = CompileSessionStatus(
            component="fcnf0pp", requested=settings.fcnf0pp_enabled
        )
        self._namespace = None
        self._eager_model = None
        self._compiled_model = None
        self._repair_allowed = True

    @property
    def key(self) -> tuple:
        return (
            str(self.device),
            self.settings.fcnf0pp_enabled,
            self.settings.mode,
            self.signature,
        )

    def _compile_loaded_model(self, *, rebuild: bool = False) -> None:
        if self._eager_model is None:
            self._eager_model = self.penn.infer.model
        else:
            self.penn.infer.model = self._eager_model
        if not self.settings.fcnf0pp_enabled:
            self.status.backend = "eager"
            PennSession._active_key = self.key
            return
        if not is_torch_compile_available():
            self.status.backend = "eager"
            self.status.error = "Triton/CUDA torch.compile backend is unavailable"
            PennSession._active_key = self.key
            return

        self._namespace = activate_compile_namespace(
            "fcnf0pp", self.signature, self.settings
        )
        if rebuild:
            reset_failed_compile_namespace(self._namespace)
            self._namespace.mkdir(parents=True, exist_ok=True)
            self.status.cache_rebuilt = True
        self._compiled_model = torch.compile(
            self._eager_model,
            mode=self.settings.mode,
            dynamic=False,
        )
        self.penn.infer.model = self._compiled_model
        self.status.active = True
        self.status.backend = f"inductor/{self.settings.mode}"
        self.status.error = None
        PennSession._active_key = self.key

    def finish_warmup(self) -> None:
        self._repair_allowed = False

    def _fall_back_to_eager(self, first_error, retry_error=None) -> None:
        if self._eager_model is not None:
            self.penn.infer.model = self._eager_model
        PennSession._active_key = self.key
        self._compiled_model = None
        self.status.active = False
        self.status.backend = "eager-fallback"
        detail = f"compile failed: {first_error}"
        if retry_error is not None:
            detail += f"; rebuild failed: {retry_error}"
        self.status.error = detail

    def predict(self, predict, *args, **kwargs):
        with self._global_lock:
            if self._compiled_model is None and self._eager_model is None:
                result = predict(*args, **kwargs)
                try:
                    self._compile_loaded_model()
                except Exception as error:
                    self._fall_back_to_eager(error)
                return result

            if self._compiled_model is None:
                return predict(*args, **kwargs)

            if PennSession._active_key != self.key:
                self.penn.infer.model = self._compiled_model
                PennSession._active_key = self.key
            if self._namespace is not None:
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(self._namespace)
            try:
                return predict(*args, **kwargs)
            except Exception as first_error:
                if not self._repair_allowed or self._namespace is None:
                    self._fall_back_to_eager(first_error)
                    return predict(*args, **kwargs)
                try:
                    self._compile_loaded_model(rebuild=True)
                    return predict(*args, **kwargs)
                except Exception as retry_error:
                    self._fall_back_to_eager(first_error, retry_error)
                    return predict(*args, **kwargs)


class RvcCompileSession:
    def __init__(
        self,
        model,
        settings: RealtimeCompileSettings,
        signature: str,
    ):
        self.eager = model.infer
        self.callable = self.eager
        self.settings = settings
        self.signature = signature
        self.namespace = None
        self.status = CompileSessionStatus(
            component="rvc", requested=settings.rvc_enabled
        )
        self._repair_allowed = True

        if not settings.rvc_enabled:
            return
        if not is_torch_compile_available():
            self.status.error = "Triton/CUDA torch.compile backend is unavailable"
            return
        try:
            self._compile()
        except Exception as first_error:
            try:
                self._compile(rebuild=True)
                self.status.cache_rebuilt = True
            except Exception as retry_error:
                self.status.error = (
                    f"compile failed: {first_error}; rebuild failed: {retry_error}"
                )

    def _compile(self, *, rebuild: bool = False) -> None:
        self.namespace = activate_compile_namespace(
            "rvc", self.signature, self.settings
        )
        if rebuild:
            reset_failed_compile_namespace(self.namespace)
            self.namespace.mkdir(parents=True, exist_ok=True)
        self.callable = torch.compile(
            self.eager,
            mode=self.settings.mode,
            dynamic=False,
        )
        self.status.active = True
        self.status.backend = f"inductor/{self.settings.mode}"
        self.status.error = None

    def finish_warmup(self) -> None:
        self._repair_allowed = False

    def _fall_back_to_eager(self, first_error, retry_error=None) -> None:
        self.callable = self.eager
        self.status.active = False
        self.status.backend = "eager-fallback"
        detail = f"compile failed: {first_error}"
        if retry_error is not None:
            detail += f"; rebuild failed: {retry_error}"
        self.status.error = detail

    def __call__(self, *args, **kwargs):
        if self.namespace is not None:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(self.namespace)
        try:
            return self.callable(*args, **kwargs)
        except Exception as first_error:
            if not self.status.active or self.namespace is None:
                raise
            if not self._repair_allowed:
                self._fall_back_to_eager(first_error)
                return self.eager(*args, **kwargs)
            try:
                reset_failed_compile_namespace(self.namespace)
                self.namespace.mkdir(parents=True, exist_ok=True)
                self.callable = torch.compile(
                    self.eager,
                    mode=self.settings.mode,
                    dynamic=False,
                )
                result = self.callable(*args, **kwargs)
                self.status.cache_rebuilt = True
                return result
            except Exception as retry_error:
                self._fall_back_to_eager(first_error, retry_error)
                return self.eager(*args, **kwargs)
