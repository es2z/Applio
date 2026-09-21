"""Optional compilation owned by one realtime pipeline, never by shared models."""

import os
import traceback
from dataclasses import dataclass

import torch

from assets.i18n.i18n import I18nAuto
from rvc.configs.config_utils import load_config, update_config

i18n = I18nAuto()

CONFIG_PATH = os.path.join(os.getcwd(), "assets", "config.json")
MODES = ("default", "reduce-overhead", "max-autotune")


@dataclass(frozen=True)
class CompileSettings:
    embedder: bool = False
    rvc: bool = False
    mode: str = "default"


def load_settings():
    config = load_config(CONFIG_PATH)
    mode = config.get("torch_compile_mode", "default")
    return CompileSettings(
        bool(config.get("realtime_compile_embedder", False)),
        bool(config.get("realtime_compile_rvc", False)),
        mode if mode in MODES else "default",
    )


def save_settings(embedder, rvc):
    if not update_config(CONFIG_PATH, {
        "realtime_compile_embedder": bool(embedder),
        "realtime_compile_rvc": bool(rvc),
    }):
        raise OSError("Could not save realtime compile settings")


class CompiledPath:
    """Keep the eager callable for recovery, including lazy compilation failures."""

    def __init__(self, name, eager, enabled, mode, device, dynamic=False, cudagraphs=None):
        """dynamic/cudagraphs default to the realtime shape: one fixed-size window per
        call, CUDA graphs wherever the mode asks for them. Training extraction passes
        dynamic=True and cudagraphs=False because every clip has its own length."""
        self.name = name
        self.eager = eager
        self.compiled = None
        self.state = "OFF"
        self.reason = ""
        if not enabled:
            return
        try:
            if torch.device(device).type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is not available")
            from torch.utils._triton import has_triton

            if not has_triton():
                raise RuntimeError("A compatible Triton backend is not available")
            from torch._inductor import list_mode_options

            options = dict(list_mode_options(mode))
            options.update({
                "triton.cudagraphs": (mode != "default") if cudagraphs is None else cudagraphs,
                "triton.cudagraph_trees": True,
                "fx_graph_cache": True,
            })
            # Do not call the CREPE setup/reset helpers: those mutate global state.
            os.environ.setdefault(
                "TORCHINDUCTOR_CACHE_DIR",
                os.path.join(os.getcwd(), ".torch_compile_cache"),
            )
            self.compiled = torch.compile(
                eager, backend="inductor", options=options, fullgraph=False,
                dynamic=dynamic,
            )
            self.state = "Preparing compilation"
        except Exception as exc:  # noqa: BLE001 - backend errors vary by PyTorch/Triton version
            self.fallback(exc)

    def fallback(self, exc):
        self.compiled = None
        self.state = "Fell back to normal inference"
        self.reason = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, UnicodeDecodeError):
            self.reason += "; start with run-applio.bat or python -X utf8."
        print(f"[Realtime compile] {self.name}: {self.reason}")
        traceback.print_exc()

    def __call__(self, *args, **kwargs):
        if self.compiled is None:
            return self.eager(*args, **kwargs)
        try:
            # Each path owns its outputs; no CUDA-graph tensor escapes into the
            # next path/iteration or the audio/SOLA buffers.
            torch.compiler.cudagraph_mark_step_begin()
            result = self.compiled(*args, **kwargs).clone()
            self.state = "Using compiled inference"
            return result
        except Exception as exc:  # noqa: BLE001 - retry eager; real model errors propagate below
            self.fallback(exc)
            # A genuine model/input failure still propagates from eager execution.
            return self.eager(*args, **kwargs)

    def status(self):
        reason = f" ({self.reason})" if self.reason else ""
        return f"{self.name}: {i18n(self.state)}{reason}"


class CompileSession:
    def __init__(self, settings, embedder, rvc, device):
        self.settings = settings
        self.embedder = CompiledPath(
            "Embedder", embedder, settings.embedder, settings.mode, device,
        )
        self.rvc = CompiledPath("RVC", rvc, settings.rvc, settings.mode, device)

    @property
    def enabled(self):
        return self.embedder.compiled is not None or self.rvc.compiled is not None

    def status(self):
        if not (self.settings.embedder or self.settings.rvc):
            return ""
        return "\n" + self.embedder.status() + "\n" + self.rvc.status()

    def close(self):
        self.embedder.compiled = None
        self.rvc.compiled = None
