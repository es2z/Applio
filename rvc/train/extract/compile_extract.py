"""Optional torch.compile for the training feature-extraction pass.

Extraction is the one part of training where compiling measurably pays off.
Measured on an RTX 4090 (torch 2.13.0+cu132 / triton 3.7.1) over 60 clips totalling
223 s, with eager timed both before and after the compiled run so that warm-up
cannot be mistaken for a speedup (min of 3 repetitions each):

    embedder (kushinada-hubert-large)   10.4 ms/file -> 7.8 ms    x1.32
    RMVPE                               24.3 ms/file -> 18.1 ms   x1.33
    FCPE                                 2.6 ms/file -> 1.7 ms    x1.52

The training step itself is deliberately left alone. Compiling net_g and net_d
measured x1.03 against an eager control that reproduced to x1.00, needs MSVC on PATH
for inductor's C++ wrapper, costs minutes of compile time per run, and would put
``_orig_mod.`` in front of every key of the saved checkpoint. Enabling TF32 matmul on
top of it measured x0.99 - this model is convolution bound and cuDNN already runs its
convolutions in TF32.

Compilation is off by default because those per-file numbers are not the whole story:
tracing costs about 17 s per worker process and per stage, which the inductor cache does
not remove, and end to end on 200 clips the compiled run measured 61 s against 27 s
eager. The saving is ~8.8 ms/file, so the break-even is around 4000 clips (roughly four
hours of dataset). Below that, leave this off.

Every clip has its own length (771 distinct lengths across the 1364 files of the
reference dataset), so everything here is compiled with ``dynamic=True`` and no CUDA
graphs: graphs need fixed shapes and inductor skips them under dynamic shapes anyway.
CREPE and mangio-crepe are not touched here - they compile themselves through
``get_torch_compile_settings`` in ``rvc/lib/predictors/f0.py``, driven by the
"Enable TorchCompile (CREPE)" setting.
"""

import os

from rvc.configs.config_utils import load_config, update_config
from rvc.realtime.compile_session import MODES
from rvc.realtime.compile_session import CompiledPath

CONFIG_PATH = os.path.join(os.getcwd(), "assets", "config.json")


def load_settings():
    """Return (enabled, mode). The mode is the one shared by every TorchCompile path."""
    config = load_config(CONFIG_PATH)
    mode = config.get("torch_compile_mode", "default")
    return (
        bool(config.get("training_compile_extraction", False)),
        mode if mode in MODES else "default",
    )


def load_enabled():
    return load_settings()[0]


def save_enabled(enabled):
    if not update_config(
        CONFIG_PATH, {"training_compile_extraction": bool(enabled)}
    ):
        raise OSError("Could not save the training extraction compile setting")


def compiled_extractor(name, eager, device):
    """Wrap one extraction callable, falling back to eager on any compilation failure.

    Returns the eager callable untouched when the setting is off, so a caller can use
    the result unconditionally.
    """
    enabled, mode = load_settings()
    if not enabled:
        return eager
    path = CompiledPath(
        name, eager, True, mode, device, dynamic=True, cudagraphs=False
    )
    if path.compiled is None:
        return eager
    print(f"[Extract] Compiling {name} ({mode}). The first file may take a while.")
    return path


def compile_f0_predictor(predictor, f0_method, device):
    """Compile the neural part of an F0 predictor in place, where there is one.

    Only rmvpe and fcpe are handled: crepe/mangio-crepe compile themselves, and swift
    runs on CPU. Both attributes replaced here are only ever called, never introspected,
    after the predictor is constructed.
    """
    if predictor is None:
        return
    if f0_method == "rmvpe":
        inner = getattr(getattr(predictor, "model", None), "model", None)
        if inner is not None:
            predictor.model.model = compiled_extractor("RMVPE", inner, device)
    elif f0_method == "fcpe":
        inner = getattr(getattr(predictor, "model", None), "infer", None)
        if inner is not None:
            predictor.model.infer = compiled_extractor("FCPE", inner, device)
