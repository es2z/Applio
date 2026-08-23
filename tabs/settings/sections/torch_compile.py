"""Persistent torch.compile settings and cache lifecycle.

UI changes are intentionally pending settings: an active realtime session owns an
immutable snapshot and is never reset from a Gradio change callback.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.configs.config_utils import load_config, update_config


CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")
CACHE_ROOT = Path(now_dir, ".torch_compile_cache", "v2")
TORCH_COMPILE_MODES = [
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
PYTORCH_TRITON_SERIES = {
    (2, 7): (3, 3),
    (2, 8): (3, 4),
    (2, 9): (3, 5),
    (2, 10): (3, 6),
    (2, 11): (3, 6),
    (2, 12): (3, 7),
    (2, 13): (3, 7),
}


@dataclass(frozen=True)
class RealtimeCompileSettings:
    crepe_enabled: bool = False
    rvc_enabled: bool = False
    mode: str = "default"
    fcnf0pp_enabled: bool = False

    @property
    def any_enabled(self) -> bool:
        return self.crepe_enabled or self.rvc_enabled or self.fcnf0pp_enabled


def bootstrap_torch_compile_environment() -> None:
    """Set cache variables before torch/Inductor is imported by the application."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(CACHE_ROOT / "shared"))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    # No fake TORCHINDUCTOR_DISABLE_TRITON variable: current Inductor ignores it.


def _parse_version(value: str) -> tuple[int, int] | None:
    try:
        major, minor, *_ = value.split(".")
        return int(major), int(minor)
    except (TypeError, ValueError):
        return None


def get_triton_status() -> tuple[bool, str]:
    if sys.platform != "win32":
        try:
            version = importlib.metadata.version("triton")
            return True, f"Triton {version}"
        except importlib.metadata.PackageNotFoundError:
            return False, "Triton is not installed"
    try:
        version = importlib.metadata.version("triton-windows")
    except importlib.metadata.PackageNotFoundError:
        return False, "triton-windows is not installed"
    parsed = _parse_version(version)
    try:
        torch_series = _parse_version(importlib.metadata.version("torch"))
    except importlib.metadata.PackageNotFoundError:
        torch_series = None
    expected = PYTORCH_TRITON_SERIES.get(torch_series)
    if expected is not None and parsed != expected:
        return (
            False,
            f"triton-windows {version} is incompatible with this PyTorch pin; "
            f"expected the {expected[0]}.{expected[1]} series",
        )
    return True, f"triton-windows {version}"


def is_triton_available() -> bool:
    return get_triton_status()[0]


def is_torch_compile_available() -> bool:
    try:
        import torch

        return bool(
            hasattr(torch, "compile")
            and torch.cuda.is_available()
            and is_triton_available()
        )
    except Exception:
        return False


def load_realtime_compile_settings() -> RealtimeCompileSettings:
    config = load_config(CONFIG_PATH)
    mode = config.get("torch_compile_mode", "default")
    if mode not in TORCH_COMPILE_MODES:
        mode = "default"
    # Migrate the original single checkbox to the CREPE-specific setting.
    crepe = bool(
        config.get(
            "torch_compile_crepe_enabled",
            config.get("torch_compile_enabled", False),
        )
    )
    return RealtimeCompileSettings(
        crepe_enabled=crepe,
        rvc_enabled=bool(config.get("torch_compile_rvc_enabled", False)),
        mode=mode,
        fcnf0pp_enabled=bool(
            config.get("torch_compile_fcnf0pp_enabled", False)
        ),
    )


def save_realtime_compile_settings(
    crepe_enabled: bool,
    rvc_enabled: bool,
    mode: str,
    fcnf0pp_enabled: bool = False,
) -> None:
    if mode not in TORCH_COMPILE_MODES:
        mode = "default"
    update_config(
        CONFIG_PATH,
        {
            "torch_compile_enabled": bool(crepe_enabled),
            "torch_compile_crepe_enabled": bool(crepe_enabled),
            "torch_compile_rvc_enabled": bool(rvc_enabled),
            "torch_compile_fcnf0pp_enabled": bool(fcnf0pp_enabled),
            "torch_compile_mode": mode,
        },
    )


def load_torch_compile_enabled() -> bool:
    return load_realtime_compile_settings().crepe_enabled


def load_torch_compile_rvc_enabled() -> bool:
    return load_realtime_compile_settings().rvc_enabled


def load_torch_compile_fcnf0pp_enabled() -> bool:
    return load_realtime_compile_settings().fcnf0pp_enabled


def load_torch_compile_mode() -> str:
    return load_realtime_compile_settings().mode


def save_torch_compile_enabled(enabled: bool) -> None:
    settings = load_realtime_compile_settings()
    save_realtime_compile_settings(
        enabled,
        settings.rvc_enabled,
        settings.mode,
        settings.fcnf0pp_enabled,
    )


def save_torch_compile_rvc_enabled(enabled: bool) -> None:
    settings = load_realtime_compile_settings()
    save_realtime_compile_settings(
        settings.crepe_enabled,
        enabled,
        settings.mode,
        settings.fcnf0pp_enabled,
    )


def save_torch_compile_fcnf0pp_enabled(enabled: bool) -> None:
    settings = load_realtime_compile_settings()
    save_realtime_compile_settings(
        settings.crepe_enabled,
        settings.rvc_enabled,
        settings.mode,
        enabled,
    )


def save_torch_compile_mode(mode: str) -> None:
    settings = load_realtime_compile_settings()
    save_realtime_compile_settings(
        settings.crepe_enabled,
        settings.rvc_enabled,
        mode,
        settings.fcnf0pp_enabled,
    )


def get_torch_compile_settings() -> tuple[bool, str]:
    """Backward-compatible settings for non-realtime CREPE callers."""
    settings = load_realtime_compile_settings()
    enabled = settings.crepe_enabled and is_torch_compile_available()
    return enabled, settings.mode


def compile_namespace(
    component: str, signature: str, settings: RealtimeCompileSettings
) -> Path:
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = str(torch.version.cuda)
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        torch_version = cuda_version = gpu = "unknown"
    try:
        triton_version = importlib.metadata.version(
            "triton-windows" if sys.platform == "win32" else "triton"
        )
    except importlib.metadata.PackageNotFoundError:
        triton_version = "missing"
    raw = "|".join(
        (
            sys.version.split()[0],
            torch_version,
            cuda_version,
            triton_version,
            gpu,
            component,
            signature,
            settings.mode,
        )
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return CACHE_ROOT / component / digest


def activate_compile_namespace(
    component: str, signature: str, settings: RealtimeCompileSettings
) -> Path:
    namespace = compile_namespace(component, signature, settings)
    namespace.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(namespace)
    return namespace


def reset_failed_compile_namespace(namespace: Path) -> None:
    """Clear only the namespace owned by the failed stopped/warmup session."""
    try:
        import torch

        if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
            torch.compiler.reset()
        elif hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
    except Exception:
        pass
    resolved = namespace.resolve()
    root = CACHE_ROOT.resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"Refusing to remove compile cache outside {root}")
    if resolved.exists():
        shutil.rmtree(resolved)


def clear_inactive_compile_caches() -> str:
    if not CACHE_ROOT.exists():
        return "No application compile caches exist."
    removed = 0
    for component in CACHE_ROOT.iterdir():
        if component.name == "shared" or not component.is_dir():
            continue
        shutil.rmtree(component)
        removed += 1
    return f"Removed {removed} inactive compile cache group(s)."
