import os
import sys
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.configs.config_utils import load_config, update_config

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")

# Available torch.compile modes
TORCH_COMPILE_MODES = ["default", "reduce-overhead", "max-autotune"]

# Cache triton availability check
_triton_available = None


def is_triton_available():
    """Check if triton is available."""
    global _triton_available
    if _triton_available is None:
        try:
            import triton
            _triton_available = True
        except ImportError:
            _triton_available = False
    return _triton_available


def is_torch_compile_available():
    """Check if torch.compile can be used.

    Returns True if:
    - triton is available
    - torch.compile exists
    - CUDA is available
    """
    return (
        is_triton_available() and
        hasattr(torch, 'compile') and
        torch.cuda.is_available()
    )


def setup_torch_compile_cache():
    """Enable torch.compile cache for faster startup on subsequent runs (PyTorch 2.4+)"""
    if hasattr(torch, "_inductor"):
        torch._inductor.config.fx_graph_cache = True
        torch._inductor.config.autotune_local_cache = True


def load_torch_compile_enabled():
    """Load torch compile enabled state from config."""
    config = load_config(CONFIG_PATH)
    return bool(config.get("torch_compile_enabled", False))


def load_torch_compile_mode():
    """Load torch compile mode from config."""
    config = load_config(CONFIG_PATH)
    return config.get("torch_compile_mode", "default")


def save_torch_compile_enabled(enabled: bool):
    """Save torch compile enabled state to config."""
    update_config(CONFIG_PATH, {"torch_compile_enabled": bool(enabled)})
    if enabled:
        setup_torch_compile_cache()


def save_torch_compile_mode(mode: str):
    """Save torch compile mode to config."""
    if mode not in TORCH_COMPILE_MODES:
        mode = "default"
    update_config(CONFIG_PATH, {"torch_compile_mode": mode})


def get_torch_compile_settings():
    """Get torch compile settings for use in inference.

    Returns (enabled, mode) tuple. The 'enabled' value will be False if:
    - User has disabled it in config
    - torch.compile is not available (no triton, no CUDA, etc.)
    """
    config = load_config(CONFIG_PATH)
    enabled = bool(config.get("torch_compile_enabled", False))
    mode = config.get("torch_compile_mode", "default")
    if mode not in TORCH_COMPILE_MODES:
        mode = "default"

    # Only actually enable if all requirements are met
    if enabled and not is_torch_compile_available():
        enabled = False

    return enabled, mode


# Initialize cache if torch compile is enabled at startup
if load_torch_compile_enabled():
    setup_torch_compile_cache()
