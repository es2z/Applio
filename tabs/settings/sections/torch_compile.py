import os
import sys
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.configs.config_utils import load_config, update_config

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")

# Available torch.compile modes
TORCH_COMPILE_MODES = ["default", "reduce-overhead", "max-autotune"]


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
    """Get both torch compile settings."""
    config = load_config(CONFIG_PATH)
    enabled = bool(config.get("torch_compile_enabled", False))
    mode = config.get("torch_compile_mode", "default")
    if mode not in TORCH_COMPILE_MODES:
        mode = "default"
    return enabled, mode


# Initialize cache if torch compile is enabled at startup
if load_torch_compile_enabled():
    setup_torch_compile_cache()
