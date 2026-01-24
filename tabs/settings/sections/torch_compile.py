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

# Track current settings to detect changes
_last_compile_settings = None


def is_triton_available():
    """Check if triton is installed."""
    global _triton_available
    if _triton_available is None:
        try:
            import triton
            _triton_available = True
        except ImportError:
            _triton_available = False
    return _triton_available


def should_use_triton():
    """Check if triton should be used for torch.compile.

    Returns False if:
    - triton is not installed
    - User has enabled 'disable triton' option
    """
    if not is_triton_available():
        return False
    # Check if user has disabled triton
    config = load_config(CONFIG_PATH)
    if config.get("torch_compile_disable_triton", False):
        return False
    return True


def is_torch_compile_available():
    """Check if torch.compile can be used.

    Returns True if:
    - torch.compile exists
    - CUDA is available

    Note: triton is optional - torch.compile works without it (with fallback).
    """
    return (
        hasattr(torch, 'compile') and
        torch.cuda.is_available()
    )


def setup_torch_compile_cache():
    """Enable torch.compile cache for faster startup on subsequent runs (PyTorch 2.4+).

    Sets up persistent caching so that compiled models are reused across application restarts.
    """
    # Set up persistent cache directory in the current working directory
    cache_dir = os.path.join(os.getcwd(), ".torch_compile_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Set environment variables for PyTorch inductor caching
    # These must be set BEFORE any torch.compile calls
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

    # Configure inductor settings programmatically if available
    if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
        torch._inductor.config.fx_graph_cache = True
        torch._inductor.config.autotune_local_cache = True
        # Set cache directory for autotune results if available
        if hasattr(torch._inductor.config, "autotune_local_cache_dir"):
            torch._inductor.config.autotune_local_cache_dir = cache_dir


def apply_triton_settings():
    """Apply triton enable/disable settings via environment variable.

    Must be called before any torch.compile calls to take effect.
    """
    config = load_config(CONFIG_PATH)
    disable_triton = config.get("torch_compile_disable_triton", False)

    if disable_triton:
        # Set environment variable to disable triton
        os.environ["TORCHINDUCTOR_DISABLE_TRITON"] = "1"
    else:
        # Remove the env var if it was previously set
        os.environ.pop("TORCHINDUCTOR_DISABLE_TRITON", None)


def reset_torchcrepe_compiled_model():
    """Reset the cached compiled model in torchcrepe.

    This should be called when TorchCompile settings change to avoid conflicts
    with the old compiled model.
    """
    try:
        import torchcrepe
        if hasattr(torchcrepe, 'core') and hasattr(torchcrepe.core, 'infer'):
            if hasattr(torchcrepe.core.infer, 'model'):
                torchcrepe.core.infer.model = None
    except ImportError:
        pass

    # Also reset torch dynamo cache to ensure clean recompilation
    try:
        torch._dynamo.reset()
    except Exception:
        pass


def load_torch_compile_enabled():
    """Load torch compile enabled state from config."""
    config = load_config(CONFIG_PATH)
    return bool(config.get("torch_compile_enabled", False))


def load_torch_compile_mode():
    """Load torch compile mode from config."""
    config = load_config(CONFIG_PATH)
    return config.get("torch_compile_mode", "default")


def load_torch_compile_disable_triton():
    """Load torch compile disable triton state from config."""
    config = load_config(CONFIG_PATH)
    return bool(config.get("torch_compile_disable_triton", False))


def save_torch_compile_enabled(enabled: bool):
    """Save torch compile enabled state to config."""
    update_config(CONFIG_PATH, {"torch_compile_enabled": bool(enabled)})
    if enabled:
        setup_torch_compile_cache()
        apply_triton_settings()
    # Reset compiled model when settings change
    reset_torchcrepe_compiled_model()


def save_torch_compile_mode(mode: str):
    """Save torch compile mode to config."""
    if mode not in TORCH_COMPILE_MODES:
        mode = "default"
    update_config(CONFIG_PATH, {"torch_compile_mode": mode})
    # Reset compiled model when mode changes
    reset_torchcrepe_compiled_model()


def save_torch_compile_disable_triton(disabled: bool):
    """Save torch compile disable triton state to config."""
    update_config(CONFIG_PATH, {"torch_compile_disable_triton": bool(disabled)})
    apply_triton_settings()
    # Reset compiled model when triton setting changes
    reset_torchcrepe_compiled_model()


def get_torch_compile_settings():
    """Get torch compile settings for use in inference.

    Returns (enabled, mode) tuple. The 'enabled' value will be False if:
    - User has disabled it in config
    - torch.compile is not available (no CUDA, etc.)

    Also applies triton settings and checks if settings have changed.
    """
    global _last_compile_settings

    config = load_config(CONFIG_PATH)
    enabled = bool(config.get("torch_compile_enabled", False))
    mode = config.get("torch_compile_mode", "default")
    disable_triton = bool(config.get("torch_compile_disable_triton", False))

    if mode not in TORCH_COMPILE_MODES:
        mode = "default"

    # Only actually enable if torch.compile is available
    if enabled and not is_torch_compile_available():
        enabled = False

    # Check if settings have changed
    current_settings = (enabled, mode, disable_triton)
    if _last_compile_settings is not None and _last_compile_settings != current_settings:
        # Settings changed, reset compiled model
        reset_torchcrepe_compiled_model()
    _last_compile_settings = current_settings

    # Apply triton settings
    if enabled:
        apply_triton_settings()

    return enabled, mode


# Initialize cache and triton settings if torch compile is enabled at startup
if load_torch_compile_enabled():
    setup_torch_compile_cache()
    apply_triton_settings()
