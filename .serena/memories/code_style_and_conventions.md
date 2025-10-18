# Code Style and Conventions

## General Observations

The codebase does **not have formal linting or formatting configurations** (no pyproject.toml, .flake8, .black.toml, etc.). However, the code follows certain patterns:

## Python Style

### Type Hints
- **Partial usage**: Some functions use type hints, but not consistently throughout
- Examples:
  ```python
  def get_value_from_args(key: str, default: Any = None) -> Any:
  def launch_gradio(server_name: str, server_port: int) -> None:
  ```
- Imports `from typing import Any` when needed

### Docstrings
- **Minimal to none**: Most functions lack docstrings
- Code is generally self-documenting through clear naming

### Naming Conventions
- **Functions/methods**: `snake_case`
  - `convert_audio()`, `load_hubert()`, `post_process_audio()`
- **Classes**: `PascalCase`
  - `VoiceConverter`, `Config`, `Realtime_Pipeline`
- **Variables**: `snake_case`
  - `model_name`, `sample_rate`, `gpu_mem`
- **Constants**: `UPPER_SNAKE_CASE`
  - `DEFAULT_SERVER_NAME`, `DEFAULT_PORT`, `MAX_PORT_ATTEMPTS`
- **Private attributes**: Not strictly enforced, occasional single underscore

### Imports
- Grouped by category (standard library, third-party, local)
- Absolute imports from project root: `from rvc.configs.config import Config`
- Relative imports within modules: `from .utils import ...`
- Common pattern: Adding `now_dir = os.getcwd()` and `sys.path.append(now_dir)`

## Code Organization

### File Structure
- **Entry points**: `app.py` (Gradio UI), `core.py` (CLI)
- **Main modules**: Under `rvc/` directory
  - `rvc/infer/` - Inference logic
  - `rvc/train/` - Training pipeline
  - `rvc/realtime/` - Real-time conversion
  - `rvc/lib/` - Shared utilities, algorithms, predictors
  - `rvc/configs/` - Configuration management
- **UI tabs**: Under `tabs/` directory (one module per tab)
- **Assets**: `assets/` for configs, themes, i18n, models

### Design Patterns

#### Singleton Pattern
Used for Config class:
```python
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Config:
    ...
```

#### Caching
- `@lru_cache` decorator used for expensive operations:
  ```python
  @lru_cache(maxsize=1)
  def load_voices_data():
      ...
  
  @lru_cache(maxsize=None)
  def import_voice_converter():
      ...
  ```

#### Class-based Components
- Major functionality encapsulated in classes:
  - `VoiceConverter` - Main inference class
  - `Config` - Configuration management
  - `Realtime_Pipeline` - Real-time processing
  - `Synthesizer` - Neural vocoder wrapper

### Error Handling
- Generally basic, not extensive try-except blocks
- Relies on natural exception propagation
- Some validation at function entry

### Comments
- Minimal inline comments
- Section headers used for grouping:
  ```python
  # Set up logging
  # Import Tabs
  # Run prerequisites
  # Initialize i18n
  ```

## PyTorch Specific

### Device Management
- Central device config through `Config` class
- Dynamic GPU memory detection
- CUDA availability checking: `torch.cuda.is_available()`

### Model Loading
- Security-conscious: `torch.load(..., weights_only=True)` 
- Model caching to avoid reloading

### Training
- Uses PyTorch distributed training (DDP)
- TensorBoard integration for monitoring
- Checkpointing system

## Gradio UI

### Tab Organization
- Each tab is a separate function returning Gradio components
- Import pattern: `from tabs.{name}.{name} import {name}_tab`
- Internationalization: All UI strings wrapped in `i18n()`

### Configuration
- User settings stored in `assets/config.json`
- Dynamic theme loading from `assets/themes/`

## Windows-Specific

### Batch Scripts
- `.bat` files for Windows execution
- Error checking: `if errorlevel 1 goto :error`
- Admin privilege check: `if /i "%cd%"=="C:\Windows\System32"`
- Environment activation via conda

### Path Handling
- Backslash separators in Windows
- Uses `os.path.join()` for cross-platform compatibility

## Best Practices Observed

1. **Don't repeat yourself**: Shared utilities in `rvc/lib/utils.py`
2. **Separation of concerns**: UI, inference, training in separate modules
3. **Configuration over hardcoding**: JSON configs for user settings
4. **Progressive enhancement**: Optional features (Discord presence, ZLUDA support)
5. **CLI + GUI**: Both interfaces for different use cases

## What to Follow When Contributing

- Use `snake_case` for functions/variables, `PascalCase` for classes
- Keep imports organized by category
- Add `now_dir` and `sys.path.append()` in entry-point scripts
- Use `Config` singleton for device/GPU configuration
- Cache expensive operations with `@lru_cache`
- Security: use `weights_only=True` for `torch.load()`
- Wrap UI strings with `i18n()` for internationalization
- Store user config in `assets/config.json`
