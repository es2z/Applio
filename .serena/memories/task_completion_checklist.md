# Task Completion Checklist

When you complete a coding task in this project, follow these guidelines:

## Testing

**There is NO formal test suite in this project.**

Testing is done through:
1. **Gradio UI**: Run `run-applio.bat` and test changes through the web interface
2. **CLI commands**: Use `core.py` subcommands to test specific functionality
3. **Manual verification**: Test the specific feature you modified

### Testing Inference Changes
```bash
# Test through CLI
env\python.exe core.py infer --input_path test.wav --output_path out.wav --pth_path model.pth --index_path model.index

# Or launch UI and test through Inference tab
run-applio.bat
```

### Testing Training Changes
```bash
# Run training pipeline
env\python.exe core.py preprocess --model_name test --dataset_path dataset
env\python.exe core.py extract --model_name test --sample_rate 40000 --f0_method rmvpe
env\python.exe core.py train --model_name test --sample_rate 40000 --total_epoch 10
```

### Testing Real-time Changes
- Launch UI with `run-applio.bat`
- Go to Realtime tab
- Configure devices and test conversion

## Linting & Formatting

**There are NO linting or formatting tools configured in this project.**

No configurations exist for:
- Black
- Flake8
- Pylint
- isort
- mypy
- pyproject.toml (no tool configuration section)

However, you should still:
1. Follow the code style conventions (see `code_style_and_conventions.md`)
2. Use consistent naming: `snake_case` for functions/variables, `PascalCase` for classes
3. Add type hints where appropriate (though not required)
4. Keep code readable and well-organized

## Code Quality Checks

### Manual Review Checklist
- [ ] Code follows existing naming conventions
- [ ] Imports are organized (stdlib, third-party, local)
- [ ] No hardcoded paths (use `os.path.join()`)
- [ ] Device handling uses `Config` class
- [ ] Model loading uses `weights_only=True`
- [ ] UI strings wrapped in `i18n()` for internationalization
- [ ] User settings saved to `assets/config.json` if needed
- [ ] No unnecessary comments (code should be self-documenting)
- [ ] Error handling appropriate for the context

## Documentation

**Do NOT create extensive documentation unless requested.**

- Do NOT create docstrings unless the function is complex
- Do NOT create README updates unless explicitly asked
- Do NOT create API documentation
- Code should be self-documenting through clear naming

If documentation is needed:
- Update CLAUDE.md if architecture changed significantly
- Add comments only for non-obvious logic

## Git Workflow

Standard git workflow (if committing changes):
```bash
git status
git add .
git commit -m "Descriptive message"
git push
```

**Note**: Most users of this fork may not be pushing changes upstream.

## Performance Considerations

If you modified performance-critical code:
- [ ] Consider using `@lru_cache` for expensive operations
- [ ] Check GPU memory usage if working with models
- [ ] Profile real-time code for latency (should be <100ms for real-time)
- [ ] Verify batch processing doesn't OOM on typical GPUs

## Configuration Changes

If you modified `assets/config.json` structure:
- [ ] Ensure backward compatibility with existing configs
- [ ] Update settings tab UI if needed
- [ ] Document new config options in CLAUDE.md

## Integration Points

If you modified:

### Inference Code (`rvc/infer/`)
- [ ] Test through both UI (Inference tab) and CLI (`core.py infer`)
- [ ] Verify batch inference still works
- [ ] Check post-processing effects if modified

### Training Code (`rvc/train/`)
- [ ] Run a short training test (10 epochs)
- [ ] Verify TensorBoard logging works
- [ ] Check checkpoint saving/loading

### Real-time Code (`rvc/realtime/`)
- [ ] Test latency with real audio devices
- [ ] Verify template save/load if modified
- [ ] Check CPU/GPU usage

### UI Tabs (`tabs/`)
- [ ] Test Gradio interface loads without errors
- [ ] Check internationalization (switch languages)
- [ ] Verify callbacks work correctly

## Platform-Specific Testing

### Windows-Specific
- [ ] Batch files work correctly if modified
- [ ] Paths use backslashes or `os.path.join()`
- [ ] Test with Windows WASAPI/WDM-KS audio if relevant

### Cross-Platform
- [ ] Use `os.path.join()` instead of hardcoded separators
- [ ] Check platform-specific code (e.g., `sys.platform == "win32"`)

## Final Steps

1. **Run the application**: `run-applio.bat`
2. **Test the specific feature** you modified
3. **Verify no errors** in console output
4. **Check functionality** matches requirements

## What NOT to Do

- ❌ Do not run automated tests (none exist)
- ❌ Do not run linters (none configured)
- ❌ Do not run formatters (none configured)
- ❌ Do not create extensive documentation
- ❌ Do not create unit tests unless specifically requested
- ❌ Do not add type hints everywhere (only where helpful)
- ❌ Do not add docstrings to simple functions
