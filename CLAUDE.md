# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Applio is a high-quality voice conversion tool built on Retrieval-Based Voice Conversion (RVC) technology. It provides a Gradio-based web interface for training voice models and performing voice conversion with various audio processing features.

**Key Technology Stack:**
- PyTorch 2.7.1 with CUDA 12.8 support
- Gradio 5.23.1 for web UI
- Multiple F0 (pitch) extraction methods: RMVPE, FCPE, CREPE, SWIFT
- Various embedder models: ContentVec, SPIN, Hubert variants
- Audio processing: librosa, soundfile, pedalboard, noisereduce
- Real-time voice conversion support

## Development Commands

### Environment Setup
```bash
# Windows installation (first time)
run-install.bat

# Launch Applio web interface
run-applio.bat
# Or with CLI arguments:
env\python.exe app.py --open --port 6969 --server-name 127.0.0.1

# Launch TensorBoard for training monitoring
run-tensorboard.bat
```

### CLI Operations via core.py
The `core.py` file provides CLI access to all Applio functionality:

```bash
# Voice inference (single file)
python core.py infer --input_path <audio> --output_path <output> --pth_path <model.pth> --index_path <model.index>

# Batch inference
python core.py batch_infer --input_folder <dir> --output_folder <dir> --pth_path <model.pth> --index_path <model.index>

# TTS with voice conversion
python core.py tts --tts_text "Hello" --tts_voice <voice> --output_rvc_path <output> --pth_path <model.pth> --index_path <model.index>

# Training pipeline
python core.py preprocess --model_name <name> --dataset_path <dir> --sample_rate 40000
python core.py extract --model_name <name> --sample_rate 40000 --f0_method rmvpe --embedder_model contentvec
python core.py train --model_name <name> --sample_rate 40000 --total_epoch 500 --save_every_epoch 50

# Utilities
python core.py model_information --pth_path <model.pth>
python core.py model_blender --model_name <name> --pth_path_1 <model1.pth> --pth_path_2 <model2.pth> --ratio 0.5
python core.py audio_analyzer --input_path <audio>
```

### Running Tests
This project does not include a formal test suite. Testing is done through the Gradio interface or CLI commands.

## Architecture

### Core Components

**1. Entry Points**
- `app.py` - Main Gradio web application launcher
- `core.py` - CLI interface exposing all functionality as subcommands

**2. Voice Conversion Engine** (`rvc/`)
- `rvc/infer/infer.py` - Main VoiceConverter class for inference
  - Handles model loading, audio conversion, batch processing
  - Post-processing effects (reverb, pitch shift, compression, etc.)
  - Audio cleaning and format conversion
- `rvc/infer/pipeline.py` - VC pipeline implementing the RVC algorithm
  - F0 extraction and processing
  - Speaker embedding retrieval via FAISS index
  - Audio synthesis through the generator network

**3. Real-time Voice Conversion** (`rvc/realtime/`)
- `rvc/realtime/pipeline.py` - Optimized pipeline for low-latency conversion
  - Uses circular buffers for streaming audio
  - Supports WASAPI/WDM-KS audio backends
  - Recent commits mention latency improvements with WDM-KS output

**4. Neural Network Models** (`rvc/lib/algorithm/`)
- `synthesizers.py` - Main Synthesizer class wrapping generators
- `generators/` - Multiple vocoder implementations:
  - `hifigan.py` - Original HiFi-GAN
  - `hifigan_mrf.py` - Multi-receptive field variant
  - `refinegan.py` - RefineGAN vocoder
- `encoders.py`, `attentions.py`, `residuals.py` - Network components

**5. F0 (Pitch) Predictors** (`rvc/lib/predictors/`)
- Multiple pitch extraction algorithms:
  - `RMVPE` - Default, most robust
  - `FCPE` - Fast and accurate
  - `CREPE` - High quality but slower
  - `SWIFT` - Fast inference
  - Hybrid modes combining multiple methods

**6. Training Pipeline** (`rvc/train/`)
- `preprocess/preprocess.py` - Audio preprocessing (chunking, filtering)
- `extract/extract.py` - Feature extraction (F0 + embeddings)
- `train.py` - Model training with overtraining detection
- `process/extract_index.py` - FAISS index generation for speaker retrieval

**7. Web Interface** (`tabs/`)
- Modular Gradio tabs: inference, train, tts, voice_blender, realtime, plugins, settings
- Each tab is self-contained with its own UI and callbacks

**8. Configuration & Utilities**
- `rvc/configs/config.py` - Central Config class (device selection, paths)
- `assets/config.json` - User settings (theme, language, precision, realtime config)
- `rvc/lib/utils.py` - Audio loading, embedding loading utilities
- `assets/i18n/` - Internationalization support

### Data Flow: Inference

1. **Load model** - VoiceConverter loads .pth checkpoint and index file
2. **Load audio** - Input audio resampled to 16kHz
3. **Extract embeddings** - Hubert/ContentVec extracts features
4. **Retrieve speaker** - FAISS index finds similar training embeddings
5. **Extract F0** - Pitch contour extracted using selected method
6. **Synthesize** - Generator network produces converted audio
7. **Post-process** - Optional effects (reverb, autotune, noise reduction)
8. **Export** - Save as WAV/MP3/FLAC/OGG/M4A

### Data Flow: Training

1. **Preprocess** - Audio split into chunks, optional filters applied
2. **Extract** - F0 curves and speaker embeddings extracted
3. **Train** - Generator and discriminator trained on processed data
4. **Index** - FAISS index built from training embeddings
5. **Export** - Model checkpoint (.pth) and index (.index) saved to `logs/<model_name>/`

### Directory Structure

```
Applio-3.5.0/
├── app.py                    # Gradio web app entry point
├── core.py                   # CLI interface
├── requirements.txt          # Python dependencies
├── assets/
│   ├── config.json          # User configuration (edited by Settings tab)
│   ├── i18n/                # Translation files
│   └── themes/              # Custom Gradio themes
├── rvc/
│   ├── configs/config.py    # Config class
│   ├── infer/              # Voice conversion inference
│   ├── realtime/           # Real-time conversion
│   ├── train/              # Training pipeline
│   ├── lib/
│   │   ├── algorithm/      # Neural network architectures
│   │   ├── predictors/     # F0 extraction models
│   │   └── tools/          # Utilities (download, TTS, analysis)
│   └── models/             # Pretrained models (downloaded on first run)
├── tabs/                    # Gradio UI tabs
│   ├── inference/
│   ├── train/
│   ├── realtime/
│   ├── tts/
│   └── settings/
└── logs/                    # Training outputs and user models
    └── <model_name>/
        ├── <model>.pth
        └── <model>.index
```

## Important Development Notes

### Model Loading
- Models are loaded with `torch.load(..., weights_only=True)` for security
- The VoiceConverter caches loaded models - only reloads if path changes
- Multiple vocoder types supported: HiFi-GAN (default), MRF HiFi-GAN, RefineGAN

### Real-time Mode Configuration
- Real-time settings stored in `assets/config.json` under `realtime` key
- Recent work focused on WASAPI input → WDM-KS output for lower latency
- Monitor device can be empty (no monitoring) or set for audio passthrough

### Audio Processing Pipeline
- All audio internally processed at 16kHz for feature extraction
- Output resampled to model's target SR (32kHz, 40kHz, or 48kHz)
- Post-processing effects applied via Pedalboard at output sample rate

### F0 Methods
- `rmvpe` - Default, best balance of speed/quality
- `fcpe` - Fastest, good for real-time
- `crepe` - Highest quality, slowest
- `hybrid[...]` - Averages multiple methods for robustness

### Embedder Models
- `contentvec` - Default, works for most languages
- `spin`, `spin-v2` - Alternative embedders
- `chinese-hubert-base`, `japanese-hubert-base`, `korean-hubert-base` - Language-specific
- `japanese-hubert-base-k2` - Japanese, `reazon-research/japanese-hubert-base-k2` (fork-specific, see below)
- `custom` - Use custom embedder (provide path via `embedder_model_custom`)

### Index Files
- Generated from training embeddings using FAISS
- Used during inference for speaker similarity retrieval
- Higher `index_rate` (0-1) = stronger model influence, may introduce artifacts
- Lower `index_rate` = more original voice characteristics preserved

### Training Best Practices
- Sample rates: 40kHz recommended for most uses, 48kHz for high quality
- Batch size: 8-16 depending on GPU memory
- Enable `overtraining_detector` to auto-stop when validation loss increases
- Use pretrained models unless you have a large dataset (>30 minutes)

### Plugin System
- Plugins can be added via `tabs/plugins/` directory
- Plugin registry stored in `assets/config.json`
- See Applio documentation for plugin development guide

### ZLUDA Support (AMD GPUs)
- AMD GPU support via ZLUDA in `assets/zluda/`
- Run with `run-applio-amd.bat` for AMD acceleration
- Requires patching based on HIP version (5.7, 6.1, or 6.2)

## Fork-Specific Features

This is a personal fork with the following customizations:

### Python 3.13 + Torch 2.8 Support
- Upgraded from Python 3.11/3.12 + Torch 2.7.1 to Python 3.13 + Torch 2.8
- Installation script: `run-install-py313.bat` for Python 3.13 environment
- Note: mangio-crepe implementation may differ slightly from upstream

### Training Tab Enhancements
- Added `mangio-crepe` as a pitch adjustment algorithm option

### Additional Embedder: `japanese-hubert-base-k2`
- `reazon-research/japanese-hubert-base-k2`, a HuBERT Base trained on ReazonSpeech v2.0
- 768-dim / 12 layers / 320-sample stride, so RVC v2 G/D, `text_enc_hidden_dim` and the
  FAISS index dimension are unchanged and existing 768-dim checkpoints stay loadable
- Downloaded through `transformers` (safetensors only, no `pytorch_model.bin`) and cached
  under `rvc/models/embedders/japanese_hubert_base_k2/`. The commit SHA is pinned in
  `JAPANESE_HUBERT_BASE_K2_REVISION`, so an upstream update cannot silently swap the
  weights under an already-trained model and a cached load needs no network call
- **Its hidden states are ~10x smaller than every other embedder's**, because its final
  LayerNorm gain is that much smaller: 0.64 per frame against 6.49 for
  `japanese-hubert-base` and 9.31 for `contentvec`. `TextEncoder` adds
  `emb_phone(feature)` straight onto a scale-free `emb_pitch` embedding
  (`rvc/lib/algorithm/encoders.py:131-133`), so raw k2 features leave the content term
  ~5.7x under-weighted against pitch, and `emb_phone` never catches up because its
  gradient scales with the input magnitude too (measured: its weight norm moved 23.8 ->
  25.5 over 240 epochs, against the ~10x needed). The symptom is a voice that cuts out
  mid-speech and never improves with more training. `EMBEDDER_FEATURE_SCALE`
  (`rvc/lib/utils.py`) multiplies k2's hidden states by 10.0 at the three embedder call
  sites via `apply_embedder_feature_scale`, landing them at 6.44 per frame. Every other
  embedder carries `feature_scale = 1.0` and is handed back untouched.
- **Unlike every other embedder, its official `preprocessor_config.json` sets
  `do_normalize: true`.** `load_embedding` records that flag on the model and
  `apply_embedder_input_normalization` (`rvc/lib/utils.py`) applies the equivalent
  zero-mean / unit-variance step at the three embedder call sites (training extraction,
  offline inference, realtime). All other embedders keep `input_do_normalize = False`
  and are fed the raw waveform exactly as before. Note this is close to a no-op for every
  embedder here: they are all `feat_extract_norm: "group"` with `conv_bias: false`, so the
  GroupNorm after the first bias-free conv already cancels any scalar gain (measured:
  under 0.5% feature change). It is kept for fidelity to the official config and for a
  future `feat_extract_norm: "layer"` embedder, which would genuinely need it.
- Changing either the embedder **or its feature scale** on an existing model folder
  re-extracts every feature and deletes the stale `.index`
  (`resolve_feature_reuse` in `rvc/train/extract/extract.py`), so features and index are
  never mixed. A folder that recorded an embedder but no `embedder_feature_scale` predates
  scaling, which is exactly a scale of 1.0; a folder that recorded nothing is left alone.

### Known: lost positional-conv weights on the legacy embedders
`contentvec`, `japanese-hubert-base` and the other `pytorch_model.bin` embedders were saved
by transformers <=4.30 with `pos_conv_embed.conv.weight_g/weight_v`, which the pinned
transformers 4.44.2 no longer recognises (it wants
`pos_conv_embed.conv.parametrizations.weight.original0/1`). Those weights are therefore
silently dropped and re-initialised on every load - the warning is hidden by the
`warnings.filterwarnings("ignore")` and transformers log level in `rvc/lib/utils.py`. The
positional conv contributes `|pos|/|h|` of 0.46 for them against 3.19 for k2, which is a
large part of why k2's features look so different. **Do not "fix" this**: the fallback is
deterministic, every existing model was trained and is used against the same features, and
loading the real weights would invalidate every trained checkpoint and index in `logs/`.
Check it with `from_pretrained(..., output_loading_info=True)`.

### Realtime Tab Enhancements
- **Template System**: Save/load device connections, model settings, and parameter values
- **WDM-KS Support**: Can use WDM-KS audio API for output
  - Enables mixed API usage (e.g., WASAPI input → WDM-KS output)
  - Improves latency in certain configurations
- **Extended F0 Methods**:
  - CREPE variants: `crepe-tiny`, `crepe-full`
  - Mangio-CREPE variants: `mangio-crepe-tiny`, `mangio-crepe-full`
  - Hybrid support infrastructure is preserved for future enhancements

## Common Pitfalls

1. **Missing prerequisites** - Run `run-install.bat` before first use
2. **Index path errors** - Ensure .index file matches the .pth model
3. **F0 extraction failures** - Try different f0_method if one fails
4. **GPU memory issues** - Reduce batch_size or use `cache_data_in_gpu=False`
5. **Audio quality problems** - Adjust `protect`, `index_rate`, and `clean_audio` settings
6. **Real-time latency** - Use FCPE or SWIFT f0_method, optimize buffer sizes

## External Resources

- Documentation: https://docs.applio.org
- Discord Support: https://discord.gg/urxFjYmYYh
- Plugin Repository: https://github.com/IAHispano/Applio-Plugins
- Compiled Versions: https://huggingface.co/IAHispano/Applio/tree/main/Compiled
