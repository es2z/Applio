# Codebase Structure

## Root Directory Layout

```
Applio/
├── app.py                    # Gradio web interface entry point
├── core.py                   # CLI interface with subcommands
├── requirements.txt          # Python dependencies
├── CLAUDE.md                # Project guidance for Claude Code
├── README.md                # Project documentation
├── LICENSE                  # MIT License
├── TERMS_OF_USE.md         # Usage terms
├── Makefile                # Linux/macOS commands
├── docker-compose.yaml     # Docker configuration
├── Dockerfile              # Docker build
├── run-*.bat               # Windows batch scripts
├── run-*.sh                # Linux/macOS shell scripts
├── .gitignore             # Git ignore patterns
├── assets/                # Static assets and configuration
├── rvc/                   # Main RVC library
├── tabs/                  # Gradio UI tabs
├── env/                   # Python virtual environment (created by installer)
├── logs/                  # Training outputs and user models (gitignored)
└── templates/             # Real-time conversion templates (gitignored)
```

## Main Modules

### `rvc/` - Core RVC Library

```
rvc/
├── configs/
│   ├── config.py          # Config singleton class
│   ├── 32000.json        # 32kHz model config
│   ├── 40000.json        # 40kHz model config
│   └── 48000.json        # 48kHz model config
├── infer/
│   ├── infer.py          # VoiceConverter class (main inference)
│   └── pipeline.py       # VC_Pipeline (RVC algorithm implementation)
├── realtime/
│   ├── core.py           # Real-time conversion entry point
│   ├── pipeline.py       # Realtime_Pipeline class
│   ├── audio.py          # Audio I/O handling
│   ├── callbacks.py      # UI callbacks
│   └── utils/            # Real-time utilities
│       ├── torch.py
│       └── vad.py
├── train/
│   ├── train.py          # Training entry point
│   ├── data_utils.py     # Dataset handling
│   ├── losses.py         # Loss functions
│   ├── mel_processing.py # Mel spectrogram processing
│   ├── utils.py          # Training utilities
│   ├── preprocess/
│   │   ├── preprocess.py # Audio preprocessing
│   │   └── slicer.py     # Audio slicing
│   ├── extract/
│   │   ├── extract.py    # Feature extraction
│   │   └── preparing_files.py
│   └── process/
│       ├── extract_index.py      # FAISS index generation
│       ├── extract_model.py      # Model extraction
│       ├── model_blender.py      # Model fusion
│       ├── model_information.py  # Model info display
│       └── change_info.py        # Model metadata editing
└── lib/
    ├── algorithm/
    │   ├── synthesizers.py        # Synthesizer wrapper
    │   ├── encoders.py            # Text/audio encoders
    │   ├── attentions.py          # Attention mechanisms
    │   ├── residuals.py           # Residual blocks
    │   ├── discriminators.py      # GAN discriminators
    │   ├── commons.py             # Common utilities
    │   ├── modules.py             # Neural network modules
    │   ├── normalization.py       # Normalization layers
    │   └── generators/
    │       ├── hifigan.py         # HiFi-GAN vocoder
    │       ├── hifigan_mrf.py     # Multi-receptive field HiFi-GAN
    │       ├── hifigan_nsf.py     # NSF variant
    │       └── refinegan.py       # RefineGAN vocoder
    ├── predictors/
    │   ├── f0.py                  # F0 predictor factory
    │   ├── F0Extractor.py         # Base F0 extractor
    │   ├── RMVPE.py               # RMVPE pitch extraction
    │   ├── FCPE.py                # FCPE pitch extraction
    │   └── onnxcrepe/             # CREPE (ONNX) pitch extraction
    │       ├── core.py
    │       ├── decode.py
    │       ├── filter.py
    │       ├── load.py
    │       └── ...
    ├── tools/
    │   ├── prerequisites_download.py  # Download pretrained models
    │   ├── model_download.py          # Download models from URLs
    │   ├── tts.py                     # Text-to-speech
    │   ├── tts_voices.json            # TTS voice list
    │   ├── split_audio.py             # Audio splitting
    │   ├── analyzer.py                # Audio analysis
    │   ├── launch_tensorboard.py      # TensorBoard launcher
    │   ├── pretrained_selector.py     # Select pretrained models
    │   └── gdown.py                   # Google Drive downloader
    ├── utils.py                       # General utilities
    └── zluda.py                       # AMD GPU support (ZLUDA hijack)
```

### `tabs/` - Gradio UI Tabs

```
tabs/
├── inference/         # Voice conversion interface
├── train/            # Model training interface
├── tts/              # Text-to-speech interface
├── voice_blender/    # Model blending interface
├── realtime/         # Real-time conversion interface
├── plugins/          # Plugin management
├── settings/         # Application settings
├── download/         # Model download
├── report/           # Bug reporting
└── extra/            # Extra utilities
```

Each tab directory typically contains:
- `{name}.py` - Main tab logic
- UI component definitions
- Callback functions

### `assets/` - Static Assets

```
assets/
├── config.json              # User configuration (gitignored)
├── ICON.ico                # Application icon
├── i18n/                   # Internationalization
│   ├── i18n.py            # I18n loader
│   └── locale/            # Translation files (en_US, ja_JP, etc.)
├── themes/                 # Gradio themes
│   ├── loadThemes.py
│   └── {theme_name}.py
├── audios/                 # User audio files (gitignored)
├── datasets/               # Training datasets (gitignored)
├── installation_checker.py # Verify installation
├── discord_presence.py     # Discord RPC
└── zluda/                  # ZLUDA for AMD GPUs
```

## Key File Locations

### Configuration Files
- **User config**: `assets/config.json` - Theme, language, precision, realtime settings
- **Model configs**: `rvc/configs/{32000,40000,48000}.json` - Sample rate specific
- **Version configs**: JSON files for different model versions

### Pretrained Models
- **Location**: `rvc/models/` (downloaded on first run)
- **Types**: Pretrained generators, discriminators, embedder models

### Training Outputs
- **Location**: `logs/{model_name}/`
- **Contents**: `.pth` checkpoints, `.index` files, training logs

### Entry Points
- **Web UI**: `app.py`
- **CLI**: `core.py`
- **Training**: `rvc/train/train.py` (called via CLI)
- **Real-time**: `rvc/realtime/core.py` (called via UI)

## Data Flow

### Inference
```
User Input (audio)
  ↓
VoiceConverter.convert_audio() [rvc/infer/infer.py]
  ↓
Load model & index
  ↓
VC_Pipeline [rvc/infer/pipeline.py]
  ├─ Extract embeddings (Hubert/ContentVec)
  ├─ F0 extraction (RMVPE/FCPE/CREPE)
  ├─ FAISS index search
  └─ Synthesizer (HiFi-GAN/RefineGAN)
  ↓
Post-processing (optional)
  ↓
Output (converted audio)
```

### Training
```
Dataset (raw audio)
  ↓
Preprocess [rvc/train/preprocess/]
  ↓
Extract features [rvc/train/extract/]
  ├─ F0 curves
  └─ Speaker embeddings
  ↓
Train [rvc/train/train.py]
  ├─ Generator (synthesizer)
  └─ Discriminator
  ↓
Extract model [rvc/train/process/extract_model.py]
  ↓
Generate index [rvc/train/process/extract_index.py]
  ↓
logs/{model_name}/
  ├─ {model}.pth
  └─ {model}.index
```

### Real-time Conversion
```
Audio Input Device
  ↓
Realtime_Pipeline [rvc/realtime/pipeline.py]
  ├─ Circular buffer
  ├─ VAD (Voice Activity Detection)
  ├─ VC_Pipeline (inference)
  └─ Latency optimization
  ↓
Audio Output Device
```

## Module Dependencies

- **Core dependencies**: PyTorch, Gradio, librosa, soundfile
- **Inference**: FAISS, torchcrepe/torchfcpe, transformers
- **Training**: TensorBoard, distributed training utilities
- **Real-time**: sounddevice, webrtcvad
- **Post-processing**: pedalboard, noisereduce

## Important Notes

- **Modular design**: Each component (inference, training, real-time) is relatively independent
- **Plugin architecture**: Extensible through `tabs/plugins/`
- **i18n support**: All UI strings can be translated
- **Config-driven**: User settings and model configs separate from code
- **Cross-platform**: Batch files for Windows, shell scripts for Linux/macOS
