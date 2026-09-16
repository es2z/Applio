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
- `japanese-hubert-large` - Japanese, 1024-dim / 24 layers (fork-specific, see below)
- `kushinada-hubert-large` - Japanese, 1024-dim / 24 layers, gated so installed by hand
  (fork-specific, see below)
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

### Not a bug: the `weight_g`/`weight_v` loading warning
`contentvec`, `japanese-hubert-base` and the other `pytorch_model.bin` embedders were saved
by transformers <=4.30 with `pos_conv_embed.conv.weight_g/weight_v`, while the pinned
transformers 4.44.2 stores that layer as
`pos_conv_embed.conv.parametrizations.weight.original0/1`. `from_pretrained(...,
output_loading_info=True)` reports the old names as `unexpected_keys` and the new ones as
`missing_keys`, which reads like the positional conv is being dropped and re-initialised.
**It is not.** transformers renames those keys while loading; the loading-info lists are
bookkeeping left over from the rename. Verified by comparing the loaded
`parametrizations.weight.original0/1` against the raw checkpoint's `weight_g`/`weight_v`:
bit-identical for both `contentvec` and `japanese-hubert-base`. The same warning appears for
`japanese-hubert-large` and is equally harmless. Do not add a remapping shim for it.

### Measured embedder characteristics
Measured on `logs/reference/reference.wav` (34.9 s), 1742 frames for every embedder:

| embedder | dim | layers | norm/frame | `\|pos\|/\|h\|` | do_normalize effect |
|---|---|---|---|---|---|
| contentvec | 768 | 12 | 9.82 | 0.94 | 0.29% |
| japanese-hubert-base | 768 | 12 | 6.35 | 0.80 | 4.99% |
| japanese-hubert-base-k2 | 768 | 12 | **0.58** | **7.30** | 0.21% |
| japanese-hubert-large | 1024 | 24 | 5.95 | 0.90 | **59.25%** |
| kushinada-hubert-large | 1024 | 24 | 8.14 | 1.36 | **49.80%** |

Two things this table settles:
- **k2 is the outlier, and its `feature_scale = 10.0` only fixes half of it.** Its hidden
  states are ~10x smaller than everyone else's *and* its positional conv is ~8x more
  dominant (`|pos|/|h| = 7.3` against ~0.9 for every other embedder, from a pos_conv weight
  norm of 33.8 vs ~16). Scaling the features up fixes the magnitude against `emb_pitch` but
  cannot change the ratio of positional to content information inside them. If a k2 model
  sounds noisy, that ratio is the first thing to suspect, not a missing parameter.
- **`japanese-hubert-large` is unremarkable on every axis except `do_normalize`.** Its
  magnitude (5.95) sits next to `japanese-hubert-base` (6.35), so it carries
  `feature_scale = 1.0`. But it is the first `feat_extract_norm: "layer"` /
  `conv_bias: true` embedder here, so the waveform normalisation is not optional for it:
  skipping it changes the features by 59%, against under 5% for every `"group"` embedder.
  The same applies to `kushinada-hubert-large`, which shares that architecture (49.80%).

### Adding another embedder
Read `docs/ADDING_AN_EMBEDDER_MODEL.md` first. It carries the measured characteristics of
every embedder here, the landmines that cost time (input normalisation, feature magnitude,
stale resume checkpoints, the realtime constructor chain), two confidently-written but
false claims about the legacy embedders, and the measurement script to run on a candidate
before writing any code.

### Additional Embedder: `japanese-hubert-large` (1024-dim)
- `yky-h/japanese-hubert-large`, a public Apache-2.0 mirror of `rinna/japanese-hubert-large`
  (the rinna repo's HF API returns 401). 24 layers, hidden size 1024, ~19k hours of
  ReazonSpeech v1. The commit SHA is pinned in `EMBEDDERS`, same as k2.
- **It is the first embedder here that is not 768-dim**, so the dimension is no longer
  assumed anywhere:
  - `text_enc_hidden_dim` in `logs/<model>/config.json` is rewritten from the width of the
    `.npy` files that were actually extracted (`generate_config` in
    `rvc/train/extract/preparing_files.py`). Only that one key is rewritten, so hand-tuned
    values like `learning_rate` survive a re-extract.
  - Inference and realtime read the width off `enc_p.emb_phone.weight`
    (`checkpoint_text_enc_hidden_dim` in `rvc/lib/utils.py`), which is correct for every
    checkpoint ever saved and needs no metadata migration. `text_enc_hidden_dim` is also
    written into the exported `.pth` for anything that wants the number without the weights.
  - The FAISS index is built at `big_npy.shape[1]` rather than a hardcoded 768, and both
    pipelines skip a mismatched index with a clear message instead of an opaque error.
  - `extract.py` writes this run's own `logs/<model>/mute.npy` with the same embedder, so
    the silent padding rows match the batch width. The shipped `logs/mute*` folders are
    only a fallback for folders extracted before that existed.
- **It is also the first `feat_extract_norm: "layer"` / `conv_bias: true` embedder**, which
  is what makes `do_normalize` load bearing rather than cosmetic - see the measured table
  above. Its waveform normalisation carries a standard deviation floor
  (`EMBEDDER_INPUT_STD_FLOOR`, default 0.01 ≈ -40 dBFS RMS): without it, zero-mean /
  unit-variance normalisation lifts -60 dBFS room tone to 0.95 RMS, a gain of about 60 dB,
  and the embedder reads that amplified noise as speech. Set it to 0.0 for the literal
  `Wav2Vec2FeatureExtractor` behaviour.
- **Warm starting from the stock 768 pretrains works and is the intended path.**
  `enc_p.emb_phone.weight` is the only tensor whose shape depends on the embedder, so
  `load_pretrained` (rules in `rvc/train/warm_start.py`) starts exactly that one from
  scratch and inherits the encoder, flow, decoder and speaker embedding; the discriminator
  loads whole. Any *other* mismatch still stops the run. Measured after the loader rewrite:
  `f0G48k.pth` into a 1024-dim 48k HiFi-GAN inherits 559 of 560 tensors (99.5% of the
  parameters). See "The pretrained loading bug" below for why that was 20% before.
- `embedder_output_layer` selects which layer the features come from, 0 meaning the last.
  It is worth experimenting with here and nowhere else: content peaks below the top layer
  of a 24-layer model while speaker identity is strongest near the bottom, which matters
  because plain HuBERT (unlike contentvec) does not remove speaker information. It is
  recorded in `model_info.json`, the resume checkpoints and the exported `.pth`, and
  inference and realtime read it back, so it never has to be set twice. Because this model
  is `do_stable_layer_norm`, an intermediate layer is a raw pre-norm residual - measured
  from 69 per frame at layer 0 to 538 at layer 23, against 5.9 for the last layer - so
  `embedder_forward` applies `encoder.layer_norm` to it.

### Additional Embedder: `kushinada-hubert-large` (1024-dim, hand installed)
- `imprt/kushinada-hubert-large`, Apache-2.0, a HuBERT Large pre-trained on **62,215 hours**
  of Japanese TV broadcast audio segmented by VAD - roughly 3x the data behind
  `japanese-hubert-large`. Same shape: 24 layers, hidden size 1024,
  `feat_extract_norm: "layer"` / `conv_bias: true` / `do_stable_layer_norm: true`, and the
  same 320x conv frontend, so it agrees with every other embedder at 1742 frames on the
  reference clip.
- **It is the first embedder that cannot be downloaded.** The Hub repo is gated behind a
  license click-through, so there is no unauthenticated fetch and no revision to pin
  against. Its `EMBEDDERS` entry carries `"local": True` plus a `"source"` URL, and
  `load_embedding` grows a third branch for that shape: load straight out of
  `rvc/models/embedders/kushinada_hubert_large/`, and raise a `FileNotFoundError` naming
  the folder and the URL when the weights are not there, rather than attempting a
  download that would 404.
  - Install by hand: accept the license at
    https://huggingface.co/imprt/kushinada-hubert-large, then put `config.json`,
    `preprocessor_config.json` and `pytorch_model.bin` in that folder.
  - The legacy `.bin` branch is **not** reusable for it. That branch would `wget` from
    `APPLIO_EMBEDDER_URL` and, worse, hardcodes `input_do_normalize = False` - which for a
    `feat_extract_norm: "layer"` model silently trains on features that are ~50% wrong.
- `feature_scale = 1.0`. Measured at **8.14** per frame, between contentvec's 9.82 and
  `japanese-hubert-base`'s 6.35, so it needs no correction against `emb_pitch`. Nothing
  like k2. `|pos|/|h| = 1.36` is mildly above the ~0.9 cluster but nowhere near k2's 7.30.
- Its waveform normalisation is load bearing for the same reason as
  `japanese-hubert-large`: skipping it changes the features by 49.80%. The
  `EMBEDDER_INPUT_STD_FLOOR` matters even more here than there - measured, the floor moves
  a -60 dBFS room-tone window's features by 75.5% and takes them from cos 0.48 to cos 0.71
  against digital silence.
- Everything else is already generic and needed no work: warm starting from the stock 768
  pretrains skips exactly `enc_p.emb_phone` (verified end to end: extract -> 1024-wide
  `.npy` and `mute.npy`, `text_enc_hidden_dim` 1024, FAISS index `d=1024`, train from
  `f0G48k.pth`, infer, and realtime through `create_pipeline`), and the resume guard
  refuses a `G_*.pth` stamped with a different embedder. That "train from `f0G48k.pth`"
  ran, but under the pretrained loading bug below: only 20% of G and none of D loaded.
- Realtime cost is indistinguishable from `japanese-hubert-large` - same architecture.
  Measured back to back on an RTX 4090 over a 1.5 s window, fp32: 12.6 ms against 13.1 ms.

### Changing the embedder on an existing model folder
Changing the embedder, its feature scale, its output layer or the input std floor
invalidates every `.npy`, the index **and** `enc_p.emb_phone` together.
`resolve_feature_reuse` re-extracts the features and deletes the index, and
`assert_resumable` (`rvc/train/utils.py`) now refuses to resume from a `G_*.pth` that was
stamped with a different embedder identity. Before that guard existed, training silently
continued from a generator whose `enc_p.emb_phone` - and the Adam moments behind it - had
been fitted to the old features, which produces a model that sounds broken and never
recovers rather than an error. If you hit the refusal, either train under a new model name
or delete the `G_*.pth` / `D_*.pth` to start again from the pretrain.

### The pretrained loading bug (4288bbea .. the warm start rewrite)
From 2026-09-05 until `rvc/train/warm_start.py` existed, **every run started from a
pretrained model - stock or custom - began with a random flow, most of a random decoder,
a random posterior encoder and an entirely random discriminator.** Resuming was never
affected. The cause: every checkpoint on disk (stock `f0G48k.pth`, and every `G_*.pth` /
`D_*.pth`, because `save_checkpoint` writes them that way) stores weight-normed layers as
`*.weight_g` / `*.weight_v`, while the live `state_dict` names them
`*.parametrizations.weight.original0/1`. The old `load_pretrained` filtered with
`key in target` *before* any renaming, and `load_state_dict(strict=False)` hid the rest.
`load_checkpoint` (resume) renamed first, which is why resuming worked while a new folder
pointed at the same `G_2333333.pth` sounded like epoch 1.

Measured on the 48k kushinada config: `f0G48k.pth` loaded 20.3% of G's parameters,
`G_2333333.pth` 20.9%, and both D files 0% (274 and 110 skipped keys). After the rewrite
they load 99.5% (only `enc_p.emb_phone.weight`, a width change), 100% and 100%.
What that meant for the sound, from `naru_dekai_20260913_kushinada_hubert_large/G_2333333.pth`
on 16 training clips before any step: mel L1 1.95 from scratch, **2.36 with the old
loader** (worse than scratch), 0.35 with the new one. And in real training of a new
folder from that G/D, `loss_avg_50/g/mel` at step 50 was 62.8 under the bug against 16.4
after the fix.

The rewrite normalises names first, decides the fate of every tensor before loading,
**exits if any pretrained tensor has no place in the model**, and always prints a
`Warm start (G)` / `Warm start (D)` summary. Read that summary; "the run trained" is not
evidence that the pretrain loaded. `load_state_dict` still accepts the legacy names on
its own through torch's weight_norm compatibility hook - the problem was only ever the
filtering in front of it.

### RefineGAN at 32k / 40k / 48k
The Vocoder radio in the Training tab is visible again (`HiFi-GAN`, `RefineGAN`). Nothing
about RefineGAN was tied to 32k except upstream only shipping 32k pretrains:
- The decoder's upsampling chain divides cleanly for every stock config (48k:
  `[12,10,2,2]`, f0 downsampled 480 -> 240 -> 120 -> 12 -> 1), and the forward output is
  exactly `segment_size` (17280) at 48k. `cond` now takes `gin_channels` instead of a
  literal 256, and `Synthesizer` passes `gin_channels` / `upsample_initial_channel`
  through; with the stock 256 / 512 the weights are unchanged.
- The decoder is independent of the embedder width, so 768 and 1024 both work through the
  existing `enc_p.emb_phone` mechanism; `enc_p` / `enc_q` / `flow` / `emb_g` are key for key
  identical between the two vocoders.
- `train.py` follows upstream for RefineGAN: `MultiPeriodDiscriminator(version="v3")`
  (scale + periods 2, 3, 5, 7, 11 + three `DiscriminatorR` STFT resolutions) and the
  multi-scale mel loss. HiFi-GAN keeps v2 and the single-scale loss, unchanged.
  `DiscriminatorR` runs its STFT in fp32 because CUDA FFT rejects half precision.
  Measured on an RTX 4090, one fp16 G+D step at batch 4 peaks at 2.7 GiB.
- Decoder size: RefineGAN 13.2M parameters, HiFi-GAN 15.7M at 48k.
- `pretrained_selector` falls back to the HiFi-GAN pretrain at the same sample rate when
  `rvc/models/pretraineds/refinegan/f0G{sr}k.pth` does not exist, and says so; with
  nothing at all it now prints that it is training from scratch instead of doing it
  silently.
- Inference, realtime, export and the blender needed no change: they build the
  `Synthesizer` from `cpt["vocoder"]`, and the blender already refuses mismatched key sets.

### SiFi-GAN at 32k / 40k / 48k
A third vocoder, from the official ICASSP 2023 implementation
([chomeyama/SiFiGAN](https://github.com/chomeyama/SiFiGAN), MIT). The decoder is split in
two: a **source network** (`dec.sn.*`) that turns a sine excitation into an excitation
signal using convolutions whose dilation follows the pitch, and a **filter network**
(`dec.fn.*`) that shapes it into the waveform. `enc_p` / `enc_q` / `flow` / `emb_g` / F0 /
speaker conditioning are untouched; only the decoder is new.

The integration is deliberately shaped so a HiFi-GAN pretrain is worth as much as possible:

- **The filter network *is* this fork's HiFi-GAN decoder.** `conv_pre`, `cond`, `m_source`
  (the same `SourceModuleHnNSF`), `fn.upsamples` <-> `dec.ups`, `fn.blocks` <->
  `dec.resblocks` and `fn.output_conv` <-> `dec.conv_post` are key for key and shape for
  shape identical. Verified for all three sample rates in `tests/test_sifigan.py`.
- **The official `assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]` is not used.**
  This fork uses the odd-rate padding from `HiFiGANNSFGenerator` instead, so the stock
  kernels work unchanged. That assertion would reject 40k outright (`[16,16,4,4]` against
  `[10,10,2,2]`), and using 2*rate kernels instead would break the 1:1 correspondence with
  `dec.ups` at 40k. The stock kernels give both.
- **The official downsample padding is off by one for 40k.** `upsample_scales[i] -
  (kernel % 2 == 0)` assumes kernel == 2*rate; generalised to `ceil((kernel - rate) / 2)`,
  which divides the length by exactly the rate for both kernel sets.
- Output is exactly `frames * prod(upsample_rates)` = `segment_size`, same as the others.
- **`forward()` now returns six elements**, the sixth being the source excitation
  (`None` for every other vocoder). `infer()` drops it. `rvc/train/train.py` is the only
  caller of `forward()`; inference and realtime use `infer()`.
- `train.py` gives SiFi-GAN the same treatment as RefineGAN: `disc_version="v3"` and the
  multi-scale mel loss. The official SiFi-GAN trains against a UnivNet multi-resolution
  spectral discriminator plus a HiFi-GAN multi-period one, which is what v3 already is, so
  **no discriminator was ported.**
- Decoder size at 48k: SiFi-GAN 27.9M parameters against HiFi-GAN 15.7M and RefineGAN
  13.2M. Whole `Synthesizer` at 48k/768: 49.8M against HiFi-GAN's 37.6M.

#### The pitch-dependent dilations
`d[i]` is `(sample_rate / dense_factors[i]) / f0`, repeated to
`frames * cumprod(upsample_rates)[i]` - the length of the feature map after stage `i`
(48k: `[432, 4320, 8640, 17280]` for a 36-frame segment). Unvoiced frames (`f0 == 0`) are
given the pitch that makes the factor exactly 1, i.e. no adaptation.

Two things about the official formula that are easy to get wrong and are reproduced here
verbatim: **`dilated_factor` is called with the full sample rate for every stage**, not
with that stage's own rate (the collater computes `df_sample_rates` but only uses it in a
length assertion), and `dense_factors` defaults to the official `[0.5, 1, 4, 8]`. The
resulting dilation-to-period ratio is `hop / (dense_factor_i * cumprod_i)`: official 24k
gives `[48, 6, 0.5, 0.125]`, this fork's 48k gives `[80, 4, 0.5, 0.125]`, so **the two
innermost stages match the official exactly** and the outer two differ because the
upsampling schedule does. `dense_factors = [0.833, 0.667, 4, 8]` would match all four.

#### Filter blocks: `rvc` (default) or `official`
Training tab > Vocoder > `SiFi-GAN` reveals a second radio. `rvc` builds the filter blocks
from this repository's `ResBlock` (kernel sizes 3/7/11, dilations 1/3/5, two convolutions
per dilation), which is what makes the 1:1 correspondence above possible. `official`
follows the paper (one convolution per dilation, kernel sizes 3/5/7) and its filter blocks
consequently have no counterpart in any existing model.

It is a structural choice, so it travels like `vocoder` does: `sys.argv[17]` (appended, so
no existing position shifts), stamped into `architecture_identity` and therefore into every
`G_*.pth` / `D_*.pth`, and into the exported `.pth`. `assert_resumable` refuses a resume
that changes it. In `extract_model` it is **read off the weights** rather than threaded
through as an argument, the same way `detect_vocoder` works: the official blocks have no
`convs2`.

#### The source regularisation loss
`rvc/train/source_loss.py` (`ResidualLoss`) plus `rvc/lib/algorithm/cheaptrick.py`, both
ported from the official repo. It asks the excitation's mel spectrum to match the target
waveform's with the CheapTrick spectral envelope divided out. **Without it the source
network is unsupervised and the source-filter decomposition never forms** - what is left is
an ordinary vocoder with quasi-periodic convolutions in it.

- Weight is `train.c_reg` in `logs/<model>/config.json`, default 1.0, `0` turns it off.
  Built only when the vocoder is SiFi-GAN, so the other vocoders' training loops are
  byte-for-byte unchanged.
- **`c_reg` is deliberately *not* in `TRAIN_SETTING_KEYS`.** `read_train_settings` returns
  a config's values only when *all* the listed keys are present, so adding one would make
  every existing model folder fall through to the stock config and show the wrong
  learning_rate / c_mel.
- CheapTrick needs `fft_size > 3 * sample_rate / f0_floor`. The defaults are
  `fft_size=4096, f0_floor=50, f0_ceil=1100` to cover this fork's F0 range rather than the
  official 2048/100/840, which would clamp everything under 100 Hz. Buffers cost ~37 MB.
- Pure torch and differentiable; measured in a real step, the source network receives
  gradient on 142 tensors.

**What it costs, measured.** Two 50 epoch runs from scratch on the same 128 clips,
identical but for `c_reg`:

| | `mel` at 50 | `kl` at 50 |
|---|---|---|
| `c_reg = 1.0` (default) | 31.81 | 1.73 |
| `c_reg = 0` | **28.42** | 1.68 |

So the loss costs **3.4 of mel** and buys 0.05 of kl. That is not an argument for turning it
off: what it is there to buy - the source-filter decomposition, and with it the pitch
controllability that is the whole point of SiFi-GAN - is not something either number can
see. But anyone comparing SiFi-GAN's mel against another vocoder should know that roughly
3 points of it are being spent here, and that `c_reg` is the knob.

#### How it compares to RefineGAN, measured
Same 128 clips, batch 4, same v3 discriminator and multi-scale mel loss, warm starts from
the same stock `f0G48k` / `f0D48k`, 50 epochs. Because the loss function and the
discriminator are identical here, these numbers *are* directly comparable (unlike a
SiFi-GAN or RefineGAN run against a HiFi-GAN one).

| | `mel` at 50 | `kl` at 50 |
|---|---|---|
| SiFi-GAN, warm | **24.84** | 1.15 |
| RefineGAN, warm | 27.37 | **0.53** |
| SiFi-GAN, scratch | **31.81** | 1.73 |
| RefineGAN, scratch | 49.52 | **0.22** |

- **SiFi-GAN converges far faster on mel**, and it is architectural rather than an artifact
  of the warm start: the gap from scratch (17.7) dwarfs the gap when both are warm started
  (2.5). It also wins from the weaker position - from the same pretrain SiFi-GAN inherits
  75.1% of its parameters where RefineGAN inherits 93.1%, because SiFi-GAN's decoder is
  bigger (27.9M against 13.2M) and its source network is entirely new.
- **RefineGAN is consistently better on `kl`**, by 0.6 warm and 1.5 from scratch, and
  `c_reg` accounts for only 0.05 of that - it is the architecture. `kl` is the number this
  fork's notes tie to conversion quality, so this is not a footnote.
- At 50 epochs SiFi-GAN has flattened (mel -0.09 over the last 10 epochs) while warm
  RefineGAN is still moving (-0.93), so **whether the mel lead survives is unmeasured**.
  50 epochs on 128 clips is an early, small experiment, and neither number is perceptual.

#### `source_scales`: why a faithful port of the official generator warm starts badly
Mel L1 against the ground truth on 16 real training clips **before any optimizer step**
(48k, kushinada-hubert-large 1024-dim, posterior-encoder path). This is the instrument
that caught the pretrained loading bug; training loss at step N mixes in optimizer
transients and cannot answer this. Reference points, all healthy:

| configuration | mel L1 |
|---|---|
| HiFi-GAN from scratch | 1.847 |
| **HiFi-GAN <- stock `f0G48k` (768 -> 1024)** | **0.685** |
| RefineGAN from scratch | 3.250 |
| **RefineGAN <- its own `G_*.pth` (100%)** | **0.435** |

Built exactly as the paper describes, SiFi-GAN warm started **worse than from scratch**.
The cause is not a missing tensor. Its filter stages compute
`fn.upsamples[i](c) + embs[-i-1]`, where `embs` comes from the **freshly initialised
source network**, while the HiFi-GAN residual blocks being inherited were trained on
`ups(x) + noise_convs(har_source)`, whose additive term is a tanh-bounded sine. Measured
rms ratio of additive term to upsampled path:

| stage | HiFi-GAN (trained on this) | SiFi-GAN at init, no gain |
|---|---|---|
| 0 | 0.67 | 0.12 |
| 1 | 0.21 | 0.95 |
| 2 | 0.17 | **3.00** |
| 3 | 0.43 | **2.68** |

The fresh source network's output has rms 0.390 against the sine's 0.030, about 13x, so
the inherited blocks are driven by a term three times their own input and run far out of
distribution.

So `SiFiGANGenerator` adds **`dec.source_scales`**, a learnable per-stage gain on that
additive term (`DEFAULT_SOURCE_SCALE_INIT`, and `sifigan_source_scale_init` on
`Synthesizer`). RefineGAN's `AdaIN` already does the same thing in this repository with a
1e-4 initialised weight. The source network is supervised directly by the regularisation
loss, so a small gain does not starve it. Chosen by measurement, every configuration
built from the same RNG state:

| gain | SiFi-GAN from scratch | SiFi-GAN <- stock `f0G48k` | SiFi-GAN <- 1024-dim RefineGAN |
|---|---|---|---|
| 1.00 (the paper) | 1.907 | **2.876** | 3.480 |
| 0.30 | 1.830 | 1.946 | 3.826 |
| 0.10 | 1.893 | 1.245 | 3.869 |
| **0.03 (default)** | 1.929 | **0.999** | 3.881 |
| 0.00 | 1.952 | 1.069 | 3.881 |

At the paper's gain the warm start is 51% *worse* than scratch; at 0.03 it is 48%
*better*. The curve is flat between 0 and 0.1 but both ends are worse than 0.03, so a
little source signal helps and a lot of it hurts.

**The RefineGAN -> SiFi-GAN port is worth nothing at initialisation** (3.5 - 3.9 at every
gain, against 1.9 from scratch) and no gain rescues it: that port carries only
`fn.blocks` and `fn.output_conv`, so the inherited blocks sit behind a random `conv_pre`,
`cond` and `fn.upsamples` whatever the source term does. Warm start SiFi-GAN from a
**HiFi-GAN** checkpoint, not a RefineGAN one, even when a same-width RefineGAN is the
model you happen to have.

The head start survives training, which a step-0 number alone cannot show. Two 50 epoch
runs on the same 128 clips, batch 4, 32 steps/epoch, both at the default gain, one warm
started from the stock `f0G48k` / `f0D48k` and one from scratch:

| epoch | `mel` warm | `mel` scratch | `kl` warm | `kl` scratch |
|---|---|---|---|---|
| 1 | 31.79 | 61.65 | 8.69 | 60.33 |
| 10 | 27.08 | 39.58 | 1.82 | 1.76 |
| 30 | 25.02 | 33.66 | 1.29 | 1.80 |
| 50 | **24.84** | 31.81 | **1.15** | 1.73 |

`fm` is higher for the warm start (8.7 against 4.4), which is what a run further along
looks like rather than a problem. The gain itself is learnable and behaves: from 0.03 the
warm started run moved it to `[0.013, 0.022, 0.032, 0.012]` and the scratch run to
`[0.035, 0.028, 0.027, 0.022]` - the model with inherited blocks pushes the source term
*down* rather than opening the gate, and neither runs away towards the paper's 1.0.

### Warm starting across embedders and vocoders
`rvc/train/warm_start.py` is the single place that decides what a pretrained G / D may
contribute. It works from meaning, not from shape equality:
- **Identity of the source.** Vocoder from the decoder's keys (works for every checkpoint
  ever saved), else the recorded `vocoder`. `save_checkpoint` now stamps `vocoder`,
  `sample_rate` and `disc_version` into every `G_*.pth` / `D_*.pth` next to the embedder
  identity. A legacy HiFi-GAN's sample rate is recognised from its transposed-conv kernel
  sizes, which differ between the stock configs, and so is a legacy SiFi-GAN's (from
  `dec.fn.upsamples`, the same transposed convolutions); a legacy RefineGAN's cannot be
  (its shapes do not depend on the sample rate). **A SiFi-GAN decoder's shapes do depend
  on it**, so it is excluded from the "shapes do not depend on the sample rate" assumption
  in `_plan_decoder` - claiming otherwise would inherit a whole decoder across rates.
- **Generator, same vocoder:** everything name for name; a shape mismatch stops the run,
  except `enc_p.emb_phone` when the embedder width differs.
- **Generator, HiFi-GAN <-> RefineGAN:** `enc_p` / `enc_q` / `flow` / `emb_g` whole, then
  `CROSS_VOCODER_DECODER_PORTS`: the 12 residual blocks
  (`dec.resblocks.{3i+k}` <-> `dec.upsample_conv_blocks.{i}.blocks.{k}.1`, same channels,
  kernel sizes and dilations, checked) and `conv_post`, converted between plain and
  weight-normed (`g = ||w||`, `v = w`; back as `g * v / ||v||`). Everything else in the
  decoder starts from scratch and is listed. Only done when both sample rates are known
  and equal; otherwise the decoder is left entirely fresh with a note. This is a good
  starting point, not an identical function: the LeakyReLU slope is 0.1 vs 0.2 and
  RefineGAN puts a fresh `input_conv` in front of each stage. Measured from
  `G_2333333.pth` into a 1024-dim 48k RefineGAN: 535 of 588 tensors, 93.6% of parameters.
- **Generator, HiFi-GAN <-> SiFi-GAN:** the largest port of the three, because SiFi-GAN's
  filter network is the HiFi-GAN decoder: `conv_pre`, `cond`, `m_source`, `fn.upsamples`
  <-> `dec.ups`, `fn.blocks` <-> `dec.resblocks` and `fn.output_conv` <-> `dec.conv_post`,
  all name for name. Measured at 48k/768: **HiFi-GAN -> SiFi-GAN 75.5%** of parameters
  (552 tensors), **SiFi-GAN -> HiFi-GAN 99.9%**. Only HiFi-GAN's `noise_convs` are dropped
  (SiFi-GAN injects the sine through its source network instead), and only `dec.sn.*` plus
  `dec.fn.downsamples.*` start fresh - the latter shrink the *excitation* for the filter
  skips, a different job from `noise_convs`, which is why they have no counterpart either.
  With `filter_resblock="official"` the residual-block entries are simply left out rather
  than allowed to clash, so the port is not abandoned as a whole: 62.2% still transfers.
- **Generator, RefineGAN <-> SiFi-GAN:** the same range as HiFi-GAN <-> RefineGAN, for the
  same reasons - the residual blocks (identical channels, kernels and dilations) and
  `conv_post` with the weight-norm conversion. Measured at 48k/768: RefineGAN -> SiFi-GAN
  66.1%, SiFi-GAN -> RefineGAN 93.6%.
- **Discriminator:** sub-discriminators matched by descriptor (`S`, `P(period)`,
  `R(resolution)`), never by index - a period 17 and a period 23 discriminator have the
  same shapes. v2 -> v3 inherits S and P2..P11 (99.4% of v3's parameters), starts the
  three R from scratch and leaves P17/23/37 out. An unrecognised layout stops the run.
- **Always weights only.** Optimizer and scaler state never cross a warm start.
- **Resume guard.** `assert_resumable` now also refuses a `G_*.pth` whose vocoder (from its
  keys) or recorded sample rate / discriminator version differs from the run. And
  `train.py` only takes the pretrain path when there is no `G_*.pth` + `D_*.pth` pair: a
  checkpoint that exists but fails to load is an error. Before, any exception - including
  a shape mismatch - quietly restarted from the pretrain and then overwrote the checkpoint.

To add another vocoder pair, write a port function that returns `(target_key,
source_keys, convert)` entries and register it in `CROSS_VOCODER_DECODER_PORTS`; the
engine validates every shape before loading anything and abandons the port as a whole if
one entry does not fit.

### Learning rate decay in the Training tab
`Learning Rate Decay` sits next to `Learning Rate` and `Mel Loss Weight` and is handled
the same way (`TRAIN_SETTING_KEYS` in `rvc/train/extract/preparing_files.py`, `--lr_decay`
on the CLI): read from `logs/<model>/config.json` `train.lr_decay`, written back on start,
and must be above 0 and at most 1. `read_train_settings` now falls back to the stock config
per key, so a run config lacking one key still shows its own values for the rest.

Unlike `learning_rate`, a changed `lr_decay` does take effect on a resume, because
`train.py` rebuilds the ExponentialLR from the config every run while the learning rate
itself comes back from the optimizer state in `G_*.pth` / `D_*.pth`. The first resumed
epoch still runs at the saved learning rate, and the new decay applies from the next one.

### Initial Generator LR Boost
Training tab > Advanced: `Initial Generator LR Boost`, with `Generator LR Multiplier`
(default 3.0) and `Boost Epochs` (default 10) shown only while ticked. Stored as
`train.g_lr_boost_multiplier` / `train.g_lr_boost_epochs` in `logs/<model>/config.json`
(`--g_lr_boost_multiplier` / `--g_lr_boost_epochs` on the CLI); epochs 0 or absent is off.
`rvc/train/lr_boost.py` multiplies only `optim_g`'s learning rate from epoch 1 through the
boost epoch, around the optimizer steps, and restores the exact previous values before
anything logs or saves. So the lr in `G_*.pth` and the ExponentialLR chain are bit for bit
the unboosted ones, the boost is decided from the absolute epoch number (a resume neither
restarts nor loses it), and with it off nothing touches the optimizer. Unticking a
previously enabled boost writes epochs 0; unticking on a config that never had one leaves
the file untouched. TensorBoard's `learning_rate` shows the effective generator rate.

### TorchCompile during training (extraction only)
`training_compile_extraction` (`assets/config.json`, default off) compiles the embedder
and the RMVPE/FCPE pitch models during training feature extraction
(`rvc/train/extract/compile_extract.py`). CREPE already compiles itself through the
"Enable TorchCompile (CREPE)" setting.

Per file this is x1.32 (embedder), x1.33 (RMVPE), x1.52 (FCPE) on an RTX 4090, measured
A-B-A so warm-up cannot be read as a speedup. But tracing costs ~17 s per worker process
per stage and the inductor cache does not remove it, so end to end on 200 clips the
compiled run measured 61 s against 27 s eager. **Break-even is around 4000 clips (~4
hours of dataset); below that, leave it off.**

Every clip has a different length (771 distinct lengths in 1364 files), so these compile
with `dynamic=True` and no CUDA graphs. Output is bit-identical for RMVPE and within
2.1e-04 relative for the embedder - the same fp32 rounding level that separates eager
from compiled anywhere else.

**The training step itself is deliberately not compiled.** net_g + net_d measured x1.03
against an eager control that reproduced to x1.00, needs MSVC on PATH for inductor's C++
wrapper, costs 60-240 s of compile time, and would prefix every saved checkpoint key with
`_orig_mod.`. Enabling TF32 matmul on top measured x0.99 - the model is convolution bound
and cuDNN already runs convolutions in TF32. Compiling does cut peak VRAM by 24%
(6.38 -> 4.84 GiB), which is worth knowing if the goal is a larger batch rather than speed.

Full numbers, including why compiling does not improve audio quality, are in
`TORCHCOMPILE_ACCURACY_REPORT.md`.

### Realtime input gating and the fixed RNG seed
`is_input_silent` in `rvc/realtime/core.py` used to be computed and then never read, so
both the VAD checkbox and the Silence Threshold slider did nothing: the model converted
room tone forever and `audio_model * sqrt(vol)` amplified it. The gate is now real, but
it waits until the **whole conversion window** is silent (`silence_blocks_to_stop`,
2 blocks for the reference template) rather than the first silent block - gating earlier
cuts a decaying tail mid-decay, which is why the decision had been discarded. A threshold
of 0 dBFS or above means "no gating", since `10 ** (0 / 20)` is 1.0 and would mute
everything. The slider now reaches -20 dB; real room tone sits near -50, out of reach of
the old -60 limit.

`VADProcessor` builds its `webrtcvad.Vad` **per call and discards the first 3 frames**.
Measured on 0.96 s blocks of -50 dBFS room tone: a detector carried through the session
called 27-31 of 32 frames speech, a fresh one 3 of 32 (its warm-up), and a fresh one with
those 3 frames dropped 0 of 29 - against 2-29 of 29 for real speech. That separation is
what lets the original "any single frame is speech" rule stand, so an utterance is never
clipped at the onset.

`realtime_seed` (`assets/config.json`, "RNG Seed" in the Realtime tab, -1 = random) fixes
the generator's per-chunk noise so the voice stops drifting between sessions. It is
applied in `AudioCallbacks.__init__` **after** the warm-up, and the warm-up now runs even
without compilation, because the lazily built F0 model's weight initialisation consumes
RNG and a second session that reuses torchcrepe's cache would otherwise diverge.

**A seed does not make two sessions identical, and the residual is `mangio-crepe`.**
Repeating the same input in one process: rmvpe, fcpe and crepe-full are bit-identical,
while mangio-crepe-full-speech drifts 25.9 cents and mangio-crepe-full 26.7 cents. The
cause is the decoder - see below. Numbers are in `TORCHCOMPILE_ACCURACY_REPORT.md`.

### Mangio-CREPE decoder
`mangio_crepe_decoder` (`assets/config.json`, default `viterbi`) picks how mangio-crepe
turns the network output into a pitch (`rvc/lib/predictors/crepe_decoder.py`). The picker
appears next to the "Pitch extraction algorithm" control and only while a mangio-crepe
method is selected; it is built once in `tabs/components.py` and reused by realtime,
inference, batch inference, TTS and the F0 curve tool.

**Training extraction is not part of this setting.** It always decodes with `viterbi`
(`TRAINING_MANGIO_CREPE_DECODER` in `rvc/train/extract/extract.py`, passed to
`MANGIO_CREPE(decoder=...)`), and the Training tab has no picker. The setting is global,
so before this a decoder chosen for inference or realtime silently changed the pitch every
later extraction trained on.

Measured on an RTX 4090, five consecutive runs of the same audio in one process:

| decoder | repeatable | worst drift | realtime cost |
|---|---|---|---|
| `viterbi` (default, what mangio-crepe always used) | no | 23.5 cents | 107 ms/block |
| `weighted_argmax` | **bit-identical** | 0 | **84 ms/block** |
| `argmax` | no | 30.9 cents | 76 ms/block |

CUDA `argmax` breaks ties arbitrarily and CREPE's bins are 20 cents apart, so one tie
moves the estimate a whole bin; `viterbi` decodes a path through those same per-frame
choices and inherits it. `weighted_argmax` averages around the peak instead, which is why
the plain `CREPE` class has always passed it explicitly.

`weighted_argmax` is both repeatable and ~24 ms/block cheaper (measured in both
orderings), but it is a different estimator: against `viterbi` it moves the output by
1.043 dB median mel distance, about what changing the RNG seed does (1.188 dB). So it
changes the voice, which is why the default is left alone and the choice is exposed
rather than made here.

### Realtime embedder precision
`embedder_precision` (`fp32` / `bf16` / `fp16`, default `fp32`) is saved in
`assets/config.json` and in realtime templates. Measured on an RTX 4090 over a 1.5 s
window: `japanese-hubert-large` 10.4 ms against `japanese-hubert-base` 5.7 ms, while
`mangio-crepe-full` alone costs 40.6 ms and `rmvpe` 18.8 ms. So the Large embedder adds
under 5 ms and F0 stays the dominant cost. bf16 measured *slightly slower* than fp32 at
this size, because the embedder is kernel-launch bound rather than compute bound - the
option is there for slower cards, and bf16 is preferred over fp16 since deep pre-norm
transformers can overflow in fp16.

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
