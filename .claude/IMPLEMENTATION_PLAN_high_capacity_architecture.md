# High-Capacity Architecture Implementation Plan

**Date:** 2025-01-04
**Status:** READY FOR IMPLEMENTATION
**Priority:** HIGH

---

## Executive Summary

This document provides a complete implementation plan for adding **high-capacity (768-dim) architecture support** to Applio, enabling users to leverage the full potential of japanese-hubert-large (1024-dim) and other embedders without the bottleneck of 192-dim compression.

**Key Goals:**
1. Add optional 768-dim `hidden_channels` support alongside existing 192-dim
2. Maintain full backward compatibility with existing 192-dim models
3. Support both 768-dim embedders (contentvec, japanese-hubert-base, etc.) and 1024-dim embedders (japanese-hubert-large)
4. Automatic detection of model architecture during inference
5. User-friendly UI for architecture selection during training

---

## Background & Motivation

### Current Limitation

RVC currently compresses embedder outputs to 192-dim `hidden_channels`:

```
japanese-hubert-large (1024-dim) → 192-dim (18.75% retention) ❌ Massive information loss
japanese-hubert-base (768-dim)   → 192-dim (25.0% retention)  ❌ Significant information loss
```

**Problem:** This compression was designed for:
- Low VRAM environments (RTX 3060 with 6GB)
- Short training sessions (10-30 minutes of data, 500 epochs)
- Quick convergence

**User's Environment:**
- RTX 4090 (24GB VRAM) ✅
- Long training sessions (10-100 hours of data, 1000-2000 epochs)
- Learning rate decay (1e-4 → 1e-7)
- Goal: Exceed japanese-hubert-base quality

### Solution: High-Capacity Architecture

Add 768-dim `hidden_channels` option:

```
japanese-hubert-large (1024-dim) → 768-dim (75% retention) ✅ Preserves most information
japanese-hubert-base (768-dim)   → 768-dim (100% retention) ✅ No compression
```

---

## Architecture Comparison

### Standard (192-dim) Architecture

```json
{
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "text_enc_hidden_dim": 768/1024,  // Dynamic based on embedder
    "n_heads": 2,
    "n_layers": 6
  }
}
```

**Parameters:** ~30M
**VRAM (batch_size=8):** ~4-6GB
**Compression ratio:**
- 768-dim embedder: 25% retention
- 1024-dim embedder: 18.75% retention

### High-Capacity (768-dim) Architecture

```json
{
  "model": {
    "inter_channels": 768,
    "hidden_channels": 768,
    "filter_channels": 2048,
    "text_enc_hidden_dim": 768/1024,  // Dynamic based on embedder
    "n_heads": 12,
    "n_layers": 8
  }
}
```

**Parameters:** ~150M
**VRAM (batch_size=6):** ~12-16GB
**Compression ratio:**
- 768-dim embedder: 100% retention (no compression)
- 1024-dim embedder: 75% retention

---

## Implementation Strategy

### Phase 1: Config Files ✅

Create new config files with `-768` suffix:

**Files to create:**
- `rvc/configs/32000-768.json`
- `rvc/configs/40000-768.json`
- `rvc/configs/48000-768.json`

**Key differences from standard configs:**

```json
{
  "train": {
    "learning_rate": 8e-5  // Slightly lower for larger model
  },
  "model": {
    "inter_channels": 768,          // 192 → 768
    "hidden_channels": 768,         // 192 → 768
    "filter_channels": 2048,        // 768 → 2048
    "n_heads": 12,                  // 2 → 12
    "n_layers": 8,                  // 6 → 8
    "p_dropout": 0.1,               // 0 → 0.1 (regularization for larger model)
    "upsample_initial_channel": 1024,  // 512 → 1024
    "gin_channels": 512             // 256 → 512
  }
}
```

### Phase 2: Model Checkpoint Metadata ✅

Add `hidden_channels` to model checkpoint metadata for automatic detection.

**File:** `rvc/train/train.py`

**Location:** Where `model_info.json` is saved

**Add to metadata:**
```python
model_info = {
    "embedder_model": embedder_name,
    "speakers_id": spk_dim,
    "text_enc_hidden_dim": text_enc_hidden_dim,
    "hidden_channels": config.model.hidden_channels,  # ← NEW
    "sample_rate": config.data.sample_rate,
    "created_at": datetime.now().isoformat()
}
```

**Also add to .pth checkpoint:**
```python
# In save_checkpoint() function
checkpoint = {
    "model": net_g.state_dict(),
    "optimizer": optim_g.state_dict(),
    "learning_rate": lr,
    "iteration": global_step,
    "epoch": epoch,
    "hps": {
        "hidden_channels": config.model.hidden_channels,  # ← NEW
        "text_enc_hidden_dim": text_enc_hidden_dim,
        # ... other params
    }
}
```

### Phase 3: Training Code Modifications ✅

**File:** `rvc/train/train.py`

**Location:** Config selection logic (around line 300 in `run()` function)

**Current code:**
```python
# Load config based on sample rate
config_path = os.path.join("rvc", "configs", f"{sample_rate}.json")
```

**New code:**
```python
# Determine config based on sample_rate and hidden_channels
if hidden_channels == 768:
    config_path = os.path.join("rvc", "configs", f"{sample_rate}-768.json")
    print(f"[High-Capacity Mode] Using 768-dim hidden_channels config: {config_path}")
else:
    config_path = os.path.join("rvc", "configs", f"{sample_rate}.json")
    print(f"[Standard Mode] Using 192-dim hidden_channels config: {config_path}")
```

**Add CLI parameter:**
```python
# In main() function argument parsing
parser.add_argument("--hidden_channels", type=int, default=192, choices=[192, 768],
                    help="Hidden channels dimension (192 for standard, 768 for high-capacity)")
```

### Phase 4: Inference Auto-Detection ✅

**Files to modify:**
- `rvc/infer/infer.py` - VoiceConverter class
- `rvc/infer/pipeline.py` - VC pipeline (if needed)

**Strategy:**

1. **Try to load from model_info.json:**
```python
def load_model(self, pth_path, index_path):
    # Try to load model_info.json
    model_dir = os.path.dirname(pth_path)
    model_info_path = os.path.join(model_dir, "model_info.json")

    hidden_channels = 192  # Default

    if os.path.exists(model_info_path):
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
            hidden_channels = model_info.get("hidden_channels", 192)
            print(f"[Auto-detect] hidden_channels={hidden_channels} from model_info.json")
```

2. **Fallback to .pth checkpoint:**
```python
    else:
        # Try to load from checkpoint
        checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
        if "hps" in checkpoint and "hidden_channels" in checkpoint["hps"]:
            hidden_channels = checkpoint["hps"]["hidden_channels"]
            print(f"[Auto-detect] hidden_channels={hidden_channels} from checkpoint")
```

3. **Final fallback: Detect from model shape:**
```python
        else:
            # Detect from emb_g.weight shape (gin_channels)
            # 192-dim models: gin_channels=256
            # 768-dim models: gin_channels=512
            gin_channels = checkpoint["model"]["emb_g.weight"].shape[1]
            hidden_channels = 768 if gin_channels >= 512 else 192
            print(f"[Auto-detect] hidden_channels={hidden_channels} inferred from gin_channels={gin_channels}")
```

### Phase 5: Realtime Code Modifications ✅

**File:** `rvc/realtime/pipeline.py`

**Same auto-detection strategy as inference.**

**Additional consideration:**
- Template system should save `hidden_channels` with the template
- When loading template, restore the correct architecture

### Phase 6: Training Tab UI ✅

**File:** `tabs/train/train.py` (or wherever the training tab is defined)

**Add Gradio component:**

```python
# In create_train_tab() or similar function

with gr.Row():
    architecture_choice = gr.Radio(
        label="Model Architecture",
        choices=["Standard (192-dim)", "High-Capacity (768-dim)"],
        value="Standard (192-dim)",
        info="Standard: Faster training, lower VRAM. High-Capacity: Better quality with 1024-dim embedders, requires RTX 4090 or similar."
    )
```

**Pass to training function:**

```python
def start_training(..., architecture_choice):
    hidden_channels = 768 if "768" in architecture_choice else 192

    # Build command
    command = [
        sys.executable, "core.py", "train",
        "--model_name", model_name,
        "--sample_rate", str(sample_rate),
        "--hidden_channels", str(hidden_channels),  # ← NEW
        # ... other args
    ]
```

### Phase 7: Inference Tab UI (Optional) ⚠️

**Only if auto-detection fails or user wants manual override**

**File:** `tabs/inference/inference.py`

**Add optional override:**

```python
with gr.Accordion("Advanced Settings", open=False):
    architecture_override = gr.Radio(
        label="Architecture Override (Auto-detect if not set)",
        choices=["Auto", "192-dim", "768-dim"],
        value="Auto",
        info="Leave as Auto for automatic detection from model file."
    )
```

**Note:** In most cases, auto-detection should work, so this is optional.

---

## File-by-File Implementation Guide

### 1. Create Config Files

**Files:** `rvc/configs/32000-768.json`, `40000-768.json`, `48000-768.json`

**Template (40000-768.json):**

```json
{
  "train": {
    "log_interval": 200,
    "seed": 1234,
    "learning_rate": 8e-5,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "lr_decay": 0.999875,
    "segment_size": 12800,
    "c_mel": 45,
    "c_kl": 1.0
  },
  "data": {
    "max_wav_value": 32768.0,
    "sample_rate": 40000,
    "filter_length": 2048,
    "hop_length": 400,
    "win_length": 2048,
    "n_mel_channels": 125,
    "mel_fmin": 0.0,
    "mel_fmax": null
  },
  "model": {
    "inter_channels": 768,
    "hidden_channels": 768,
    "filter_channels": 2048,
    "text_enc_hidden_dim": 768,
    "n_heads": 12,
    "n_layers": 8,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "upsample_rates": [10,10,2,2],
    "upsample_initial_channel": 1024,
    "upsample_kernel_sizes": [16,16,4,4],
    "use_spectral_norm": false,
    "gin_channels": 512,
    "spk_embed_dim": 109
  }
}
```

**For 32000-768.json and 48000-768.json:** Copy and change only `sample_rate`, `hop_length`, and related data params.

---

### 2. Modify `rvc/train/train.py`

#### **A. Add CLI argument (in `main()` function):**

Find the argument parser section and add:

```python
parser.add_argument(
    "--hidden_channels",
    type=int,
    default=192,
    choices=[192, 768],
    help="Hidden channels dimension: 192 (standard) or 768 (high-capacity)"
)
```

Pass to `run()`:

```python
run(
    rank,
    n_gpus,
    experiment_dir,
    pretrainG,
    pretrainD,
    total_epoch,
    save_every_weights,
    config,
    device,
    device_id,
    hidden_channels=args.hidden_channels  # ← ADD
)
```

#### **B. Modify `run()` function signature:**

```python
def run(
    rank,
    n_gpus,
    experiment_dir,
    pretrainG,
    pretrainD,
    custom_total_epoch,
    custom_save_every_weights,
    config,
    device,
    device_id,
    hidden_channels=192  # ← ADD with default
):
```

#### **C. Config selection logic:**

Find where config is loaded (search for `config.json` or where hps/config object is created):

```python
# Determine which config file to use based on sample_rate and hidden_channels
if hidden_channels == 768:
    config_file = f"{config.data.sample_rate}-768.json"
    print(f"[High-Capacity Mode] Using {config_file}")
else:
    config_file = f"{config.data.sample_rate}.json"
    print(f"[Standard Mode] Using {config_file}")

config_path = os.path.join("rvc", "configs", config_file)

# Load config
with open(config_path, "r") as f:
    config = json.load(f)
    config = HParams(**config)  # or however config is loaded
```

#### **D. Save hidden_channels to model_info.json:**

Find where `model_info.json` is created/saved:

```python
model_info = {
    "embedder_model": embedder_name,
    "speakers_id": spk_dim,
    "text_enc_hidden_dim": text_enc_hidden_dim,
    "hidden_channels": config.model.hidden_channels,  # ← ADD
    "sample_rate": config.data.sample_rate,
    # ... other fields
}

with open(model_info_path, "w") as f:
    json.dump(model_info, f, indent=2)
```

#### **E. Save hidden_channels to checkpoint:**

Find `save_checkpoint()` function or where checkpoints are saved:

```python
checkpoint = {
    "model": net_g.state_dict(),
    "optimizer": optim_g.state_dict(),
    "learning_rate": lr,
    "iteration": iteration,
    "epoch": epoch,
    "hps": {
        "hidden_channels": config.model.hidden_channels,  # ← ADD
        "text_enc_hidden_dim": text_enc_hidden_dim,
        "sample_rate": config.data.sample_rate,
        # ... other hps
    }
}

torch.save(checkpoint, checkpoint_path)
```

---

### 3. Modify `core.py` (CLI interface)

Find the `train` subcommand and add the `--hidden_channels` argument:

```python
# In train subcommand
train_parser.add_argument(
    "--hidden_channels",
    type=int,
    default=192,
    choices=[192, 768],
    help="Hidden channels: 192 (standard) or 768 (high-capacity)"
)
```

Pass it to the train function:

```python
from rvc.train.train import main as train_main

train_main(
    # ... existing args
    hidden_channels=args.hidden_channels  # ← ADD
)
```

---

### 4. Modify `rvc/infer/infer.py` (VoiceConverter class)

#### **A. Auto-detection in model loading:**

Find the `load_model()` or similar method:

```python
def load_model(self, pth_path, index_path):
    """
    Load RVC model with automatic architecture detection.
    """
    model_dir = os.path.dirname(pth_path)
    model_info_path = os.path.join(model_dir, "model_info.json")

    # Default values
    hidden_channels = 192
    text_enc_hidden_dim = 768

    # Try to load from model_info.json first
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
                hidden_channels = model_info.get("hidden_channels", 192)
                text_enc_hidden_dim = model_info.get("text_enc_hidden_dim", 768)
                print(f"[Auto-detect] Architecture from model_info.json: hidden_channels={hidden_channels}, text_enc_hidden_dim={text_enc_hidden_dim}")
        except Exception as e:
            print(f"[Warning] Failed to load model_info.json: {e}")

    # Load checkpoint
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)

    # Fallback: Try to load from checkpoint metadata
    if "hps" in checkpoint:
        hidden_channels = checkpoint["hps"].get("hidden_channels", hidden_channels)
        text_enc_hidden_dim = checkpoint["hps"].get("text_enc_hidden_dim", text_enc_hidden_dim)
        print(f"[Auto-detect] Architecture from checkpoint: hidden_channels={hidden_channels}, text_enc_hidden_dim={text_enc_hidden_dim}")

    # Final fallback: Infer from model weights
    if "model" in checkpoint:
        # Check gin_channels (emb_g.weight shape)
        if "emb_g.weight" in checkpoint["model"]:
            gin_channels = checkpoint["model"]["emb_g.weight"].shape[1]
            inferred_hidden = 768 if gin_channels >= 512 else 192
            if hidden_channels == 192 and inferred_hidden == 768:
                hidden_channels = inferred_hidden
                print(f"[Auto-detect] Inferred hidden_channels={hidden_channels} from gin_channels={gin_channels}")

        # Check text_enc_hidden_dim from emb_phone layer
        if "enc_p.emb_phone.weight" in checkpoint["model"]:
            text_enc_hidden_dim = checkpoint["model"]["enc_p.emb_phone.weight"].shape[0]
            print(f"[Auto-detect] Detected text_enc_hidden_dim={text_enc_hidden_dim} from emb_phone layer")

    # Load appropriate config
    sample_rate = checkpoint.get("sample_rate", 40000)  # or however you get sample_rate

    if hidden_channels == 768:
        config_path = os.path.join("rvc", "configs", f"{sample_rate}-768.json")
    else:
        config_path = os.path.join("rvc", "configs", f"{sample_rate}.json")

    with open(config_path, "r") as f:
        config = json.load(f)
        config = HParams(**config)

    # Override text_enc_hidden_dim in config
    config.model.text_enc_hidden_dim = text_enc_hidden_dim

    # Initialize model with correct architecture
    from rvc.lib.algorithm.synthesizers import Synthesizer

    net_g = Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        text_enc_hidden_dim=text_enc_hidden_dim,
        # ... other args
    )

    # Load weights
    net_g.load_state_dict(checkpoint["model"], strict=False)

    # ... rest of loading logic
```

---

### 5. Modify `rvc/realtime/pipeline.py`

**Apply the same auto-detection logic as inference.**

Find where the model is loaded and apply the same strategy.

---

### 6. Modify Training Tab UI

**File:** Find the training tab (likely `tabs/train/train.py` or similar)

#### **A. Add UI component:**

```python
with gr.Row():
    architecture_choice = gr.Radio(
        label="Model Architecture",
        choices=["Standard (192-dim hidden channels)", "High-Capacity (768-dim hidden channels)"],
        value="Standard (192-dim hidden channels)",
        info="📊 Standard: Faster, less VRAM (~4-6GB), suitable for short training sessions. "
             "🚀 High-Capacity: Better quality especially with japanese-hubert-large, requires RTX 4090 or similar (~12-16GB VRAM)."
    )
```

#### **B. Pass to training function:**

```python
def start_training_wrapper(..., architecture_choice):
    hidden_channels = 768 if "768" in architecture_choice else 192

    # Build CLI command
    command = [
        sys.executable,
        "core.py",
        "train",
        "--model_name", model_name,
        "--sample_rate", str(sample_rate),
        "--hidden_channels", str(hidden_channels),  # ← ADD
        # ... other args
    ]

    # Execute
    subprocess.run(command)
```

#### **C. Connect to Gradio event:**

```python
train_button.click(
    fn=start_training_wrapper,
    inputs=[
        model_name_input,
        sample_rate_input,
        # ... other inputs
        architecture_choice,  # ← ADD
    ],
    outputs=[output_textbox]
)
```

---

### 7. Update Inference Tab (Optional)

**Only add if auto-detection proves unreliable in testing.**

```python
with gr.Accordion("Advanced Model Settings", open=False):
    architecture_override = gr.Radio(
        label="Architecture Override",
        choices=["Auto-detect", "Force 192-dim", "Force 768-dim"],
        value="Auto-detect",
        info="Usually auto-detection works. Override only if inference fails."
    )
```

---

## Testing Strategy

### Test Case 1: 192-dim with japanese-hubert-base ✅

**Steps:**
1. Train model with Standard (192-dim) + japanese-hubert-base
2. Verify `model_info.json` contains `"hidden_channels": 192`
3. Load in inference - should auto-detect 192-dim
4. Verify audio quality matches previous implementation

**Expected:** Full backward compatibility

### Test Case 2: 768-dim with japanese-hubert-base ✅

**Steps:**
1. Train model with High-Capacity (768-dim) + japanese-hubert-base
2. Verify `model_info.json` contains `"hidden_channels": 768`
3. Load in inference - should auto-detect 768-dim
4. Verify audio quality exceeds 192-dim version

**Expected:** No compression, 100% retention

### Test Case 3: 768-dim with japanese-hubert-large ⭐

**Steps:**
1. Train model with High-Capacity (768-dim) + japanese-hubert-large
2. Verify `model_info.json` contains `"hidden_channels": 768`, `"text_enc_hidden_dim": 1024`
3. Load in inference - should auto-detect both correctly
4. Verify audio quality significantly exceeds japanese-hubert-base

**Expected:** 75% retention of 1024-dim features, best quality

### Test Case 4: Legacy model compatibility ✅

**Steps:**
1. Load old model trained before this implementation (no `hidden_channels` in metadata)
2. Should default to 192-dim
3. Verify inference still works

**Expected:** Full backward compatibility

### Test Case 5: Realtime conversion ✅

**Steps:**
1. Load 768-dim model in realtime tab
2. Verify auto-detection works
3. Test realtime conversion

**Expected:** Works seamlessly

---

## Expected VRAM Usage

| Configuration | Batch Size | Training VRAM | Inference VRAM |
|--------------|-----------|---------------|----------------|
| 192-dim, 768-dim embedder | 8 | ~4-6GB | ~2-3GB |
| 192-dim, 1024-dim embedder | 8 | ~4-6GB | ~2-3GB |
| 768-dim, 768-dim embedder | 6 | ~10-14GB | ~4-5GB |
| 768-dim, 1024-dim embedder | 6 | ~12-16GB | ~5-6GB |

**RTX 4090 (24GB VRAM):** All configurations supported with headroom

---

## Performance Expectations

### Training Speed

| Configuration | Steps/sec (RTX 4090) | Relative Speed |
|--------------|---------------------|----------------|
| 192-dim | ~8-10 | 1.0x (baseline) |
| 768-dim | ~2-3 | 0.25-0.3x |

**Note:** 768-dim is slower but converges to higher quality

### Quality Improvements (Estimated)

| Metric | 192-dim baseline | 768-dim improvement |
|--------|-----------------|---------------------|
| Pitch accuracy | ±0.5 semitones | ±0.1-0.2 semitones |
| Speaker similarity | Baseline | +30-50% |
| High-freq quality | Baseline | +40-60% |
| Prosody naturalness | Baseline | +20-40% |

---

## Backward Compatibility Checklist

✅ **Old models (no hidden_channels metadata):** Default to 192-dim
✅ **Config files:** New `-768` configs don't affect existing configs
✅ **CLI:** `--hidden_channels` defaults to 192
✅ **UI:** Default selection is "Standard (192-dim)"
✅ **Inference:** Auto-detection with graceful fallback

---

## Future Enhancements (Optional)

### 1. Even Larger Architectures

For users with A100/H100:
- 1536-dim hidden_channels
- 4096-dim filter_channels

### 2. Mixed Precision Optimizations

Leverage BF16 on RTX 4090 for faster training

### 3. Architecture Search

Automated hyperparameter search for optimal architecture

---

## Summary for Another Claude Instance

**If you're implementing this feature, follow these steps:**

1. ✅ **Create 3 config files:** `32000-768.json`, `40000-768.json`, `48000-768.json` with the architecture shown in this doc
2. ✅ **Modify `rvc/train/train.py`:** Add `--hidden_channels` CLI arg, config selection logic, save to metadata
3. ✅ **Modify `core.py`:** Add `--hidden_channels` to train subcommand
4. ✅ **Modify `rvc/infer/infer.py`:** Add auto-detection logic in model loading
5. ✅ **Modify `rvc/realtime/pipeline.py`:** Same auto-detection as inference
6. ✅ **Modify training tab UI:** Add architecture selection radio button
7. ✅ **Test all 5 test cases** listed above

**Key principle:** Auto-detect everywhere possible, manual override only as fallback.

**Backward compatibility:** Default to 192-dim when metadata is missing.

---

**Status:** Ready for implementation
**Estimated Time:** 4-6 hours
**Risk:** Low (backward compatible, well-isolated changes)
