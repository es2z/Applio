# Implementation Plan: Adding japanese-hubert-large Embedder to Applio

## Executive Summary

This document provides a comprehensive plan for adding `japanese-hubert-large` as a new embedder model option in Applio's Training tab. The primary challenge is that japanese-hubert-large outputs **1024-dimensional** embeddings, while all existing embedders (contentvec, japanese-hubert-base, etc.) output **768-dimensional** embeddings. This dimensional difference requires careful handling throughout the training and inference pipelines.

## Background

### Current Embedder Models
| Model | Dimensions | Source |
|-------|-----------|--------|
| contentvec | 768 | Applio repo |
| spin | 768 | Applio repo |
| spin-v2 | 768 | Applio repo |
| chinese-hubert-base | 768 | Applio repo |
| japanese-hubert-base | 768 | Applio repo |
| korean-hubert-base | 768 | Applio repo |
| **japanese-hubert-large** | **1024** | **rinna (to be added)** |
| custom | variable | user-provided |

### Key Technical Details

**japanese-hubert-large specifications:**
- Hidden size: 1024
- Classifier projection size: 256
- Layers: 24
- Attention heads: 16
- Source: https://huggingface.co/rinna/japanese-hubert-large

**RVC Architecture:**
- Uses `text_enc_hidden_dim` parameter in `Synthesizer` class to handle embedder dimensions
- Default `text_enc_hidden_dim` is 768 (defined in config JSONs)
- Currently hardcoded as: `768 if version == "v2" else 256`
- This hardcoding must be replaced with dynamic dimension detection

## Critical Issue: Dimensional Mismatch

### Current Behavior
```python
# In rvc/infer/infer.py:477 and rvc/realtime/pipeline.py:64
self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
```

This hardcoded logic assumes all v2 models use 768-dimensional embedders. **Models trained with japanese-hubert-large (1024-dim) will fail to load** with this logic.

### Required Solution
1. Save `text_enc_hidden_dim` in model checkpoints during training
2. Load `text_enc_hidden_dim` from checkpoints during inference
3. Fall back to embedder-based inference if not found (backward compatibility)
4. Fall back to version-based inference as final fallback (legacy support)

## Implementation Plan

### Phase 1: Core Embedder Infrastructure

#### 1.1 Add Embedder Definition
**File:** `rvc/lib/utils.py`

**Changes:**
```python
# Line ~108-115: Add to embedding_list
embedding_list = {
    "contentvec": os.path.join(embedder_root, "contentvec"),
    "spin": os.path.join(embedder_root, "spin"),
    "spin-v2": os.path.join(embedder_root, "spin-v2"),
    "chinese-hubert-base": os.path.join(embedder_root, "chinese_hubert_base"),
    "japanese-hubert-base": os.path.join(embedder_root, "japanese_hubert_base"),
    "korean-hubert-base": os.path.join(embedder_root, "korean_hubert_base"),
    "japanese-hubert-large": os.path.join(embedder_root, "japanese_hubert_large"),  # ADD THIS
}

# Line ~117-124: Add to online_embedders
online_embedders = {
    # ... existing entries ...
    "japanese-hubert-large": "https://huggingface.co/rinna/japanese-hubert-large/resolve/main/pytorch_model.bin",  # ADD THIS
}

# Line ~126-133: Add to config_files
config_files = {
    # ... existing entries ...
    "japanese-hubert-large": "https://huggingface.co/rinna/japanese-hubert-large/resolve/main/config.json",  # ADD THIS
}
```

#### 1.2 Create Dimension Mapping Function
**File:** `rvc/lib/utils.py`

**Add after `load_embedding` function (around line 157):**
```python
def get_embedder_dim(embedder_model: str) -> int:
    """
    Returns the output dimension for a given embedder model.

    Args:
        embedder_model (str): Name of the embedder model

    Returns:
        int: Output dimension (768 or 1024)
    """
    embedder_dims = {
        "contentvec": 768,
        "spin": 768,
        "spin-v2": 768,
        "chinese-hubert-base": 768,
        "japanese-hubert-base": 768,
        "korean-hubert-base": 768,
        "japanese-hubert-large": 1024,
        # custom embedders default to 768 for safety
    }
    return embedder_dims.get(embedder_model, 768)
```

### Phase 2: Training Pipeline Updates

#### 2.1 Save text_enc_hidden_dim in model_info.json
**File:** `rvc/train/extract/extract.py`

**Location:** Around line 206, after saving `embedder_model`

**Changes:**
```python
# Existing code (line ~197-206):
chosen_embedder_model = (
    embedder_model_custom if embedder_model == "custom" else embedder_model
)
file_path = os.path.join(exp_dir, "model_info.json")
if os.path.exists(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
else:
    data = {}
data["embedder_model"] = chosen_embedder_model

# ADD THIS:
from rvc.lib.utils import get_embedder_dim
text_enc_dim = get_embedder_dim(embedder_model)
data["text_enc_hidden_dim"] = text_enc_dim

# Existing code continues:
with open(file_path, "w") as f:
    json.dump(data, f, indent=4)
```

#### 2.2 Load and Use text_enc_hidden_dim During Training
**File:** `rvc/train/train.py`

**Location:** Around line 381-392 (where embedder_name is loaded from model_info)

**Changes:**
```python
# Existing code (line ~381-392):
# defaults
embedder_name = "contentvec"
spk_dim = config.model.spk_embed_dim  # 109 default speakers

try:
    with open(model_info_path, "r") as f:
        model_info = json.load(f)
        embedder_name = model_info["embedder_model"]
        spk_dim = model_info["speakers_id"]
except Exception as e:
    print(f"Could not load model info file: {e}. Using defaults.")

# ADD THIS BLOCK:
from rvc.lib.utils import get_embedder_dim

# Determine text_enc_hidden_dim from embedder or use default
text_enc_hidden_dim = 768  # default
if embedder_name:
    text_enc_hidden_dim = get_embedder_dim(embedder_name)
    print(f"Using text_enc_hidden_dim={text_enc_hidden_dim} for embedder '{embedder_name}'")

# Try to load from model_info if available (for resuming training)
try:
    with open(model_info_path, "r") as f:
        model_info = json.load(f)
        if "text_enc_hidden_dim" in model_info:
            text_enc_hidden_dim = model_info["text_enc_hidden_dim"]
            print(f"Loaded text_enc_hidden_dim={text_enc_hidden_dim} from model_info.json")
except:
    pass

# Later, around line 415-424 (Synthesizer initialization):
net_g = Synthesizer(
    config.data.filter_length // 2 + 1,
    config.train.segment_size // config.data.hop_length,
    **config.model,
    use_f0=True,
    sr=config.data.sample_rate,
    vocoder=vocoder,
    checkpointing=checkpointing,
    randomized=randomized,
    text_enc_hidden_dim=text_enc_hidden_dim,  # MODIFY: Add explicit parameter
)
```

**Note:** The `**config.model` unpacking already includes `text_enc_hidden_dim=768` from the JSON config files, but explicitly passing it will override that value.

#### 2.3 Save text_enc_hidden_dim in Model Checkpoint
**File:** `rvc/train/process/extract_model.py`

**Location:** Around line 44-96 (where model metadata is saved)

**Changes:**
```python
# Around line 44-49 (loading model_info.json):
if os.path.exists(os.path.join(model_dir, "model_info.json")):
    with open(os.path.join(model_dir, "model_info.json"), "r") as f:
        data = json.load(f)
        dataset_length = data.get("total_dataset_duration", None)
        embedder_model = data.get("embedder_model", None)
        speakers_id = data.get("speakers_id", 1)
        text_enc_hidden_dim = data.get("text_enc_hidden_dim", 768)  # ADD THIS
else:
    dataset_length = None
    embedder_model = None  # ADD THIS
    text_enc_hidden_dim = 768  # ADD THIS

# Around line 57-98 (building opt dict):
opt = OrderedDict(
    weight={
        key: value.half() for key, value in ckpt.items() if "enc_q" not in key
    }
)
# ... existing config list ...
opt["epoch"] = epoch
opt["step"] = step
opt["sr"] = sr
opt["f0"] = pitch_guidance
opt["version"] = version
opt["creation_date"] = datetime.datetime.now().isoformat()
# ... hash and other fields ...
opt["embedder_model"] = embedder_model
opt["speakers_id"] = speakers_id
opt["vocoder"] = vocoder
opt["text_enc_hidden_dim"] = text_enc_hidden_dim  # ADD THIS
```

### Phase 3: Inference Pipeline Updates

#### 3.1 Update Standard Inference
**File:** `rvc/infer/infer.py`

**Location:** Around line 474-484 (model loading)

**Changes:**
```python
# Existing code (line ~474-479):
self.use_f0 = self.cpt.get("f0", 1)

self.version = self.cpt.get("version", "v1")
self.text_enc_hidden_dim = 768 if self.version == "v2" else 256  # OLD LINE
self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")

# REPLACE WITH:
self.version = self.cpt.get("version", "v1")

# NEW: Load text_enc_hidden_dim with fallback chain
if "text_enc_hidden_dim" in self.cpt:
    # Priority 1: Use saved dimension from checkpoint
    self.text_enc_hidden_dim = self.cpt["text_enc_hidden_dim"]
    print(f"Loaded text_enc_hidden_dim={self.text_enc_hidden_dim} from checkpoint")
elif "embedder_model" in self.cpt:
    # Priority 2: Infer from embedder model name
    from rvc.lib.utils import get_embedder_dim
    self.text_enc_hidden_dim = get_embedder_dim(self.cpt["embedder_model"])
    print(f"Inferred text_enc_hidden_dim={self.text_enc_hidden_dim} from embedder '{self.cpt['embedder_model']}'")
else:
    # Priority 3: Fall back to version-based (legacy support)
    self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
    print(f"Using version-based text_enc_hidden_dim={self.text_enc_hidden_dim} (legacy)")

self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")

# Rest of code remains the same (line 479-484):
self.net_g = Synthesizer(
    *self.cpt["config"],
    use_f0=self.use_f0,
    text_enc_hidden_dim=self.text_enc_hidden_dim,
    vocoder=self.vocoder,
)
```

#### 3.2 Update Real-time Inference
**File:** `rvc/realtime/pipeline.py`

**Location:** Around line 61-72 (model loading in RealTimeVC.__init__)

**Changes:** Apply the exact same logic as in 3.1 above:
```python
# Around line 61-65:
self.use_f0 = self.cpt.get("f0", 1)

self.version = self.cpt.get("version", "v1")
self.text_enc_hidden_dim = 768 if self.version == "v2" else 256  # OLD LINE
self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")

# REPLACE WITH (same as inference.py):
self.version = self.cpt.get("version", "v1")

# Load text_enc_hidden_dim with fallback chain
if "text_enc_hidden_dim" in self.cpt:
    self.text_enc_hidden_dim = self.cpt["text_enc_hidden_dim"]
    print(f"[Realtime] Loaded text_enc_hidden_dim={self.text_enc_hidden_dim} from checkpoint")
elif "embedder_model" in self.cpt:
    from rvc.lib.utils import get_embedder_dim
    self.text_enc_hidden_dim = get_embedder_dim(self.cpt["embedder_model"])
    print(f"[Realtime] Inferred text_enc_hidden_dim={self.text_enc_hidden_dim} from embedder '{self.cpt['embedder_model']}'")
else:
    self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
    print(f"[Realtime] Using version-based text_enc_hidden_dim={self.text_enc_hidden_dim} (legacy)")

self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")

# Rest continues unchanged
```

### Phase 4: UI Updates

#### 4.1 Add to Training Tab UI
**File:** `tabs/train/train.py`

**Location:** Line 531-545 (embedder_model Radio widget)

**Changes:**
```python
embedder_model = gr.Radio(
    label=i18n("Embedder Model"),
    info=i18n("Model used for learning speaker embedding."),
    choices=[
        "contentvec",
        "spin",
        "spin-v2",
        "chinese-hubert-base",
        "japanese-hubert-base",
        "japanese-hubert-large",  # ADD THIS
        "korean-hubert-base",
        "custom",
    ],
    value="contentvec",
    interactive=True,
)
```

**Note:** The order places japanese-hubert-large between japanese-hubert-base and korean-hubert-base for logical grouping.

#### 4.2 UI Considerations for Other Tabs

**Files to Update:**
- `tabs/inference/inference.py` (line ~1124 and ~1759)
- `tabs/tts/tts.py` (line ~307)
- `tabs/realtime/realtime.py` (line ~919)

These tabs all have `embedder_model` Radio widgets that allow users to specify which embedder to use during inference. While the model checkpoint contains the embedder information, these UI options allow users to override it if needed.

**Changes Required:**
Add `"japanese-hubert-large"` to the choices list in each tab, similar to the Training tab:

**Example for Inference Tab (apply similarly to TTS and Realtime):**
```python
embedder_model = gr.Radio(
    label=i18n("Embedder Model"),
    info=i18n("Model used for learning speaker embedding."),
    choices=[
        "contentvec",
        "spin",
        "spin-v2",
        "chinese-hubert-base",
        "japanese-hubert-base",
        "japanese-hubert-large",  # ADD THIS
        "korean-hubert-base",
        "custom",
    ],
    value="contentvec",
    interactive=True,
)
```

**Important Note:**
While users can select the embedder in these tabs, **they should use the same embedder that was used during training**. The UI selection is primarily for:
1. Ensuring the correct embedder is loaded for feature extraction
2. Allowing explicit specification when model metadata is missing
3. Advanced users who want to experiment with cross-embedder inference (not recommended)

**Best Practice:**
The inference pipeline already loads the embedder from the model's saved configuration. These UI options serve as overrides, so users should generally leave them matching the training embedder.

### Phase 5: Testing and Validation

#### 5.1 Pre-implementation Testing Checklist
- [ ] Verify all existing embedders still download correctly
- [ ] Confirm default text_enc_hidden_dim=768 in config JSONs (40000.json, 32000.json, 48000.json)
- [ ] Check that get_embedder_dim() function is imported correctly

#### 5.2 Post-implementation Testing Checklist

**Training:**
- [ ] Train a new model using japanese-hubert-large
- [ ] Verify model_info.json contains `"text_enc_hidden_dim": 1024`
- [ ] Verify final .pth checkpoint contains `"text_enc_hidden_dim": 1024`
- [ ] Verify training completes without dimension errors

**Inference:**
- [ ] Load japanese-hubert-large model in Inference tab
- [ ] Verify text_enc_hidden_dim=1024 is loaded correctly
- [ ] Perform voice conversion with the model
- [ ] Verify audio output quality

**Real-time:**
- [ ] Load japanese-hubert-large model in Realtime tab
- [ ] Verify text_enc_hidden_dim=1024 is loaded correctly
- [ ] Test real-time voice conversion

**TTS:**
- [ ] Use japanese-hubert-large model in TTS tab
- [ ] Verify TTS + voice conversion works correctly

**Backward Compatibility:**
- [ ] Load old models trained before this change
- [ ] Verify they still work (should fall back to version-based dim)
- [ ] Load models trained with other embedders (contentvec, etc.)
- [ ] Verify they work correctly (should use 768 or infer from embedder_model)

**Edge Cases:**
- [ ] Test custom embedder with unknown dimensions (should default to 768)
- [ ] Test model trained with japanese-hubert-large, then inferred with wrong embedder_model in config (should fail gracefully with clear error)

#### 5.3 Expected Download Behavior
When japanese-hubert-large is selected for the first time:
1. `load_embedding()` in utils.py will check for model files
2. If not found, it will download:
   - pytorch_model.bin (1.26 GB) from rinna's HuggingFace
   - config.json (1.78 KB) from rinna's HuggingFace
3. Files will be saved to `rvc/models/embedders/japanese_hubert_large/`

### Phase 6: Documentation and Edge Cases

#### 6.1 Model Compatibility Matrix

| Training Embedder | Inference Embedder | Compatible? | Notes |
|------------------|-------------------|-------------|-------|
| japanese-hubert-large | japanese-hubert-large | ✅ Yes | Ideal case |
| japanese-hubert-large | Any other | ⚠️ No | Dimension mismatch will cause errors |
| Any 768-dim | japanese-hubert-large | ⚠️ No | Dimension mismatch |
| Any 768-dim | Any 768-dim | ✅ Yes | Cross-embedder usage works |

**Important:** Users must use the **same embedder during inference** that was used during training. This is already true in current Applio, but becomes more critical with mixed dimensions.

#### 6.2 Error Handling
If a user tries to load a model trained with japanese-hubert-large but the embedder files are missing:
- `load_embedding()` will automatically download them
- No additional error handling needed beyond existing implementation

If dimension mismatch occurs (shouldn't with this implementation):
- PyTorch will raise a tensor shape mismatch error
- Error message will indicate incompatible model architecture

#### 6.3 User-Facing Documentation Updates
Consider updating:
- CLAUDE.md: Add japanese-hubert-large to embedder list
- README or documentation: Explain dimension differences
- Training tab UI tooltip: Mention japanese-hubert-large is for Japanese audio and has higher dimensionality

### Phase 7: Optional Enhancements

#### 7.1 Config JSON Updates (Optional)
**Files:** `rvc/configs/40000.json`, `32000.json`, `48000.json`

The default `text_enc_hidden_dim: 768` can remain, as it will be overridden during training based on the embedder. However, for clarity, you could add a comment (though JSON doesn't support comments):

Alternative: Add validation in train.py to warn if config dimension doesn't match embedder dimension.

#### 7.2 Validation Helper (Optional)
**File:** `rvc/lib/utils.py`

Add a validation function:
```python
def validate_embedder_dim(embedder_model: str, text_enc_hidden_dim: int) -> bool:
    """
    Validates that the text_enc_hidden_dim matches the embedder model.

    Returns:
        bool: True if valid, False if mismatch
    """
    expected_dim = get_embedder_dim(embedder_model)
    if expected_dim != text_enc_hidden_dim:
        print(f"WARNING: Dimension mismatch! Embedder '{embedder_model}' expects {expected_dim}, but got {text_enc_hidden_dim}")
        return False
    return True
```

Call this during training and inference to warn users of potential issues.

## Summary of Files to Modify

| File | Changes | Complexity |
|------|---------|-----------|
| `rvc/lib/utils.py` | Add embedder definition, create get_embedder_dim() | Medium |
| `tabs/train/train.py` | Add UI choice | Low |
| `tabs/inference/inference.py` | Add UI choice (2 locations) | Low |
| `tabs/tts/tts.py` | Add UI choice | Low |
| `tabs/realtime/realtime.py` | Add UI choice | Low |
| `rvc/train/extract/extract.py` | Save text_enc_hidden_dim to model_info.json | Low |
| `rvc/train/train.py` | Load and use text_enc_hidden_dim | Medium |
| `rvc/train/process/extract_model.py` | Load and save text_enc_hidden_dim in checkpoint | Medium |
| `rvc/infer/infer.py` | Load text_enc_hidden_dim with fallback | Medium |
| `rvc/realtime/pipeline.py` | Load text_enc_hidden_dim with fallback | Medium |

**Total:** 10 files, ~120-180 lines of code changes

**Files NOT Requiring Changes:**
- `core.py` - CLI interface passes parameters through unchanged
- `tabs/realtime/template.py` - Template system saves/loads embedder_model unchanged
- `rvc/train/extract/preparing_files.py` - Only reads embedder_name, no logic changes needed
- `rvc/train/process/model_information.py` - Only displays embedder_model info
- `rvc/realtime/callbacks.py`, `rvc/realtime/core.py` - Pass-through functions only

## Risk Assessment

### High Risk
- **None** (if implementation follows plan)

### Medium Risk
- **Backward compatibility**: Mitigated by fallback chain in inference
- **Dimension mismatch errors**: Mitigated by saving dimension in checkpoint

### Low Risk
- **Download failures**: Existing error handling in load_embedding() covers this
- **Config file conflicts**: Overriding text_enc_hidden_dim works correctly

## Implementation Order

1. **Phase 1.1-1.2**: Add embedder infrastructure (utils.py)
2. **Phase 4.1**: Add UI option to Training tab (easy to test)
3. **Phase 4.2**: Add UI options to Inference, TTS, and Realtime tabs
4. **Phase 2.1**: Save dimension in model_info.json
5. **Phase 2.2**: Use dimension during training
6. **Phase 2.3**: Save dimension in checkpoint
7. **Phase 3.1-3.2**: Load dimension during inference
8. **Phase 5**: Testing
9. **Phase 6-7**: Documentation and optional enhancements

**Rationale for Order:**
- Steps 1-3 are UI/infrastructure only and can be completed and tested independently
- Steps 4-6 modify the training pipeline (must be done together)
- Step 7 modifies the inference pipeline (depends on training changes)
- Step 8 validates everything works end-to-end
- Step 9 improves quality of life

## Success Criteria

- [ ] japanese-hubert-large appears in Training tab embedder dropdown
- [ ] Model trains successfully with japanese-hubert-large
- [ ] Model checkpoint contains text_enc_hidden_dim=1024
- [ ] Model loads correctly in Inference, TTS, and Realtime tabs
- [ ] Voice conversion quality is good (subjective, compare with japanese-hubert-base)
- [ ] Old models still work (backward compatibility)
- [ ] Other embedders still work (regression testing)

## Additional Notes

### Why japanese-hubert-large?
- Larger model → Better speaker representation for Japanese audio
- More training data and parameters → Potentially higher quality
- Already proven in research: https://huggingface.co/rinna/japanese-hubert-large

### Alternative: Use Hugging Face Transformers Hub
Instead of manual download, could use `HubertModel.from_pretrained("rinna/japanese-hubert-large")`. However, current implementation uses local caching, so keeping consistency is preferred.

### Future Embedders
This implementation makes it easy to add future embedders with different dimensions:
1. Add to embedding_list, online_embedders, config_files
2. Add to get_embedder_dim()
3. Add to UI choices
4. Done!

Example: If a 512-dim or 2048-dim embedder is released, just update the mappings.

---

**Document Version:** 1.0
**Created:** 2025-11-02
**For:** Applio Fork by ultrathink
**Target Audience:** Future Claude instances, developers, maintainers
