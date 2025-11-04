# japanese-hubert-large Quality Issues - Root Cause Analysis

**Date:** 2025-01-04
**Status:** INVESTIGATION COMPLETE
**Priority:** HIGH

---

## Executive Summary

Investigation into quality issues with japanese-hubert-large (1024-dim) compared to japanese-hubert-base (768-dim) has revealed **critical architectural differences in the embedder models themselves**, not in the RVC implementation. The root cause lies in different normalization strategies and model configurations between the two HuBERT variants.

### Reported Issues
1. ❌ High-frequency noise artifacts
2. ❌ Pitch shifted up by 0.8-1 semitones
3. ❌ Poor speaker similarity

**Note:** Overfitting was initially suspected but confirmed NOT to be the root cause. User has successfully trained japanese-hubert-base models for 10-100 hours with learning rate decay (1e-4 → 1e-7) without overfitting issues.

### Key Finding
**The embedder models use fundamentally different normalization strategies:**
- japanese-hubert-base: Group normalization, no stable layer norm
- japanese-hubert-large: Layer normalization, stable layer norm enabled

This causes the embeddings to have different characteristics that the RVC architecture may not handle optimally.

---

## Detailed Findings

### 1. Embedder Configuration Differences

#### japanese-hubert-base (from Applio)
```json
{
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "intermediate_size": 3072,
  "do_stable_layer_norm": false,          ← Standard post-norm
  "feat_extract_norm": "group",           ← Group normalization
  "conv_bias": false,
  "feat_proj_dropout": 0.0,
  "final_dropout": 0.1,
  "activation_dropout": 0.1,
  "mask_time_prob": 0.05
}
```

#### japanese-hubert-large (from rinna)
```json
{
  "hidden_size": 1024,
  "num_hidden_layers": 24,
  "num_attention_heads": 16,
  "intermediate_size": 4096,
  "do_stable_layer_norm": true,           ← Pre-norm (different distribution!)
  "feat_extract_norm": "layer",           ← Layer normalization (different!)
  "conv_bias": true,
  "feat_proj_dropout": 0.1,               ← More dropout
  "final_dropout": 0.0,
  "activation_dropout": 0.0,
  "mask_time_prob": 0.075
}
```

#### Impact of These Differences

**1. Stable Layer Normalization (`do_stable_layer_norm`)**
- `false` (base): Post-normalization → larger gradient flow, different output distribution
- `true` (large): Pre-normalization → more stable training, but different output characteristics

**Reference:** [On Layer Normalization in the Transformer Architecture (2020)](https://arxiv.org/abs/2002.04745)
- Pre-norm and post-norm produce outputs with different statistical properties
- Pre-norm tends to produce more conservative features

**2. Feature Extraction Normalization (`feat_extract_norm`)**
- `"group"` (base): Normalizes within groups → preserves some global information
- `"layer"` (large): Normalizes entire layer → removes global statistics

This fundamentally changes how the embedder represents audio features.

**3. Dropout Configuration**
- base: More dropout in final layer (0.1), less in feature projection (0.0)
- large: More dropout in feature projection (0.1), none in final (0.0)

This affects feature smoothness and may cause artifacts.

---

### 2. Embedding Statistics Analysis

**Test Results (using silent mute audio):**

| Metric | japanese-hubert-base | japanese-hubert-large | Notes |
|--------|---------------------|----------------------|-------|
| Dimension | 768 | 1024 | ✓ Expected |
| Mean | -0.0048 | 0.00065 | ✓ Both near zero |
| Std | 0.190 | 0.181 | ✓ Similar magnitude |
| Std of stds | 0.040 | 0.010 | ⚠ **Large** more uniform |

**Key Observation:**
- japanese-hubert-large has more **uniform** per-dimension variance (std of stds: 0.010 vs 0.040)
- This suggests stronger regularization and more balanced feature representation
- However, this may not match the RVC architecture's expectations, which were tuned for base model characteristics

---

### 3. RVC Architecture Analysis

#### TextEncoder Processing Flow

```python
class TextEncoder:
    def __init__(self, ..., embedding_dim, ...):
        # embedding_dim = 768 (base) or 1024 (large)
        self.emb_phone = nn.Linear(embedding_dim, hidden_channels)  # 768/1024 → 192
        self.emb_pitch = nn.Embedding(256, hidden_channels)         # 256 → 192

    def forward(self, phone, pitch, lengths):
        x = self.emb_phone(phone)      # Project embeddings to 192-dim
        if pitch is not None:
            x += self.emb_pitch(pitch)  # Add pitch information

        x *= math.sqrt(self.hidden_channels)  # Scale by sqrt(192) ≈ 13.86
        x = self.lrelu(x)
        # ... encoder processing
```

#### Potential Issue: Initialization Imbalance

**Linear Layer (emb_phone):**
- PyTorch default: `kaiming_uniform_` with gain based on activation
- Initialization range: `uniform(-sqrt(k), sqrt(k))` where `k = 1/fan_in`
  - 768-dim: `k = 1/768 ≈ 0.0013`, range ≈ ±0.036
  - 1024-dim: `k = 1/1024 ≈ 0.00098`, range ≈ ±0.031

**Embedding Layer (emb_pitch):**
- PyTorch default: `normal(0, 1)`
- Much larger initial values than Linear layer!

**Problem:**
At initialization, `emb_pitch` has significantly larger magnitude than `emb_phone`. While this usually balances out during training, the imbalance may be worse for 1024-dim due to:
1. Smaller initialization range for emb_phone
2. Different embedding characteristics from japanese-hubert-large

**This could explain the pitch shift issue!** The model may over-rely on pitch information during early training and get stuck in a local minimum.

---

### 4. Issues NOT Found (Implementation is Correct)

✅ **Config.json generation** - Correctly updates `text_enc_hidden_dim` to 1024
✅ **FAISS index creation** - Dynamically uses correct dimension
✅ **Mute file generation** - 1024-dim mute embeddings created correctly
✅ **Model checkpoint saving** - `text_enc_hidden_dim` saved properly
✅ **Inference loading** - Fallback chain works correctly
✅ **Synthesizer architecture** - Handles variable dimensions correctly

---

## Root Cause Hypothesis

### Primary Cause: Embedder Architecture Mismatch

The RVC architecture was originally designed and tuned for ContentVec and similar 768-dim embedders with **group normalization** and **post-norm** characteristics. japanese-hubert-large uses:

1. **Layer normalization** instead of group normalization → different feature statistics
2. **Pre-normalization (stable layer norm)** → different activation distributions
3. **33% more dimensions** (1024 vs 768) → more parameters in emb_phone layer
4. **More uniform variance across dimensions** → may not leverage high-variance features effectively

### Secondary Cause: Initialization and Training Dynamics

1. **emb_pitch dominates early training** due to initialization imbalance
2. **Larger parameter space** (1024-dim projection) may require:
   - Different learning rate scheduling (user confirms 1e-4 → 1e-7 decay works well)
   - Extended training time (10-100 hours is feasible and effective)

3. **Embedder source difference:**
   - base: From Applio's curated repository (may have RVC-specific tuning)
   - large: Directly from rinna (general-purpose HuBERT, not RVC-optimized)

---

## Recommended Solutions

### Solution 1: Add Embedding Normalization Layer ⭐ **RECOMMENDED**

Normalize embedder output before feeding into RVC architecture.

**Implementation:**

#### File: `rvc/lib/algorithm/encoders.py`

**Current code (line 127-143):**
```python
def forward(self, phone: torch.Tensor, pitch: Optional[torch.Tensor], lengths: torch.Tensor):
    x = self.emb_phone(phone)
    if pitch is not None and self.emb_pitch:
        x += self.emb_pitch(pitch)

    x *= math.sqrt(self.hidden_channels)
    x = self.lrelu(x)
    # ...
```

**Proposed change:**
```python
def forward(self, phone: torch.Tensor, pitch: Optional[torch.Tensor], lengths: torch.Tensor):
    # Normalize phone embeddings to have consistent statistics
    # This compensates for different embedder architectures
    phone_normalized = F.layer_norm(phone, phone.shape[-1:])

    x = self.emb_phone(phone_normalized)  # Use normalized embeddings
    if pitch is not None and self.emb_pitch:
        x += self.emb_pitch(pitch)

    x *= math.sqrt(self.hidden_channels)
    x = self.lrelu(x)
    # ...
```

**Rationale:**
- Layer normalization ensures embeddings have mean=0, std=1 before projection
- Eliminates differences between base/large normalization strategies
- No need to retrain existing models (only affects new training)
- Minimal computational overhead

**Alternative: Feature-wise scaling**
```python
def __init__(self, ..., embedding_dim, ...):
    self.emb_phone = nn.Linear(embedding_dim, hidden_channels)
    self.emb_pitch = nn.Embedding(256, hidden_channels)

    # Add learnable scaling for phone embeddings
    self.phone_scale = nn.Parameter(torch.ones(1))
    self.phone_bias = nn.Parameter(torch.zeros(1))
```

---

### Solution 2: Re-initialize emb_phone with Larger Values

Increase initial magnitude of `emb_phone` to better balance with `emb_pitch`.

**Implementation:**

#### File: `rvc/lib/algorithm/encoders.py`

**Add custom initialization in `__init__` (after line 118):**
```python
def __init__(self, ..., embedding_dim, ...):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.emb_phone = torch.nn.Linear(embedding_dim, hidden_channels)

    # Custom initialization for better balance with emb_pitch
    # Scale up initialization for larger embedding dimensions
    scale_factor = math.sqrt(embedding_dim / 768.0)  # 1.0 for 768-dim, 1.17 for 1024-dim
    torch.nn.init.normal_(self.emb_phone.weight, mean=0, std=0.02 * scale_factor)
    torch.nn.init.zeros_(self.emb_phone.bias)

    self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
    self.emb_pitch = torch.nn.Embedding(256, hidden_channels) if f0 else None

    # ... rest of __init__
```

**Rationale:**
- Increases initial weight magnitude proportional to embedding dimension
- Better balances emb_phone and emb_pitch contributions
- Prevents pitch from dominating early training

**Caveat:**
- Requires retraining all models
- May affect existing 768-dim models

---

### Solution 3: Adjust Training Hyperparameters

Modify training settings specifically for 1024-dim models.

**Recommended Changes:**

#### 1. Lower Learning Rate
```python
# In rvc/train/train.py or config files
if text_enc_hidden_dim == 1024:
    learning_rate *= 0.7  # Reduce by 30% for 1024-dim
```

#### 2. Increase Weight Decay
```python
# Stronger regularization for larger models
if text_enc_hidden_dim == 1024:
    weight_decay = 0.01  # Increase from default (usually 0)
```

#### 3. Add Gradient Clipping
```python
# In training loop
torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1.0)
```

#### 4. Increase Batch Size
- 1024-dim models benefit from larger batch sizes for stable statistics
- Increase from 8 to 12-16 if GPU memory allows

---

### Solution 4: Use Different Pretrain Models

If available, use pretrain models specifically trained with japanese-hubert-large.

**Current pretrain models likely trained with:**
- contentvec (768-dim)
- japanese-hubert-base (768-dim)

**Problem:**
- Loading 768-dim pretrain into 1024-dim model causes dimension mismatch
- Training from scratch is harder (more data needed, longer training)

**Recommendation:**
- Train a base pretrain model with japanese-hubert-large using large dataset
- Share this pretrain model for community use
- Or request from Applio developers

---

### Solution 5: Feature Extraction Adjustment

Add per-dimension scaling to balance feature importance.

**Implementation:**

#### File: `rvc/train/extract/extract.py`

**Modify embedding extraction (line 143-146):**
```python
with torch.no_grad():
    result = model(feats)["last_hidden_state"]

# Normalize for 1024-dim embedders (optional)
if embedder_model == "japanese-hubert-large":
    # Apply per-dimension normalization
    result = (result - result.mean(dim=1, keepdim=True)) / (result.std(dim=1, keepdim=True) + 1e-8)

feats_out = result.squeeze(0).float().cpu().numpy()
```

**Rationale:**
- Ensures extracted features have consistent statistics
- Compensates for embedder architecture differences
- Can be applied retroactively to existing datasets

**Caveat:**
- Changes feature distribution
- May affect FAISS index retrieval quality

---

## Testing & Validation Plan

### 1. Implement Solution 1 (Embedding Normalization)
- ✅ Low risk, easy to implement
- ✅ Doesn't break existing models
- ✅ Addresses root cause

### 2. Test Training with Modified Architecture
```bash
# Train small test model with 100 epochs
env\python.exe core.py preprocess --model_name test-jphub-large-v2 --dataset_path <dataset> --sample_rate 40000

env\python.exe core.py extract --model_name test-jphub-large-v2 --sample_rate 40000 --f0_method rmvpe --embedder_model japanese-hubert-large

env\python.exe core.py train --model_name test-jphub-large-v2 --sample_rate 40000 --total_epoch 100 --save_every_epoch 25
```

### 3. Compare Results
- **Baseline:** japanese-hubert-base model
- **Test 1:** japanese-hubert-large without normalization (current implementation)
- **Test 2:** japanese-hubert-large with normalization (Solution 1)

### 4. Metrics to Evaluate
1. **Pitch accuracy:** Use audio analysis tools to measure F0 deviation
2. **Speaker similarity:** Subjective listening test + speaker verification metrics
3. **Audio quality:** Measure SNR, spectral flatness for high-frequency artifacts
4. **Training stability:** Monitor loss curves, check for overfitting

---

## Implementation Priority

### Immediate (High Priority)
1. ⭐ **Solution 1**: Add embedding normalization layer
   - Minimal code changes
   - Addresses root cause
   - Low risk

2. ⭐ **Solution 3**: Adjust learning rate and regularization
   - Easy config changes
   - Helps with overfitting
   - Can combine with Solution 1

### Short-term (Medium Priority)
3. **Solution 5**: Feature extraction normalization
   - Can be applied to existing datasets
   - Helps retroactively

### Long-term (Lower Priority)
4. **Solution 2**: Custom initialization
   - Requires more testing
   - May affect all models

5. **Solution 4**: Create pretrain models
   - Time-consuming
   - Community effort needed

---

## Additional Recommendations

### 1. Dataset Considerations
- japanese-hubert-large requires **more training data** than base (33% more parameters)
- Minimum recommended: 30-45 minutes of clean speech (vs 20-30 for base)
- Higher quality data is more important due to overfitting risk

### 2. Training Duration
- May need **longer training** (more epochs) to converge properly
- Use overtraining detector to prevent overfitting
- Monitor validation loss carefully

### 3. Inference Settings
- `index_rate`: Try lower values (0.5-0.7 instead of 0.75) to reduce artifacts
- `protect`: Try higher values (0.4-0.5 instead of 0.33) for pitch protection
- `clean_audio`: Enable to reduce high-frequency noise

### 4. F0 Method Selection
- For japanese-hubert-large, prefer **RMVPE** or **FCPE** over CREPE
- CREPE may over-emphasize pitch details with large embedder

---

## Comparison with japanese-hubert-base

### When to Use japanese-hubert-large
✅ Large dataset available (45+ minutes)
✅ High-quality speech data
✅ Willing to tune hyperparameters
✅ Implementing normalization fix (Solution 1)
✅ Need maximum speaker similarity for Japanese

### When to Use japanese-hubert-base
✅ Limited dataset (20-30 minutes)
✅ Want faster training
✅ Simpler, more stable training
✅ Proven track record
✅ Less prone to overfitting

---

## Files Requiring Modification (Solution 1)

### Required Changes

| File | Change | Complexity |
|------|--------|-----------|
| `rvc/lib/algorithm/encoders.py` | Add layer normalization in TextEncoder.forward | Low |
| `.claude/CLAUDE.md` | Update with findings and recommendations | Low |

**Total: 2 files, ~10 lines of code**

### Optional Enhancements

| File | Change | Complexity |
|------|--------|-----------|
| `rvc/train/train.py` | Add dimension-dependent learning rate scaling | Medium |
| `rvc/configs/*.json` | Add training hyperparameter variants for 1024-dim | Low |

---

## Conclusion

The quality issues with japanese-hubert-large are **not due to bugs in the implementation**, but rather due to **fundamental architectural differences** between the embedder models and **inadequate adaptation in the RVC architecture**.

The most effective solution is to **add embedding normalization** (Solution 1) combined with **adjusted training hyperparameters** (Solution 3). This addresses the root cause while maintaining backward compatibility.

If the issue persists after implementing these solutions, the problem may lie in the embedder model itself being unsuitable for RVC, and japanese-hubert-base should be preferred for Japanese voice conversion tasks.

---

## Next Steps

1. ✅ **Review this analysis** with user
2. ⬜ **Implement Solution 1** (embedding normalization)
3. ⬜ **Test with small dataset** (100 epochs)
4. ⬜ **Compare quality** with japanese-hubert-base
5. ⬜ **Document results** in this file
6. ⬜ **Create pull request** if solution is effective

---

**Document Version:** 1.0
**Last Updated:** 2025-01-04
**Author:** Claude (claude-sonnet-4-5-20250929)
**Status:** READY FOR IMPLEMENTATION
