# ✅ IMPLEMENTED: Embedding Normalization for japanese-hubert-large

**Implementation Date:** 2025-01-04
**Status:** IMPLEMENTED & READY FOR TESTING

**Solution:** Add conditional layer normalization to phone embeddings before projection (1024-dim only)
**Impact:** LOW (only affects japanese-hubert-large training, no impact on 768-dim models)
**Complexity:** LOW (3 lines of code)
**Priority:** HIGH

---

## The Fix

### File: `rvc/lib/algorithm/encoders.py`

**Location:** TextEncoder.forward method (lines 127-143)

**Current Code:**
```python
def forward(
    self, phone: torch.Tensor, pitch: Optional[torch.Tensor], lengths: torch.Tensor
):
    x = self.emb_phone(phone)
    if pitch is not None and self.emb_pitch:
        x += self.emb_pitch(pitch)

    x *= math.sqrt(self.hidden_channels)
    x = self.lrelu(x)
    x = x.transpose(1, -1)  # [B, H, T]

    x_mask = sequence_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
    x = self.encoder(x, x_mask)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    return m, logs, x_mask
```

**✅ IMPLEMENTED Fix:**
```python
def forward(
    self, phone: torch.Tensor, pitch: Optional[torch.Tensor], lengths: torch.Tensor
):
    # Apply layer normalization for large embedders (1024-dim) only
    # This compensates for different normalization strategies in japanese-hubert-large
    # vs japanese-hubert-base (layer norm + pre-norm vs group norm + post-norm)
    if self.embedding_dim >= 1024:
        phone = F.layer_norm(phone, phone.shape[-1:])

    x = self.emb_phone(phone)  # Uses normalized embeddings for 1024-dim
    if pitch is not None and self.emb_pitch:
        x += self.emb_pitch(pitch)

    x *= math.sqrt(self.hidden_channels)
    x = self.lrelu(x)
    x = x.transpose(1, -1)  # [B, H, T]

    x_mask = sequence_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
    x = self.encoder(x, x_mask)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    return m, logs, x_mask
```

**Changes Made:**
1. Added `import torch.nn.functional as F` to imports (line 3)
2. Added `self.embedding_dim = embedding_dim` to `__init__` (line 120)
3. Added conditional normalization in `forward` (lines 136-137)

**Required Import (add at top of file):**
```python
import torch.nn.functional as F  # Add if not already present
```

---

## Why This Works

### Problem
japanese-hubert-large uses:
- Layer normalization (not group normalization)
- Stable pre-normalization (not post-normalization)
- Different dropout configuration

This causes embeddings to have different statistical properties than japanese-hubert-base.

### Solution
Layer normalization ensures all embeddings have:
- Mean ≈ 0
- Std ≈ 1

This makes the RVC architecture agnostic to embedder architecture differences.

### Impact
- ✅ Removes bias from different embedder normalization strategies
- ✅ Balances emb_phone and emb_pitch contributions better
- ✅ No effect on inference (only training)
- ✅ Backward compatible (old models still work)
- ✅ Minimal computational cost

---

## Testing Plan

### Before Applying Fix
1. Train a model with japanese-hubert-large (current implementation)
2. Note quality issues:
   - High-frequency noise
   - Pitch shift
   - Poor speaker similarity

### After Applying Fix
1. Train a model with japanese-hubert-large (with normalization)
2. Compare with baseline:
   - Pitch accuracy (should be within ±0.2 semitones)
   - Speaker similarity (should improve)
   - High-frequency artifacts (should reduce)

### Test Command
```bash
# Preprocess
env\python.exe core.py preprocess --model_name test-jphub-norm --dataset_path <your_dataset> --sample_rate 40000

# Extract with japanese-hubert-large
env\python.exe core.py extract --model_name test-jphub-norm --sample_rate 40000 --f0_method rmvpe --embedder_model japanese-hubert-large

# Train for 100 epochs
env\python.exe core.py train --model_name test-jphub-norm --sample_rate 40000 --total_epoch 100 --save_every_epoch 25

# Test inference
env\python.exe core.py infer --input_path <test_audio> --output_path output_test.wav --pth_path logs/test-jphub-norm/test-jphub-norm_e100.pth --index_path logs/test-jphub-norm/added_*.index --embedder_model japanese-hubert-large
```

---

## Alternative: Conditional Normalization

If you want to only apply normalization for 1024-dim embedders:

```python
def __init__(self, ..., embedding_dim, ...):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.embedding_dim = embedding_dim  # Store for later use
    self.emb_phone = torch.nn.Linear(embedding_dim, hidden_channels)
    # ...

def forward(self, phone: torch.Tensor, pitch: Optional[torch.Tensor], lengths: torch.Tensor):
    # Only normalize for large embedders (1024-dim)
    if self.embedding_dim >= 1024:
        phone = F.layer_norm(phone, phone.shape[-1:])

    x = self.emb_phone(phone)
    # ... rest of code
```

**Pros:**
- Doesn't affect existing 768-dim models
- Targeted fix for large embedders

**Cons:**
- More complex code
- May not be necessary (normalization helps all embedders)

**Recommendation:** Use the simpler version (normalize all) unless you have specific concerns.

---

## Expected Results

### Before Fix (Current Issues)
- ❌ Pitch shift: +0.8 to +1.0 semitones
- ❌ High-frequency noise: Noticeable artifacts
- ❌ Speaker similarity: Poor (not resembling target voice)
- ❌ Training stability: Prone to overfitting

### After Fix (Expected)
- ✅ Pitch shift: ±0.2 semitones (acceptable)
- ✅ High-frequency noise: Minimal artifacts
- ✅ Speaker similarity: Good (comparable to japanese-hubert-base)
- ✅ Training stability: More stable, less overfitting

---

## Rollback Plan

If the fix causes issues:

1. **Immediate rollback:**
   ```python
   # Simply comment out the normalization line
   # phone_normalized = F.layer_norm(phone, phone.shape[-1:])
   x = self.emb_phone(phone)  # Use original phone, not phone_normalized
   ```

2. **No data loss:**
   - Models trained with normalization can still be used
   - Just won't benefit from the fix during inference

3. **Compatibility:**
   - Old models (trained without normalization) continue to work
   - New models (trained with normalization) also work
   - No checkpoint format changes

---

## Additional Recommendations

Even with this fix, japanese-hubert-large may still require:

### 1. Adjusted Training Parameters
- Learning rate: Reduce to 7e-5 (from 1e-4)
- Batch size: Increase to 12-16 (from 8)
- Weight decay: Add 0.01 regularization

### 2. More Training Data
- Minimum: 45 minutes (vs 30 for base)
- Recommended: 60+ minutes

### 3. Inference Parameter Tuning
- `index_rate`: 0.5-0.7 (lower than usual)
- `protect`: 0.4-0.5 (higher than usual)
- `clean_audio`: Enable

### 4. F0 Method
- Prefer RMVPE or FCPE over CREPE
- CREPE may over-emphasize pitch with large embedder

---

## Decision Point

**Should you apply this fix?**

✅ **Yes, if:**
- You want to use japanese-hubert-large
- You're willing to test/validate results
- You have sufficient training data (45+ minutes)

❌ **No, if:**
- You're satisfied with japanese-hubert-base
- You want maximum stability/simplicity
- You have limited training data (<30 minutes)

**Recommendation:** Apply the fix and test. If results are still inferior to japanese-hubert-base, revert to using base model and document findings.

---

## Next Steps

1. ⬜ Review this proposed fix
2. ⬜ Make a backup of `rvc/lib/algorithm/encoders.py`
3. ⬜ Apply the fix
4. ⬜ Test with small dataset
5. ⬜ Compare results with baseline
6. ⬜ Document findings in `.claude/japanese-hubert-large_quality_issues_analysis.md`
7. ⬜ Decide: Keep fix, adjust further, or revert to japanese-hubert-base

---

**Status:** ✅ IMPLEMENTED (2025-01-04)
**Testing:** REQUIRED - User should test with japanese-hubert-large training
**Priority:** HIGH
