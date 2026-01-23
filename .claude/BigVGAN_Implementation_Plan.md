# BigVGAN-v2 Integration Plan for Applio

This document contains the complete implementation plan for integrating NVIDIA's BigVGAN-v2 vocoder into Applio.

## 1. Overview

### 1.1 What is BigVGAN?
BigVGAN is a neural vocoder from NVIDIA that converts mel-spectrograms to audio waveforms. Key features:
- **Anti-aliased periodic activation (Snake/SnakeBeta)**: Trainable periodic activation functions
- **AMP Blocks**: Anti-aliased Multi-Periodicity residual blocks
- **High quality audio synthesis** at various sample rates

### 1.2 BigVGAN-v2 Models Available
| Model | Sample Rate | Hop Size | Mel Bins | Upsample Ratio |
|-------|-------------|----------|----------|----------------|
| bigvgan_v2_44khz_128band_256x | 44100 Hz | 256 | 128 | 256x |
| bigvgan_v2_44khz_128band_512x | 44100 Hz | 512 | 128 | 512x |
| bigvgan_v2_24khz_100band_256x | 24000 Hz | 256 | 100 | 256x |
| bigvgan_v2_22khz_80band_256x | 22050 Hz | 256 | 80 | 256x |

**We will implement the 44kHz 256x model** as it provides the best quality for RVC voice conversion.

## 2. Architecture Analysis

### 2.1 BigVGAN-v2 44kHz 256x Configuration
```json
{
    "sampling_rate": 44100,
    "hop_size": 256,
    "n_fft": 1024,
    "win_size": 1024,
    "num_mels": 128,
    "upsample_rates": [4, 4, 2, 2, 2, 2],
    "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
    "upsample_initial_channel": 1536,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "activation": "snakebeta",
    "snake_logscale": true
}
```

**Upsample calculation**: 4 × 4 × 2 × 2 × 2 × 2 = 256 (matches hop_size)

### 2.2 BigVGAN Generator Architecture
```
Input: Mel-spectrogram [B, num_mels=128, T]
    ↓
conv_pre: Conv1d(128 → 1536, k=7, p=3)
    ↓
For each upsample stage (6 stages):
    ↓ ConvTranspose1d (upsample)
    ↓ AMPBlock × 3 (resblock_kernel_sizes=[3,7,11])
    ↓
activation_post: Activation1d(SnakeBeta)
    ↓
conv_post: Conv1d(ch → 1, k=7, p=3)
    ↓
tanh or clamp to [-1, 1]
    ↓
Output: Audio waveform [B, 1, T × 256]
```

### 2.3 Key Components

#### Snake/SnakeBeta Activation
```python
# Snake: x + 1/α × sin²(xα)
# SnakeBeta: x + 1/β × sin²(xα)
# α, β are trainable parameters (per-channel)
```

#### AMPBlock (Anti-aliased Multi-Periodicity Block)
- Uses Snake/SnakeBeta activation wrapped in Activation1d
- Activation1d: Upsample → Apply activation → Downsample (anti-aliasing)
- AMPBlock1: Two conv layers per dilation (conv1 + conv2)
- AMPBlock2: One conv layer per dilation

#### Alias-Free Activation (Activation1d)
```python
def forward(x):
    x = upsample(x)      # 2x upsample with Kaiser filter
    x = activation(x)     # Snake/SnakeBeta
    x = downsample(x)     # 2x downsample with Kaiser filter
    return x
```

## 3. Applio Integration Analysis

### 3.1 Applio's Vocoder Interface

Current vocoders in Applio:
| Vocoder | F0 Required | Sample Rates | Input |
|---------|-------------|--------------|-------|
| HiFi-GAN | No | 32k, 40k, 48k | Latent z |
| HiFi-GAN NSF | Yes | 32k, 40k, 48k | Latent z + F0 |
| MRF HiFi-GAN | Yes | 32k, 40k, 48k | Latent z + F0 |
| RefineGAN | Yes | 32k only | Mel + F0 |

**BigVGAN**: Takes mel-spectrogram input, no F0 required

### 3.2 Key Differences from Applio's Architecture

1. **Input Type**:
   - Applio vocoders: Latent space z (inter_channels=192)
   - BigVGAN: Mel-spectrogram (num_mels=128)

2. **F0 Conditioning**:
   - Applio: Most vocoders use F0 for pitch guidance
   - BigVGAN: No F0 input (pure mel-to-audio)

3. **Sample Rate**:
   - Applio: 32kHz, 40kHz, 48kHz
   - BigVGAN-v2: 44.1kHz (need to add new config)

### 3.3 Integration Strategy

**Option A: Adapt BigVGAN to accept latent z**
- Add conv_pre layer to convert 192 → 128 channels
- Train from scratch with Applio's pipeline
- Pros: Maintains RVC architecture consistency
- Cons: Cannot use NVIDIA's pretrained weights

**Option B: Use BigVGAN with mel-spectrogram input** (Recommended)
- Skip the PosteriorEncoder → use mel directly
- Requires significant pipeline changes
- Pros: Can potentially use pretrained weights
- Cons: Major architectural change

**Option C: BigVGAN as F0-aware vocoder** (Hybrid Approach)
- Similar to HiFi-GAN NSF: add F0 source module
- Input: latent z (converted to 128 channels) + F0
- Train from scratch
- Pros: Best integration with RVC, maintains F0 pitch guidance
- This is the approach we will take

## 4. Implementation Plan

### 4.1 Files to Create

1. **`rvc/lib/algorithm/generators/bigvgan.py`**
   - BigVGANGenerator class (adapted for Applio)
   - Snake, SnakeBeta activation classes
   - Activation1d, UpSample1d, DownSample1d classes
   - AMPBlock1, AMPBlock2 classes

2. **`rvc/configs/44100.json`**
   - New config for 44.1kHz sample rate
   - Matched to BigVGAN's architecture

### 4.2 Files to Modify

1. **`rvc/lib/algorithm/synthesizers.py`**
   - Add BigVGAN vocoder selection
   - Line ~85: Add `elif vocoder == "BigVGAN":`

2. **`rvc/lib/tools/pretrained_selector.py`**
   - Add BigVGAN folder mapping
   - Add pretrained model path selection

3. **`rvc/lib/tools/prerequisites_download.py`**
   - Add BigVGAN pretrained model download URLs
   - Note: We'll need to create custom pretraineds for RVC

4. **`rvc/train/train.py`**
   - Add BigVGAN discriminator version selection
   - Line ~449: Add BigVGAN case

5. **`tabs/train/train.py`**
   - Add "BigVGAN" to vocoder choices
   - Add sample rate restriction for BigVGAN (44100 only)

6. **`rvc/infer/infer.py`** (if needed)
   - Verify BigVGAN loading works in inference

### 4.3 BigVGANGenerator Implementation

```python
class BigVGANGenerator(torch.nn.Module):
    """
    BigVGAN Generator adapted for Applio RVC.

    Differences from original BigVGAN:
    1. Input: latent z (inter_channels) instead of mel-spectrogram
    2. Optional F0 conditioning via source module
    3. Speaker conditioning (g) support
    """

    def __init__(
        self,
        initial_channel: int,      # 192 (inter_channels)
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,  # 1536 for BigVGAN
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
        checkpointing: bool = False,
    ):
        # ... implementation

    def forward(self, x, f0=None, g=None):
        # x: latent features [B, 192, T]
        # f0: pitch (optional) [B, T']
        # g: speaker embedding [B, 256, 1]
        # ... implementation
```

### 4.4 44100.json Configuration

```json
{
  "train": {
    "log_interval": 200,
    "seed": 1234,
    "learning_rate": 1e-4,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "lr_decay": 0.999875,
    "segment_size": 16384,
    "c_mel": 45,
    "c_kl": 1.0
  },
  "data": {
    "max_wav_value": 32768.0,
    "sample_rate": 44100,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 128,
    "mel_fmin": 0.0,
    "mel_fmax": null
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "text_enc_hidden_dim": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0,
    "resblock": "1",
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "upsample_rates": [4,4,2,2,2,2],
    "upsample_initial_channel": 1536,
    "upsample_kernel_sizes": [8,8,4,4,4,4],
    "use_spectral_norm": false,
    "gin_channels": 256,
    "spk_embed_dim": 109
  }
}
```

## 5. Pretrained Models

### 5.1 The Pretrained Model Problem

**NVIDIA's pretrained BigVGAN models cannot be directly used** because:
1. They expect mel-spectrogram input (128 channels)
2. Applio's RVC uses latent space z (192 channels)
3. Different input/output dimensionality

### 5.2 Solutions

**Option 1: Train from scratch** (Recommended for initial implementation)
- Use Applio's training pipeline
- Requires training D and G models together
- Quality will improve with training data

**Option 2: Transfer learning**
- Initialize AMP blocks from NVIDIA's pretrained
- Train only input/output adaptation layers
- Requires custom adaptation code

**Option 3: Create adapter layers**
- Add projection layer: 192 → 128 → BigVGAN
- May lose information in dimension reduction

**We will implement Option 1** initially, with potential for Option 2 later.

### 5.3 Pretrained Storage

Store pretrained models in:
```
rvc/models/pretraineds/bigvgan/
├── f0G44100.pth    # Generator pretrained
└── f0D44100.pth    # Discriminator pretrained
```

## 6. UI Changes

### 6.1 Training Tab Modifications

In `tabs/train/train.py`:

```python
# Add BigVGAN to vocoder choices
vocoder = gr.Radio(
    label=i18n("Vocoder"),
    choices=["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN", "BigVGAN"],
    value="HiFi-GAN",
)

# Add sample rate restriction callback
def update_sample_rate_for_vocoder(vocoder):
    if vocoder == "BigVGAN":
        return gr.update(choices=["44100"], value="44100")
    elif vocoder == "RefineGAN":
        return gr.update(choices=["32000"], value="32000")
    else:
        return gr.update(choices=["32000", "40000", "48000"], value="40000")
```

### 6.2 Sample Rate Display

Note: Applio currently shows 32k/40k/48k. We need to:
1. Add 44100 as a new sample rate option
2. Show 44100 only when BigVGAN is selected

## 7. Testing Plan

### 7.1 Unit Tests
1. Test BigVGANGenerator forward pass
2. Test AMPBlock forward/backward
3. Test Snake/SnakeBeta activation
4. Test Activation1d anti-aliasing

### 7.2 Integration Tests
1. Training: Verify model trains without errors
2. Checkpoint: Verify model saves/loads correctly
3. Inference: Verify voice conversion works
4. Real-time: Verify real-time inference works

### 7.3 Quality Tests
1. Compare output quality with HiFi-GAN
2. Test on various input voices
3. Measure latency for real-time conversion

## 8. Known Limitations

1. **Sample rate**: BigVGAN-v2 supports up to 44.1kHz only
2. **Training data**: Requires RVC-format training data
3. **No pretrained**: Must train from scratch initially
4. **CUDA kernels**: Optional optimization not implemented

## 9. Future Improvements

1. **CUDA kernel support**: Implement optimized CUDA kernels for inference
2. **Transfer learning**: Use NVIDIA's pretrained weights for initialization
3. **Multi-sample-rate**: Adapt BigVGAN for 32k/40k/48k if needed
4. **CQTD discriminator**: Optionally add CQT discriminator from BigVGAN

## 10. References

- [NVIDIA BigVGAN GitHub](https://github.com/NVIDIA/BigVGAN)
- [BigVGAN Paper (ICLR 2023)](https://arxiv.org/abs/2206.04658)
- [HuggingFace Models](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_256x)
- [Alias-Free GAN](https://github.com/junjun3518/alias-free-torch)

---

## Appendix A: Complete BigVGAN Code Reference

### A.1 Snake Activation
```python
class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x
```

### A.2 SnakeBeta Activation
```python
class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x
```

### A.3 Kaiser-Sinc Filter
```python
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Kaiser window parameters
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # Sinc filter
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        filter_ /= filter_.sum()

    return filter_.view(1, 1, kernel_size)
```

---

**Document Version**: 1.0
**Created**: 2026-01-19
**Author**: Claude Code Assistant
