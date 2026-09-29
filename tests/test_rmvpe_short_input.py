"""RMVPE-family pitch on clips shorter than half of RMVPE's 32-frame padding block."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rvc.lib.predictors.RMVPE import _reflect_pad_right


@pytest.mark.parametrize("n_frames", range(17, 70))
def test_padding_is_unchanged_where_a_single_reflection_fits(n_frames):
    mel = torch.randn(1, 128, n_frames)
    n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
    expected = F.pad(mel, (0, n_pad), mode="reflect") if n_pad else mel
    assert torch.equal(_reflect_pad_right(mel, n_pad), expected)


@pytest.mark.parametrize("n_frames", [1, 2, 5, 14, 16])
def test_short_input_is_padded_to_the_block(n_frames):
    mel = torch.randn(1, 128, n_frames)
    n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
    padded = _reflect_pad_right(mel, n_pad)
    assert padded.shape[-1] == 32
    assert torch.equal(padded[..., :n_frames], mel)


@pytest.mark.skipif(
    not Path("rvc/models/predictors/rmvpe.pt").is_file() or not torch.cuda.is_available(),
    reason="RMVPE weights and CUDA required",
)
# Under 1024 samples (64 ms) the STFT's own centring pad fails first; preprocessing never
# produces clips that short, so that limit is left alone.
@pytest.mark.parametrize("samples", [1100, 2081, 2400])
def test_rmvpe_handles_a_130_ms_clip(samples):
    from rvc.lib.predictors.f0 import RMVPE

    t = np.arange(samples) / 16000
    x = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    f0 = RMVPE(device="cuda", sample_rate=16000, hop_size=160).get_f0(x, filter_radius=0.03)
    assert len(f0) == samples // 160 + 1


@pytest.mark.skipif(
    not Path("rvc/models/predictors/hpa-rmvpe-112000.pt").is_file() or not torch.cuda.is_available(),
    reason="HPA-RMVPE weights and CUDA required",
)
def test_hpa_aligned_filled_extracts_a_130_ms_clip():
    from rvc.train.extract.extract import FeatureInput

    t = np.arange(2081) / 16000
    x = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    f0 = FeatureInput("hpa-rmvpe-112000-aligned-filled", "cuda").compute_f0(x)
    assert len(f0) == 2081 // 160 + 1
