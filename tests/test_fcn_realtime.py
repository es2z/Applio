import numpy as np
import pytest
import torch
import torchaudio.transforms as tat

from rvc.lib.predictors.fcn.adapter import DEFAULT_WEIGHT, FCNPredictor
from rvc.realtime.fcn_session import CaptureResampler, FCNRealtimeSession

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_capture_resampling_irregular_chunks_preserves_phase():
    resampler = tat.Resample(48000, 16000).cuda()
    stream = CaptureResampler(resampler, "cuda")
    audio = torch.tensor(
        np.random.default_rng(993).normal(size=20003).astype(np.float32), device="cuda"
    )
    pieces, start = [], 0
    for size in [1, 2, 20, 47, 719, 15, 997, 8000, 2000, 8202]:
        pieces.append(stream.push(audio[start : start + size]))
        start += size
    pieces.append(stream.push(audio[start:], final=True))
    torch.testing.assert_close(
        torch.cat(pieces), resampler(audio), atol=1e-6, rtol=1e-5
    )
    assert len(stream.buffer) < stream.context + 6


def test_delayed_audio_and_pitch_share_absolute_grid():
    if not DEFAULT_WEIGHT.exists():
        pytest.skip("Local converted weights required")
    predictor = FCNPredictor("cuda")
    resampler = tat.Resample(48000, 16000).cuda()
    session = FCNRealtimeSession(predictor, resampler, 8000)
    audio = 0.1 * torch.sin(
        torch.arange(48001, device="cuda") * (2 * torch.pi * 220 / 48000)
    )
    reference_audio = resampler(audio)
    reference_pitch = predictor.extract_track(reference_audio).pitch_hz
    pos = 0
    for size in [31, 200, 8000, 1547, 20000, 18223]:
        window, pitch = session.push(audio[pos : pos + size])
        pos += size
        end = session.window_end
        count = min(8000, end)
        if count:
            torch.testing.assert_close(
                window[-count:],
                reference_audio[end - count : end],
                atol=1e-6,
                rtol=1e-5,
            )
            frames = count // 160
            expected = reference_pitch[end // 160 - frames : end // 160]
            actual = pitch[-frames:].cpu().numpy()
            # The noncausal wrap startup discrepancy is only at global frame 0.
            skip = 1 if end <= 8000 else 0
            assert np.max(np.abs(1200 * np.log2(actual[skip:] / expected[skip:]))) < 0.1
        assert (
            len(session.audio)
            < session.window_samples + session.stream.holdback_samples + 160
        )
    session.reset()
    assert session.window_end == session.stream.sample_count == 0
    assert not len(session.audio)
