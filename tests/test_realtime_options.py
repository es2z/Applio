"""Opt-in realtime options (rvc/realtime/options.py), all off by default.

The identity tests replace the voice conversion with "return the window you were
given", so the delay between input and output is the window layout's alone - no audio
device, no processing time. That is the number docs/realtime-latency-quality.md quotes.
"""

import unittest
from unittest.mock import patch

import numpy as np
import torch
import torchaudio.transforms as tat

from rvc.realtime import core
from rvc.realtime.options import FIELD_ORDER, RealtimeOptions
from rvc.realtime.resample import StatefulResampler


class IdentityPipeline:
    """Returns the 16 kHz window it is given at 48 kHz, trimmed like the generator."""

    def __init__(self, options):
        self.tgt_sr = 48000
        self.device = "cpu"
        self.f0_method = "rmvpe"
        self.options = options
        self.up = tat.Resample(16000, 48000, dtype=torch.float32)
        self.stream = None

    def configure_stream(self, p_len, skip_head, return_length, shift, trim):
        self.stream = dict(
            p_len=p_len, skip_head=skip_head, return_length=return_length,
            shift=shift, trim=trim,
        )

    def reset_stream_state(self):
        pass

    def voice_conversion(self, audio, pitch, pitchf, f0_up_key, index_rate, p_len,
                         silence_front, skip_head, return_length, *args):
        out = self.up(audio.clone())[: p_len * 480]
        if self.stream["trim"]:
            out = out[-return_length * 480:]
        return out


def make_changer(options, chunk_ms=960, crossfade=0.138, extra=0.22):
    def fake_create_pipeline(**kwargs):
        return IdentityPipeline(kwargs.get("options") or RealtimeOptions())

    with patch.object(core, "create_pipeline", side_effect=fake_create_pipeline):
        return core.VoiceChanger(
            int(chunk_ms * 48000 / 1000 / 128), crossfade, extra,
            silent_threshold=0, options=options,
        )


def measure_delay_ms(changer, seconds=8.0):
    block = changer.block_frame
    n = int(seconds * 48000) // block * block
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).astype(np.float32)
    # Band-limit so the 48 -> 16 -> 48 kHz round trip keeps it intact.
    x = tat.Resample(16000, 48000)(tat.Resample(48000, 16000)(torch.from_numpy(x)))
    x = (x.numpy()[:n] * 0.1).astype(np.float32)
    y = np.concatenate([changer.on_request(x[i:i + block])[0] for i in range(0, n, block)])
    from scipy.signal import correlate

    half = n // 2
    seg = y[half: half + 48000 * 2]
    span = 48000 * 2
    ref = x[half - span: half + 48000 * 2]
    # Lag L means y[t] matches x[t - L]; search 0..2 s by FFT cross-correlation.
    scores = correlate(ref, seg, mode="valid", method="fft")[::-1]
    best = int(np.argmax(scores))
    match = x[half - best: half - best + 48000 * 2]
    corr = np.dot(seg, match) / (np.linalg.norm(seg) * np.linalg.norm(match) + 1e-9)
    return best / 48.0, corr


class OptionsTests(unittest.TestCase):
    def test_defaults_are_the_legacy_behaviour(self):
        options = RealtimeOptions()
        self.assertEqual(options.output_position, "legacy")
        self.assertEqual(options.transport, "legacy")
        self.assertFalse(options.needs_realtime_synthesis)
        self.assertEqual(options.changed(), [])

    def test_from_dict_is_tolerant_and_round_trips(self):
        options = RealtimeOptions.from_dict(
            {"output_position": "trim", "lookahead_s": "0.05", "gain_mode": "nope",
             "apex_noise_voiced_ratio": 3, "unknown": 1}
        )
        self.assertEqual(options.output_position, "trim")
        self.assertAlmostEqual(options.lookahead_s, 0.05)
        self.assertEqual(options.gain_mode, "legacy")
        self.assertEqual(options.apex_noise_voiced_ratio, 1.0)
        self.assertEqual(RealtimeOptions.from_dict(options.to_dict()), options)
        self.assertEqual(RealtimeOptions.from_values(*options.values()), options)
        self.assertEqual(len(options.values()), len(FIELD_ORDER))


class StatefulResamplerTests(unittest.TestCase):
    def test_streaming_equals_the_whole_signal_delayed(self):
        resampler = tat.Resample(48000, 16000, dtype=torch.float32)
        stream = StatefulResampler(resampler)
        torch.manual_seed(0)
        x = torch.randn(48000 * 2)
        whole = resampler(torch.cat((torch.zeros(stream.delay_samples), x)))
        pieces = [stream.push(x[i:i + 4800]) for i in range(0, x.shape[0], 4800)]
        streamed = torch.cat(pieces)
        self.assertEqual(streamed.shape[0], x.shape[0] // 3)
        # Away from the start (where the zero history differs from zero padding).
        error = (streamed[100:] - whole[100: streamed.shape[0]]).abs().max().item()
        self.assertLess(error, 1e-5)

    def test_stateless_blocks_are_what_it_fixes(self):
        resampler = tat.Resample(48000, 16000, dtype=torch.float32)
        torch.manual_seed(0)
        x = torch.randn(48000 * 2)
        whole = resampler(x)
        blocks = torch.cat([resampler(x[i:i + 4800]) for i in range(0, x.shape[0], 4800)])
        self.assertGreater((blocks - whole).abs().max().item(), 1e-2)


class WindowLayoutTests(unittest.TestCase):
    def test_legacy_plays_the_oldest_part_of_the_window(self):
        changer = make_changer(RealtimeOptions())
        delay, corr = measure_delay_ms(changer)
        self.assertGreater(corr, 0.99)
        self.assertAlmostEqual(delay, 370.0, delta=1.0)
        self.assertAlmostEqual(changer.vc_model.algorithmic_delay_ms, 370.0, delta=1.0)

    def test_extra_is_look_ahead_in_legacy_and_free_context_in_trim(self):
        legacy = make_changer(RealtimeOptions(), extra=1.0)
        trim = make_changer(RealtimeOptions(output_position="trim"), extra=1.0)
        legacy_delay, _ = measure_delay_ms(legacy)
        trim_delay, corr = measure_delay_ms(trim)
        self.assertAlmostEqual(legacy_delay, 1150.0, delta=1.0)
        self.assertGreater(corr, 0.99)
        self.assertAlmostEqual(trim_delay, 150.0, delta=1.0)
        self.assertAlmostEqual(trim.vc_model.algorithmic_delay_ms, 150.0, delta=1.0)

    def test_look_ahead_adds_one_for_one_and_only_in_trim(self):
        options = RealtimeOptions(output_position="trim", lookahead_s=0.05)
        changer = make_changer(options, chunk_ms=240, crossfade=0.08, extra=2.5)
        delay, corr = measure_delay_ms(changer)
        self.assertGreater(corr, 0.99)
        self.assertAlmostEqual(delay, 140.0, delta=1.0)
        self.assertAlmostEqual(changer.vc_model.algorithmic_delay_ms, 140.0, delta=1.0)
        ignored = make_changer(RealtimeOptions(lookahead_s=0.05))
        self.assertAlmostEqual(measure_delay_ms(ignored)[0], 370.0, delta=1.0)

    def test_trim_renders_only_the_newest_part(self):
        changer = make_changer(RealtimeOptions(output_position="trim"), extra=2.5)
        runtime = changer.vc_model
        stream = runtime.pipeline.stream
        self.assertTrue(stream["trim"])
        self.assertEqual(stream["skip_head"] + stream["return_length"], stream["p_len"])
        # block 960 + crossfade 138 + SOLA 10 ms, rounded up to whole 10 ms frames.
        self.assertEqual(stream["return_length"], 111)
        self.assertEqual(stream["shift"], 96)

    def test_stateful_input_keeps_the_layout(self):
        options = RealtimeOptions(output_position="trim", input_resample="stateful")
        changer = make_changer(options)
        self.assertIsNotNone(changer.vc_model.stateful_in)
        delay, corr = measure_delay_ms(changer)
        self.assertGreater(corr, 0.99)
        # A few samples of resampler delay on top of the layout's 150 ms.
        self.assertAlmostEqual(delay, 150.0 + 24 / 48.0, delta=1.0)


class AlignedGainTests(unittest.TestCase):
    def test_steady_input_gets_the_legacy_level(self):
        legacy = make_changer(RealtimeOptions(output_position="trim"))
        aligned = make_changer(RealtimeOptions(output_position="trim", gain_mode="aligned"))
        block = legacy.block_frame
        rng = np.random.default_rng(1)
        x = (rng.standard_normal(block * 4) * 0.05).astype(np.float32)
        for changer in (legacy, aligned):
            for i in range(0, block * 4, block):
                out = changer.on_request(x[i:i + block])[0]
            changer.last = out
        ratio = np.sqrt(np.mean(aligned.last ** 2) / np.mean(legacy.last ** 2))
        self.assertAlmostEqual(ratio, 1.0, delta=0.05)

    def test_gain_follows_the_rendered_moments(self):
        changer = make_changer(RealtimeOptions(output_position="trim", gain_mode="aligned"))
        runtime = changer.vc_model
        runtime.convert_buffer.zero_()
        runtime.convert_buffer[-16000:] = 0.1  # only the newest second is loud
        gain = runtime.aligned_gain(runtime.return_length * 480)
        self.assertEqual(gain.shape[0], runtime.return_length * 480)
        self.assertLess(gain[0].item(), gain[-1].item())


class StreamEvalTests(unittest.TestCase):
    def test_identity_stream_has_invisible_seams_and_a_broken_one_does_not(self):
        from rvc.realtime import stream_eval

        changer = make_changer(RealtimeOptions(output_position="trim"), chunk_ms=240, crossfade=0.05)
        rng = np.random.default_rng(2)
        t = np.arange(48000 * 6) / 48000
        # A voice-like signal: harmonics under a slow syllable envelope.
        audio = (
            (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
            * sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 8))
            * 0.05
            + 0.002 * rng.standard_normal(t.shape)
        ).astype(np.float32)
        output, timings = stream_eval.stream(changer, audio, synchronize=False)
        self.assertEqual(len(timings), -(-len(audio) // changer.block_frame))
        clean = stream_eval.seam_score(output, changer.block_frame, changer.crossfade_frame)
        self.assertLess(abs(clean - 1.0), 0.15)
        broken = output.copy()
        for start in range(0, len(broken), changer.block_frame):
            broken[start : start + 480] *= -1.0  # a phase flip at every join
        self.assertGreater(
            stream_eval.seam_score(broken, changer.block_frame, changer.crossfade_frame),
            clean + 0.3,
        )

    def test_alignment_finds_the_streaming_delay(self):
        from rvc.realtime import stream_eval

        changer = make_changer(RealtimeOptions(output_position="trim"))
        t = np.arange(48000 * 6) / 48000
        audio = ((np.sin(2 * np.pi * 0.7 * t) > 0) * np.sin(2 * np.pi * 200 * t) * 0.1).astype(np.float32)
        output, _ = stream_eval.stream(changer, audio, synchronize=False)
        expected = changer.vc_model.algorithmic_delay_ms * 48
        lag = stream_eval.align(audio, output, expected)
        self.assertLessEqual(abs(lag - expected), 480)
        # The legacy gain scales the identity output, so compare against a wrong lag.
        right = stream_eval.mel_distance(audio, output, lag)
        wrong = stream_eval.mel_distance(audio, output, lag + 4800)
        self.assertLess(right, 0.7 * wrong)


if __name__ == "__main__":
    unittest.main()
