"""Stage 4 opt-in options: input high-pass (IO1), clip stage (IO4), gate fade (IO5),
latent noise scale (P1), and the level readout."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy import signal

from rvc.realtime import core
from rvc.realtime.io_processing import InputHighpass, LevelMeter, dbfs, gate_fade_out
from rvc.realtime.options import DEFAULT_LATENT_NOISE_SCALE, RealtimeOptions
from rvc.realtime.pipeline import RealtimeVoiceConverter
from rvc.realtime.synthesis import SynthesisPlan, realtime_infer
from tests.test_apex_gan import build_synthesizer
from tests.test_realtime_options import IdentityPipeline


class NoisyPipeline(IdentityPipeline):
    """The identity plus the noise floor a real generator emits even for silence."""

    def voice_conversion(self, *args):
        out = super().voice_conversion(*args)
        generator = torch.Generator().manual_seed(int(out.shape[0]))
        return out + 0.01 * torch.randn(out.shape, generator=generator)


def changer(options, silent_threshold=0, chunk_ms=240, crossfade=0.05, extra=0.5, noisy=False):
    pipeline = NoisyPipeline if noisy else IdentityPipeline

    def fake_create_pipeline(**kwargs):
        return pipeline(kwargs.get("options") or RealtimeOptions())

    with patch.object(core, "create_pipeline", side_effect=fake_create_pipeline):
        return core.VoiceChanger(
            int(chunk_ms * 48000 / 1000 / 128), crossfade, extra,
            silent_threshold=silent_threshold, options=options,
        )


class OptionTests(unittest.TestCase):
    def test_defaults_change_nothing(self):
        options = RealtimeOptions()
        self.assertEqual(options.changed(), [])
        self.assertFalse(options.needs_realtime_synthesis)
        vc = changer(options)
        self.assertIsNone(vc.highpass)
        self.assertEqual(vc.gate_fade_frames, 0)

    def test_a_slider_at_067_is_the_default_latent_scale(self):
        self.assertEqual(
            RealtimeOptions.from_dict({"latent_noise_scale": 0.67}).latent_noise_scale,
            DEFAULT_LATENT_NOISE_SCALE,
        )
        lowered = RealtimeOptions.from_dict({"latent_noise_scale": 0.5})
        self.assertEqual(lowered.latent_noise_scale, 0.5)
        self.assertTrue(lowered.needs_realtime_synthesis)
        self.assertEqual(RealtimeOptions.from_dict(lowered.to_dict()), lowered)


class HighpassTests(unittest.TestCase):
    def test_streaming_equals_the_training_filter_on_the_whole_signal(self):
        # rvc/train/preprocess/preprocess.py: butter(N=5, Wn=48, "high", fs) + lfilter.
        b, a = signal.butter(N=5, Wn=48, btype="high", fs=48000)
        rng = np.random.default_rng(0)
        x = (0.1 * rng.standard_normal(48000 * 2)).astype(np.float32)
        whole = signal.lfilter(b, a, x.astype(np.float64))
        hp = InputHighpass(48)
        streamed = np.concatenate([hp.process(x[i:i + 11520]) for i in range(0, len(x), 11520)])
        # The same filter; what is left is the (b, a) form's rounding at 48 Hz / 48 kHz.
        self.assertLess(np.abs(streamed - whole).max(), 1e-5)

    def test_it_removes_rumble_and_keeps_the_voice_band(self):
        t = np.arange(48000) / 48000
        hp = InputHighpass(48)
        rumble = hp.process((0.1 * np.sin(2 * np.pi * 15 * t)).astype(np.float32))
        hp.reset()
        voice = hp.process((0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32))
        self.assertLess(np.abs(rumble[24000:]).max(), 0.002)
        self.assertGreater(np.abs(voice[24000:]).max(), 0.099)


class ClipStageTests(unittest.TestCase):
    def converter(self, clip):
        vc = RealtimeVoiceConverter.__new__(RealtimeVoiceConverter)
        vc.synthesis = None
        vc.clip_output = clip
        return vc

    def test_generator_stage_clips_and_output_stage_does_not(self):
        loud = lambda *args: torch.tensor([0.5, 2.0, -3.0])
        clipped = self.converter(True).inference(None, None, None, None, None, infer_audio=loud)
        kept = self.converter(False).inference(None, None, None, None, None, infer_audio=loud)
        self.assertEqual(clipped.tolist(), [0.5, 1.0, -1.0])
        self.assertEqual(kept.tolist(), [0.5, 2.0, -3.0])

    def test_the_peak_is_measured_before_clipping(self):
        vc = self.converter(True)
        vc.inference(None, None, None, None, None, infer_audio=lambda *a: torch.tensor([0.5, -3.0]))
        self.assertEqual(float(vc.last_peak), 3.0)


class GateFadeTests(unittest.TestCase):
    def run_gate(self, fade_ms):
        vc = changer(RealtimeOptions(gate_fade_ms=fade_ms), silent_threshold=-40, noisy=True)
        block = vc.block_frame
        rng = np.random.default_rng(3)
        loud = (0.2 * rng.standard_normal(block * 6)).astype(np.float32)
        room = (0.001 * rng.standard_normal(block * 8)).astype(np.float32)  # -60 dBFS
        out, gated = [], None
        for i, chunk in enumerate(np.split(np.concatenate((loud, room)), 14)):
            was_silent = vc.output_was_silent
            out.append(vc.on_request(chunk)[0])
            if gated is None and i > 2 and vc.output_was_silent and not was_silent:
                gated = i
                self.tail_after_gate = float(vc.sola_buffer.abs().max())
        self.assertIsNotNone(gated)
        return vc, out, gated

    def test_without_fade_the_output_is_cut(self):
        vc, out, i = self.run_gate(0)
        self.assertEqual(np.abs(out[i]).max(), 0.0)
        self.assertGreater(abs(out[i - 1][-1]), 0.0)

    def test_with_fade_the_tail_dies_away(self):
        vc, out, i = self.run_gate(30)
        fade = int(0.030 * 48000)
        head = out[i][:fade]
        # It starts where the previous block's audio would have gone on...
        self.assertGreater(np.abs(head[:48]).max(), 0.0)
        self.assertEqual(np.abs(out[i][fade:]).max(), 0.0)
        # ...and dies away (cos^2 from 1 to 0).
        self.assertLess(np.abs(head[-200:]).max(), 0.2 * np.abs(head[:200]).max())
        # The tail is used up, so a restart fades in from silence.
        self.assertEqual(self.tail_after_gate, 0.0)

    def test_flushes_reopen_the_gate_unless_held(self):
        def blocks_out(hold):
            vc = changer(RealtimeOptions(gate_hold=hold), silent_threshold=-40, noisy=True)
            block = vc.block_frame
            rng = np.random.default_rng(3)
            loud = (0.2 * rng.standard_normal(block * 6)).astype(np.float32)
            room = (0.001 * rng.standard_normal(block * 30)).astype(np.float32)
            chunks = np.split(np.concatenate((loud, room)), 36)
            outs = [np.abs(vc.on_request(c)[0]).max() for c in chunks]
            return sum(o > 0 for o in outs[16:])  # well after the first gate

        self.assertGreater(blocks_out(False), 0)  # room tone keeps leaking out
        self.assertEqual(blocks_out(True), 0)

    def test_fade_shape(self):
        tail = torch.ones(1000)
        faded = gate_fade_out(tail, 480)
        self.assertEqual(faded.shape[0], 480)
        self.assertAlmostEqual(float(faded[0]), 1.0, places=6)
        self.assertAlmostEqual(float(faded[-1]), 0.0, places=6)


class LatentScaleTests(unittest.TestCase):
    @torch.no_grad()
    def test_scale_zero_removes_the_latent_noise(self):
        torch.manual_seed(0)
        net_g = build_synthesizer("APEX-GAN").eval()
        frames = 30
        phone = torch.randn(1, frames, 768)
        lengths = torch.tensor([frames])
        pitchf = torch.full((1, frames), 180.0)
        pitch = torch.full((1, frames), 60, dtype=torch.long)
        sid = torch.tensor([0])
        dec_noise = torch.randn(1, frames * 480, 1)
        plan = SynthesisPlan(latent_scale=0.0)
        a = realtime_infer(net_g, phone, lengths, pitch, pitchf, sid, plan,
                           z_noise=torch.randn(1, 192, frames), dec_noise=dec_noise)
        b = realtime_infer(net_g, phone, lengths, pitch, pitchf, sid, plan,
                           z_noise=torch.randn(1, 192, frames), dec_noise=dec_noise)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(SynthesisPlan().latent_scale, DEFAULT_LATENT_NOISE_SCALE)


class LevelMeterTests(unittest.TestCase):
    def test_reports_generator_and_output_peaks(self):
        meter = LevelMeter(240)
        self.assertEqual(meter.status(), "")
        meter.record(torch.tensor(0.5), np.full(10, 0.1, dtype=np.float32))
        meter.record(torch.tensor(1.5), np.full(10, 0.2, dtype=np.float32))
        snap = meter.snapshot()
        self.assertAlmostEqual(snap["generator_max"], 1.5)
        self.assertEqual(snap["generator_over_blocks"], 1)
        self.assertAlmostEqual(snap["output_max"], 0.2, places=6)
        self.assertIn("1 blocks over 0 dBFS", meter.status())
        self.assertAlmostEqual(dbfs(0.5), -6.02, places=2)

    def test_changer_records_every_block(self):
        vc = changer(RealtimeOptions())
        vc.on_request(np.zeros(vc.block_frame, dtype=np.float32))
        self.assertIn("output peak", vc.level_status())


if __name__ == "__main__":
    unittest.main()
