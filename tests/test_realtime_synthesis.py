"""The realtime generator call behind the opt-in options (rvc/realtime/synthesis.py) and
the time-locked state the pipeline keeps for it."""

import unittest
from types import SimpleNamespace

import torch

from rvc.lib.algorithm.generators.apex_gan import ExcitationSynthesizer
from rvc.realtime.pipeline import Realtime_Pipeline
from rvc.realtime.synthesis import SynthesisPlan, realtime_infer
from tests.test_apex_gan import build_decoder, build_synthesizer


def inputs(frames, dim=768):
    torch.manual_seed(3)
    phone = torch.randn(1, frames, dim)
    lengths = torch.tensor([frames])
    pitchf = 180.0 + 40.0 * torch.sin(torch.linspace(0, 6.0, frames)).unsqueeze(0)
    pitch = torch.full((1, frames), 60, dtype=torch.long)
    return phone, lengths, pitch, pitchf, torch.tensor([0])


class RealtimeInferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.net_g = build_synthesizer("APEX-GAN").eval()

    @torch.no_grad()
    def test_without_options_it_is_synthesizer_infer(self):
        args = inputs(40)
        torch.manual_seed(7)
        expected = self.net_g.infer(*args)[0]
        torch.manual_seed(7)
        actual = realtime_infer(self.net_g, *args, SynthesisPlan())
        self.assertTrue(torch.equal(expected, actual))

    @torch.no_grad()
    def test_trim_renders_return_length_frames(self):
        args = inputs(60)
        plan = SynthesisPlan(skip_head=40, return_length=20)
        out = realtime_infer(self.net_g, *args, plan)
        self.assertEqual(out.shape[-1], 20 * 480)
        self.assertEqual(plan.flow_head, 16)

    @torch.no_grad()
    def test_given_noise_replaces_the_draws(self):
        args = inputs(40)
        plan = SynthesisPlan(skip_head=10, return_length=30)
        z_noise = torch.randn(1, 192, 40 - plan.flow_head)
        dec_noise = torch.randn(1, 30 * 480, 1)
        torch.manual_seed(1)
        first = realtime_infer(self.net_g, *args, plan, z_noise, dec_noise)
        torch.manual_seed(2)
        second = realtime_infer(self.net_g, *args, plan, z_noise, dec_noise)
        self.assertTrue(torch.equal(first, second))


class TimeLockedExcitationTests(unittest.TestCase):
    """Two overlapping windows render the same instant the same way only when noise
    and phase are tied to the instant rather than to the window."""

    @torch.no_grad()
    def _overlap_error(self, locked, fixed=False):
        torch.manual_seed(0)
        decoder = build_decoder().eval()
        reference = decoder.noise_envelope_reference(1.0) if fixed else None
        frames, shift, margin = 40, 16, 6
        total = frames + shift
        x = torch.randn(1, 192, total)
        f0 = 170.0 + 50.0 * torch.sin(torch.linspace(0, 5.0, total)).unsqueeze(0)
        g = torch.randn(1, 256, 1)
        noise = torch.randn(1, total * 480, 1)

        def window(start, phase0):
            kwargs = {}
            if locked:
                kwargs = dict(
                    noise=noise[:, start * 480 : (start + frames) * 480],
                    phase0=phase0,
                )
            return decoder(
                x[:, :, start : start + frames],
                f0[:, start : start + frames],
                g,
                envelope_reference=reference,
                **kwargs,
            )[0, 0]

        phase0 = (
            torch.remainder((f0[0, :shift].double() / 48000) * 480, 1.0)
            .sum()
            .remainder(1.0)
            .view(1)
        )
        first = window(0, torch.zeros(1, dtype=torch.float64))
        second = window(shift, phase0)
        overlap_first = first[(shift + margin) * 480 : (frames - margin) * 480]
        overlap_second = second[margin * 480 : (frames - shift - margin) * 480]
        return ((overlap_first - overlap_second).norm() / overlap_first.norm()).item()

    def test_locked_windows_agree_where_free_ones_do_not(self):
        locked = self._overlap_error(locked=True)
        free = self._overlap_error(locked=False)
        # Measured: 0.0019 locked, 1.28 free.
        self.assertLess(locked, 0.005)
        self.assertGreater(free, 100 * locked)

    def test_the_fixed_envelope_takes_most_of_the_rest(self):
        # Part of what is left is the window-mean noise envelope, which differs
        # between the two windows; the fixed reference is the same for both.
        # Measured: 0.0019 -> 0.00037.
        locked = self._overlap_error(locked=True)
        fixed = self._overlap_error(locked=True, fixed=True)
        self.assertLess(fixed, 0.5 * locked)

    def test_envelope_reference_matches_a_window_with_that_voicing(self):
        decoder = build_decoder().eval()
        synth = ExcitationSynthesizer(48000, hop_length=480)
        torch.manual_seed(4)
        for ratio in (0.5, 0.65, 0.8):
            f0 = torch.full((1, 400), 200.0)
            f0[:, int(400 * ratio):] = 0.0
            _, noise = synth(f0)
            magnitude, _ = decoder.stft.transform(noise.squeeze(1))
            window_mean = magnitude.mean(dim=1).mean().item()
            reference = decoder.noise_envelope_reference(ratio)
            self.assertAlmostEqual(reference / window_mean, 1.0, delta=0.05)


class StreamStateTests(unittest.TestCase):
    def test_f0_cache_rewrites_only_the_newest_frames(self):
        state = SimpleNamespace(f0_cache_frames=(3, 5))
        buffer = torch.arange(10, dtype=torch.float32)
        new = torch.full((10,), -1.0)
        Realtime_Pipeline._write_pitch(state, new, buffer)
        # Shifted by 3, and only the newest 5 replaced.
        self.assertEqual(buffer.tolist(), [3, 4, 5, 6, 7, -1, -1, -1, -1, -1])

    def test_without_the_cache_the_window_is_replaced(self):
        state = SimpleNamespace(f0_cache_frames=None)
        buffer = torch.arange(10, dtype=torch.float32)
        Realtime_Pipeline._write_pitch(state, torch.full((10,), -1.0), buffer)
        self.assertEqual(buffer.tolist(), [-1.0] * 10)

    def test_noise_moves_with_the_window(self):
        z = torch.randn(1, 4, 10)
        d = torch.randn(1, 10 * 480, 1)
        state = SimpleNamespace(z_noise=z.clone(), dec_noise=d.clone(), stream={"shift": 3}, model_window=480)
        Realtime_Pipeline._advance_noise(state)
        self.assertTrue(torch.equal(state.z_noise[:, :, :7], z[:, :, 3:]))
        self.assertTrue(torch.equal(state.dec_noise[:, : 7 * 480], d[:, 3 * 480 :]))
        self.assertFalse(torch.equal(state.z_noise[:, :, 7:], z[:, :, :3]))


if __name__ == "__main__":
    unittest.main()
