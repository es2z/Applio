"""The realtime generator call behind the opt-in options (rvc/realtime/synthesis.py)."""

import unittest
import torch

from rvc.lib.algorithm.generators.apex_gan import ExcitationSynthesizer
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


class EnvelopeReferenceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
