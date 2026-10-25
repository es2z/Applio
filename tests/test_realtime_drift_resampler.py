"""The ring transport's drift resampler (rvc/realtime/drift_resampler.py)."""

import unittest

import numpy as np

from rvc.realtime.drift_resampler import DriftResampler

RATE = 48000
BLOCK = 240 * 48


def sine(freq, start, count):
    return (0.5 * np.sin(2 * np.pi * freq * (np.arange(count) + start) / RATE)).astype(np.float32)


def run(freq, ppms, block=BLOCK):
    resampler, t, out = DriftResampler(), 0, []
    for ppm in ppms:
        out.append(resampler.process(sine(freq, t, block), 1 + ppm / 1e6))
        t += block
    return out


class DriftResamplerTests(unittest.TestCase):
    def test_the_output_count_follows_the_ratio_exactly(self):
        resampler = DriftResampler()
        counts = [len(resampler.process(np.zeros(BLOCK), 1.0)) for _ in range(5)]
        # The first call is short by the 32-sample delay, every later one exact.
        self.assertEqual(counts, [BLOCK - 32] + [BLOCK] * 4)
        resampler = DriftResampler()
        total = sum(len(resampler.process(np.zeros(BLOCK), 1.001)) for _ in range(100))
        self.assertAlmostEqual(total, 100 * BLOCK * 1.001, delta=40)

    def test_no_step_when_the_ratio_crosses_one_to_one(self):
        # What soxr's variable-rate mode cannot do: 0 -> -1 ppm corrupted samples.
        for ppms in ([0, 0, -1, -1], [5, 5, -5, -5], [-210, 300, 950, -1000, 0]):
            played = np.concatenate(run(440.0, ppms))[RATE // 10 :]
            clean = 0.5 * (2 * np.pi * 440.0 / RATE) ** 2
            self.assertLess(float(np.abs(np.diff(played, 2)).max()), clean * 1.05, ppms)

    def test_flat_to_twenty_kilohertz(self):
        for freq in (100.0, 1000.0, 10000.0, 18000.0, 20000.0):
            played = np.concatenate(run(freq, [0, 300, -300, 0]))[RATE // 10 :]
            gain = np.sqrt(2 * np.mean(played.astype(np.float64) ** 2)) / 0.5
            self.assertAlmostEqual(20 * np.log10(gain), 0.0, delta=0.02, msg=freq)

    def test_at_one_to_one_it_lines_up_with_its_input(self):
        x = np.random.default_rng(0).standard_normal(BLOCK * 3).astype(np.float32) * 0.1
        resampler = DriftResampler()
        y = np.concatenate([resampler.process(x[i : i + BLOCK], 1.0) for i in range(0, len(x), BLOCK)])
        # Band-limited at 0.94 of Nyquist, so white noise comes back nearly, not exactly.
        error = y[1000 : len(x) - 1000] - x[1000 : len(x) - 1000]
        self.assertLess(np.sqrt(np.mean(error ** 2)) / 0.1, 0.25)
        lag = int(np.argmax(np.correlate(y[:4000], x[:4000], mode="full"))) - 3999
        self.assertEqual(lag, 0)


if __name__ == "__main__":
    unittest.main()
