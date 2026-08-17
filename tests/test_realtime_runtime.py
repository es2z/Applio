import unittest
from unittest.mock import patch

import numpy as np

from rvc.realtime.audio import Audio
from rvc.realtime.runtime import (
    ElasticBufferController,
    FloatRingBuffer,
    RuntimeAudioShape,
)


class RuntimeAudioShapeTests(unittest.TestCase):
    def test_chunk_is_runtime_variable(self):
        for requested in (128, 256, 512, 960, 1200):
            shape = RuntimeAudioShape.create(requested)
            self.assertAlmostEqual(shape.requested_chunk_ms, requested)
            self.assertEqual(shape.context_frames % 128, 0)
            self.assertEqual(shape.hop_frames, shape.context_frames)

    def test_overlap_keeps_context_and_fixes_shorter_hop(self):
        shape = RuntimeAudioShape.create(
            960,
            processing_mode="overlap",
            hop_mode="manual",
            manual_hop_ms=320,
        )
        self.assertEqual(shape.context_frames, 46_080)
        self.assertEqual(shape.hop_frames, 15_360)
        self.assertEqual(shape.effective_hop_ms, 320)

    def test_auto_hop_uses_measured_p95_and_ten_ms_grid(self):
        shape = RuntimeAudioShape.create(
            512,
            processing_mode="overlap",
            hop_mode="auto",
            measured_p95_ms=71,
        )
        self.assertEqual(shape.effective_hop_ms, 110)

    def test_manual_hop_cannot_exceed_chunk(self):
        with self.assertRaises(ValueError):
            RuntimeAudioShape.create(
                128,
                processing_mode="overlap",
                hop_mode="manual",
                manual_hop_ms=320,
            )


class RingBufferTests(unittest.TestCase):
    def test_wraparound_preserves_order(self):
        ring = FloatRingBuffer(6)
        self.assertEqual(ring.write(np.arange(4, dtype=np.float32)), 4)
        np.testing.assert_array_equal(ring.read(3), [0, 1, 2])
        self.assertEqual(ring.write(np.arange(4, 8, dtype=np.float32)), 4)
        np.testing.assert_array_equal(ring.read(5), [3, 4, 5, 6, 7])


class ElasticBufferTests(unittest.TestCase):
    @patch("rvc.realtime.runtime.time.monotonic")
    def test_growth_is_additive_ten_ms_not_exponential(self, monotonic):
        controller = ElasticBufferController(10)
        monotonic.side_effect = [100.0, 101.0, 104.0, 105.0]
        self.assertEqual(controller.record_inference(330, 320), 0)
        self.assertEqual(controller.record_inference(330, 320), 480)
        self.assertEqual(controller.extra_buffer_ms, 10)
        self.assertEqual(controller.record_inference(330, 320), 0)
        self.assertEqual(controller.record_inference(330, 320), 480)
        self.assertEqual(controller.extra_buffer_ms, 20)

    def test_sola_style_growth_adds_exact_step(self):
        audio = np.sin(np.linspace(0, 8 * np.pi, 4_800)).astype(np.float32)
        expanded = Audio._insert_sola_style(audio, 480)
        self.assertEqual(len(expanded), len(audio) + 480)
        self.assertTrue(np.all(np.isfinite(expanded)))

    @patch("rvc.realtime.runtime.time.monotonic")
    def test_shrink_is_ten_ms_and_only_after_stable_silence(self, monotonic):
        controller = ElasticBufferController(10)
        controller.extra_buffer_ms = 20
        controller._last_miss_at = 100.0
        monotonic.side_effect = [200.0, 201.1, 201.5, 203.2]
        self.assertEqual(controller.observe_silence(True), 0)
        self.assertEqual(controller.observe_silence(True), -480)
        self.assertEqual(controller.observe_silence(True), 0)
        self.assertEqual(controller.observe_silence(True), -480)
        self.assertEqual(controller.extra_buffer_ms, 0)

    @patch("rvc.realtime.runtime.time.monotonic")
    def test_sustained_slow_average_sets_overload(self, monotonic):
        controller = ElasticBufferController(10)
        monotonic.side_effect = [100.0 + second for second in range(31)]
        for _ in range(31):
            controller.record_inference(11.0, 10.0)
        self.assertTrue(controller.overloaded)
        self.assertGreater(controller.backlog_ms, 0)


if __name__ == "__main__":
    unittest.main()
