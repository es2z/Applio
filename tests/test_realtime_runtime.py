import unittest
import threading
import time
from unittest.mock import patch

import numpy as np

from rvc.realtime.audio import Audio
from rvc.realtime.devices import AudioDeviceRef
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

    def test_drop_oldest_keeps_live_edge(self):
        ring = FloatRingBuffer(6)
        ring.write(np.arange(4, dtype=np.float32))
        self.assertEqual(
            ring.write(
                np.arange(4, 8, dtype=np.float32), overflow_policy="drop_oldest"
            ),
            4,
        )
        np.testing.assert_array_equal(ring.read(6), [2, 3, 4, 5, 6, 7])
        self.assertEqual(ring.overwritten_samples, 2)

    def test_trim_and_replace_never_leave_stale_prefix(self):
        ring = FloatRingBuffer(6)
        ring.write(np.arange(6, dtype=np.float32))
        self.assertEqual(ring.trim_to_latest(3), 3)
        np.testing.assert_array_equal(ring.read(3), [3, 4, 5])
        ring.write(np.arange(4, dtype=np.float32))
        ring.replace(np.arange(10, 13, dtype=np.float32))
        np.testing.assert_array_equal(ring.read(3), [10, 11, 12])


class AudioTransportTests(unittest.TestCase):
    @staticmethod
    def _device(index, direction, *, host_index=3):
        return AudioDeviceRef(
            index=index,
            name=f"Device {index}",
            host_api="Windows WASAPI",
            host_api_index=host_index,
            max_input_channels=2 if direction == "input" else 0,
            max_output_channels=2 if direction == "output" else 0,
            default_samplerate=48_000,
            direction=direction,
            duplicate_ordinal=0,
            fingerprint=f"device-{direction}-{index}",
        )

    def test_same_host_api_different_devices_use_separate_streams(self):
        input_device = self._device(28, "input")
        output_device = self._device(22, "output")
        self.assertFalse(Audio._can_use_native_duplex(input_device, output_device))

    def test_one_bidirectional_portaudio_device_can_use_duplex(self):
        input_device = self._device(7, "input")
        output_device = self._device(7, "output")
        input_device = AudioDeviceRef(
            **{**input_device.__dict__, "max_output_channels": 2}
        )
        output_device = AudioDeviceRef(
            **{**output_device.__dict__, "max_input_channels": 2}
        )
        self.assertTrue(Audio._can_use_native_duplex(input_device, output_device))

    def test_half_speed_device_clock_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "requested 48000.*observed"):
            Audio._validate_observed_rate("input", 37, 48_000, 24_060)

    def test_small_device_clock_measurement_error_is_accepted(self):
        Audio._validate_observed_rate("input", 28, 48_000, 47_672)

    def test_multi_second_reported_io_latency_is_rejected(self):
        audio = Audio(callbacks=object())
        audio.device_latency_ms = 4_740.7
        with self.assertRaisesRegex(RuntimeError, "abnormal I/O latency"):
            audio._validate_reported_io_latency()

    def test_stop_closes_stream_even_when_not_running(self):
        class FakeStream:
            def __init__(self):
                self.stopped = False
                self.closed = False

            def stop(self):
                self.stopped = True

            def close(self):
                self.closed = True

        audio = Audio(callbacks=object())
        stream = FakeStream()
        audio.input_stream = stream
        audio.running = False
        audio.stop()
        self.assertTrue(stream.stopped)
        self.assertTrue(stream.closed)
        self.assertIsNone(audio.input_stream)

    def test_buffer_capacities_follow_runtime_shape_not_eight_seconds(self):
        for requested_chunk_ms in (128, 512, 960, 1_200):
            with self.subTest(chunk=requested_chunk_ms):
                shape = RuntimeAudioShape.create(requested_chunk_ms)
                audio = Audio(callbacks=object(), runtime_shape=shape)
                audio._create_buffers(separate_stream_clocks=False)
                self.assertEqual(
                    audio.internal_input_ring.capacity,
                    shape.context_frames + 4_800 + 8_192,
                )
                self.assertEqual(
                    audio.input_ring.capacity,
                    max(shape.context_frames + 4_800 + 8_192, 16_384),
                )
                self.assertEqual(
                    audio.output_ring.capacity,
                    max(shape.hop_frames + 14_880 + 8_192, 16_384),
                )

    def test_worker_catches_up_to_latest_context_after_backlog(self):
        class PassThroughCallbacks:
            def change_voice(self, samples, *_args):
                return samples.copy(), 0.0, [0.0, 0.0, 0.0], None

        shape = RuntimeAudioShape.create(128, processing_mode="fixed_chunk")
        audio = Audio(callbacks=PassThroughCallbacks(), runtime_shape=shape)
        audio._create_buffers(separate_stream_clocks=False)
        backlog_frames = shape.context_frames + 4_800 + 1_000
        source = np.arange(backlog_frames, dtype=np.float32)
        audio.input_ring.write(source)
        audio.running = True
        worker = threading.Thread(target=audio._worker_loop, daemon=True)
        worker.start()
        audio._input_event.set()
        deadline = time.monotonic() + 1.0
        while audio.output_ring.available < shape.hop_frames:
            if time.monotonic() >= deadline:
                self.fail("worker did not produce catch-up output")
            time.sleep(0.001)
        audio.running = False
        audio._input_event.set()
        worker.join(1.0)

        output = audio.output_ring.read(shape.hop_frames)
        self.assertEqual(audio.backlog_recoveries, 1)
        self.assertEqual(len(output), shape.hop_frames)
        self.assertAlmostEqual(output[-1], source[-1])
        self.assertGreaterEqual(audio.stale_input_dropped_frames, 5_800)

    def test_normal_output_sawtooth_is_not_mistaken_for_stale_backlog(self):
        shape = RuntimeAudioShape.create(
            960,
            processing_mode="overlap",
            hop_mode="manual",
            manual_hop_ms=800,
        )
        audio = Audio(callbacks=object(), runtime_shape=shape)
        audio._create_buffers(separate_stream_clocks=False)
        residual = np.zeros(8_000, dtype=np.float32)
        audio.output_ring.write(residual)
        audio._write_processed(np.zeros(shape.hop_frames, dtype=np.float32))
        self.assertEqual(audio.stale_output_dropped_frames, 0)
        self.assertEqual(
            audio.output_ring.available, len(residual) + shape.hop_frames
        )


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

    def test_extra_buffer_growth_obeys_configured_cap(self):
        controller = ElasticBufferController(10, max_extra_buffer_ms=20)
        controller._last_adjust_at = -10.0
        with patch("rvc.realtime.runtime.time.monotonic", return_value=100.0):
            controller.record_underflow()
        controller._last_adjust_at = -10.0
        with patch("rvc.realtime.runtime.time.monotonic", return_value=101.0):
            controller.record_underflow()
        controller._last_adjust_at = -10.0
        with patch("rvc.realtime.runtime.time.monotonic", return_value=102.0):
            controller.record_underflow()
        self.assertEqual(controller.extra_buffer_ms, 20)

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
