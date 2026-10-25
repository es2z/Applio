"""The ring transport (rvc/realtime/ring_transport.py) driven by simulated devices.

No audio device is opened. A simulation calls the PortAudio callbacks in host-period
steps and runs the worker's pump() when a block's conversion would finish, with a fake
clock that says how long the conversion took. That makes the timing deterministic, so
the tests can check the delay the transport adds and how its output reserve behaves.
"""

import math
import sys
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from rvc.realtime.ring_buffer import FloatRingBuffer, ReserveController
from rvc.realtime.ring_transport import ConversionThread, RingAudio, pad_quiet, trim_quiet

RATE = 48000
PERIOD = 480  # 10 ms host buffers


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class IdentityCallbacks:
    """change_voice returns its input and advances the clock by the processing time."""

    def __init__(self, clock, processing_ms):
        self.clock = clock
        self.processing_ms = processing_ms  # a number or a callable(block index)
        self.blocks = 0
        self.resets = 0

    def change_voice(self, data, *args):
        p = self.processing_ms(self.blocks) if callable(self.processing_ms) else self.processing_ms
        self.blocks += 1
        self.clock.now += p / 1000.0
        return np.asarray(data, dtype=np.float32).copy(), 0.0, [0, p, 0], None

    def reset_stream(self):
        self.resets += 1


def make_audio(processing_ms, block_ms=960, warmup=(), separate=False, input_rate=RATE):
    clock = FakeClock()
    callbacks = IdentityCallbacks(clock, processing_ms)
    audio = RingAudio(
        callbacks, block_frame=int(block_ms * 48), warmup_timings_ms=warmup
    )
    audio.clock = clock
    audio.controller = ReserveController(warmup, clock=clock)
    audio.input_rate, audio.output_rate = input_rate, RATE
    audio._allocate(separate_clocks=separate)
    audio.running = True
    return audio, clock, callbacks


def simulate(audio, clock, callbacks, signal, input_ppm=0.0, trace=None, phase_ms=3.3):
    """Run `signal` through the transport; returns what the output device played.

    Event driven: the output device asks for PERIOD samples every 10 ms, the input
    device delivers PERIOD samples every 10 ms of its own clock (`input_ppm` faster),
    `phase_ms` apart. A block's pump() is timed so that it finishes its fake
    processing exactly when the conversion would have.
    """
    in_period = PERIOD / RATE / (1 + input_ppm / 1e6)
    out_period = PERIOD / RATE
    next_in, next_out = phase_ms / 1000.0, 0.0
    pumps = []
    out, total_in, captured, blocks = [], 0, 0, 0
    block = audio.block_frame

    def processing(index):
        p = callbacks.processing_ms
        return p(index) if callable(p) else p

    while total_in < len(signal):
        t = min(next_in, next_out, pumps[0][0] if pumps else math.inf)
        if pumps and pumps[0][0] == t:
            _, p = pumps.pop(0)
            clock.now = t - p / 1000.0  # change_voice moves it on by p
            audio.pump()
            continue
        clock.now = t
        if t == next_in:
            chunk = signal[total_in : total_in + PERIOD]
            total_in += len(chunk)
            audio.input_callback(chunk.reshape(-1, 1), len(chunk), None, None)
            captured += len(chunk)
            while captured >= block:
                captured -= block
                p = processing(blocks)
                blocks += 1
                pumps.append((t + p / 1000.0, p))
            next_in += in_period
        else:
            outdata = np.zeros((PERIOD, 1), dtype=np.float32)
            audio.output_callback(outdata, PERIOD, None, None)
            out.append(outdata[:, 0].copy())
            if trace is not None:
                trace.append(audio.output_underflows)
            next_out += out_period
    return np.concatenate(out)


def delay_ms(signal, played):
    from scipy.signal import correlate

    start = len(signal) // 2
    seg = played[start : start + RATE * 2]
    ref = signal[start - RATE * 3 : start + RATE * 2]
    scores = correlate(ref, seg, mode="valid", method="fft")[::-1]
    lag = int(np.argmax(scores))
    return lag / 48.0


def voice_like(seconds, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.1 * rng.standard_normal(t.shape) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t))).astype(np.float32)


class RingBufferTests(unittest.TestCase):
    def test_wraps_and_reports_short_reads(self):
        ring = FloatRingBuffer(8)
        self.assertEqual(ring.write(np.arange(6)), 6)
        self.assertEqual(ring.read(4).tolist(), [0, 1, 2, 3])
        self.assertEqual(ring.write(np.arange(6, 12)), 6)
        self.assertEqual(ring.write(np.arange(3)), 0)  # full
        self.assertEqual(ring.read(10).tolist(), [4, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual(ring.underrun_reads, 2)
        ring.write(np.arange(5))
        self.assertEqual(ring.discard(3), 3)
        self.assertEqual(ring.read(5).tolist(), [3, 4])

    def test_a_reader_and_a_writer_on_two_threads_never_wait_or_lose_samples(self):
        # Lock free: neither side can find the other "inside" and give up, which is
        # what used to turn a contended callback into a period of silence.
        ring = FloatRingBuffer(1000)
        total = 200_000
        received = []

        def produce():
            sent = 0
            while sent < total:
                chunk = np.arange(sent, min(sent + 137, total), dtype=np.float32)
                sent += ring.write(chunk)

        producer = threading.Thread(target=produce)
        producer.start()
        got = 0
        while got < total:
            chunk = ring.read(97)
            received.append(chunk)
            got += len(chunk)
        producer.join()
        joined = np.concatenate(received)
        self.assertTrue(np.array_equal(joined, np.arange(total, dtype=np.float32)))


class QuietCorrectionTests(unittest.TestCase):
    def test_pad_and_trim_change_the_length_exactly_and_stay_quiet(self):
        rng = np.random.default_rng(0)
        audio = (1e-4 * rng.standard_normal(4800)).astype(np.float32)
        self.assertEqual(len(pad_quiet(audio, 480)), 5280)
        self.assertEqual(len(trim_quiet(audio, 480)), 4320)
        self.assertLess(np.abs(pad_quiet(audio, 480)).max(), 1e-3)

    def test_a_loud_block_is_never_corrected(self):
        clock = FakeClock()
        controller = ReserveController([40.0], clock=clock)
        for _ in range(9):
            clock.now += 1.0
            self.assertEqual(controller.update(40.0, 200.0, quiet=False, drift=False), 0.0)
        clock.now += 1.0
        self.assertLess(controller.update(40.0, 200.0, quiet=True, drift=False), -100.0)


class DuplexTests(unittest.TestCase):
    def test_delay_is_one_chunk_plus_processing_plus_reserve(self):
        audio, clock, callbacks = make_audio(50.0, warmup=[50.0] * 5)
        signal = voice_like(20)
        played = simulate(audio, clock, callbacks, signal)
        measured = delay_ms(signal, played)
        # 960 ms chunk + 50 ms processing + 30 ms reserve. The legacy transport
        # adds a whole chunk (960 ms) here instead of the 80 ms.
        self.assertAlmostEqual(measured, 960 + 50 + 30, delta=5)
        self.assertAlmostEqual(audio.estimated_latency_ms(), measured, delta=15)
        self.assertEqual(audio.output_underflows, 0)
        # Identity in, identity out: nothing was trimmed or padded in loud audio.
        lag = int(round(measured * 48))
        error = played[lag + RATE * 2 : lag + RATE * 10] - signal[RATE * 2 : RATE * 10]
        self.assertLess(np.abs(error).max(), 1e-6)

    def test_jitter_within_the_warmup_p95_never_underflows(self):
        rng = np.random.default_rng(1)
        times = rng.uniform(30, 70, 200)
        audio, clock, callbacks = make_audio(
            lambda i: float(times[i % 200]), warmup=list(times[:5]) + [70.0]
        )
        simulate(audio, clock, callbacks, voice_like(30))
        self.assertEqual(audio.output_underflows, 0)

    def test_slower_processing_underflows_once_and_then_settles(self):
        signal = voice_like(40)
        signal[RATE * 15 :] *= 1e-4  # quiet from 15 s on
        audio, clock, callbacks = make_audio(
            lambda i: 40.0 if i < 8 else 90.0, warmup=[40.0] * 5
        )
        trace = []
        simulate(audio, clock, callbacks, signal, trace=trace)
        # Block 8 arrives 50 ms later than planned, 20 ms past the reserve: one
        # gap, which itself moves the playback later, and nothing after it.
        self.assertGreater(audio.output_underflows, 0)
        self.assertEqual(trace[-1], trace[len(trace) // 3])
        # The underflow added a 10 ms step, which 30 s without another one and 1 s
        # of quiet took back again; the 90 ms blocks are planned for by the target.
        self.assertEqual(audio.controller.underflows, 1)
        self.assertEqual(audio.controller.reserve_ms, 30)
        self.assertGreaterEqual(audio.controller.need_ms, 90.0)
        errors = list(audio.controller.errors)
        self.assertLessEqual(abs(float(np.median(errors))), 10.0)


    def test_an_occasional_slow_block_clicks_once_not_every_time(self):
        # One block in 25 (~24 s) takes 70 ms longer, in speech with pauses. Aiming
        # at the p95 forgot it between occurrences: 6 underflows in 240 s, each a
        # click. The slowest block is now remembered for two minutes.
        signal = voice_like(150)
        t = np.arange(len(signal)) / RATE
        signal[(t % 4.5) > 3.0] *= 1e-4
        rng = np.random.default_rng(3)
        times = [45 + rng.uniform(-5, 5) + (70 if i % 25 == 24 else 0) for i in range(400)]
        audio, clock, callbacks = make_audio(lambda i: float(times[i]), warmup=[45.0] * 5)
        simulate(audio, clock, callbacks, signal)
        self.assertEqual(audio.controller.underflows, 1)

    def test_a_larger_margin_absorbs_what_the_default_does_not(self):
        audio, clock, callbacks = make_audio(
            lambda i: 40.0 if i < 8 else 90.0, warmup=[40.0] * 5
        )
        audio.controller = ReserveController([40.0] * 5, clock=clock, margin_ms=60.0)
        simulate(audio, clock, callbacks, voice_like(20))
        self.assertEqual(audio.output_underflows, 0)


class UnderflowConcealmentTests(unittest.TestCase):
    def _audio(self):
        audio, _, _ = make_audio(40.0)
        audio._playback_started = True
        return audio

    def _render(self, audio, frames=PERIOD, status=None):
        outdata = np.zeros((frames, 1), dtype=np.float32)
        audio.output_callback(outdata, frames, None, status)
        return outdata[:, 0].copy()

    def test_running_dry_mid_period_fades_out_and_back_in(self):
        audio = self._audio()
        audio.output_ring.write(np.full(PERIOD + 200, 0.5, dtype=np.float32))
        first = self._render(audio)
        second = self._render(audio)  # 200 samples, then nothing
        audio.output_ring.write(np.full(PERIOD, 0.5, dtype=np.float32))
        third = self._render(audio)
        played = np.concatenate((first, second, third))
        # Without the fades the steps are 0.5 at both ends of the gap.
        self.assertLess(np.abs(np.diff(played)).max(), 0.5 / 90)
        self.assertEqual(audio.output_underflows, 1)
        self.assertAlmostEqual(float(third[-1]), 0.5)

    def test_running_dry_at_a_period_boundary_ramps_from_the_last_sample(self):
        audio = self._audio()
        audio.output_ring.write(np.full(PERIOD, -0.4, dtype=np.float32))
        first = self._render(audio)
        second = self._render(audio)  # nothing at all
        played = np.concatenate((first, second))
        self.assertLess(np.abs(np.diff(played)).max(), 0.4 / 90)
        self.assertEqual(float(second[-1]), 0.0)

    def test_full_periods_are_untouched(self):
        audio = self._audio()
        signal = voice_like(0.1)
        audio.output_ring.write(signal)
        played = np.concatenate([self._render(audio) for _ in range(len(signal) // PERIOD)])
        self.assertTrue(np.array_equal(played, signal[: len(played)]))

    def test_late_device_callbacks_are_counted_apart_from_ring_underflows(self):
        audio = self._audio()
        audio.output_ring.write(np.zeros(PERIOD, dtype=np.float32))

        class Flags:
            output_underflow = True

            def __bool__(self):
                return True

        self._render(audio, status=Flags())
        self.assertEqual(audio.device_output_underflows, 1)
        self.assertEqual(audio.output_underflows, 0)
        self.assertIn("device 0/1", audio.status_text())


class SeparateClockTests(unittest.TestCase):
    def test_drift_between_two_clocks_is_taken_up_by_the_resampler(self):
        audio, clock, callbacks = make_audio(40.0, warmup=[40.0] * 5, separate=True)
        self.assertTrue(audio.drift_enabled)
        signal = voice_like(150)
        simulate(audio, clock, callbacks, signal, input_ppm=200.0)
        # The input runs 200 ppm fast; without correction the ring would have grown
        # by 30 ms over the run. The resampler settles near -200 ppm instead.
        self.assertAlmostEqual(audio.controller.drift_ppm, -200.0, delta=30.0)
        self.assertEqual(audio.output_underflows, 0)
        self.assertEqual(audio.controller.adjustments, 0)  # all loud: drift alone did it
        errors = list(audio.controller.errors)
        self.assertLess(abs(float(np.median(errors))), 10.0)


class FakeStream:
    """A sounddevice stream that calls its callback from a thread every 10 ms."""

    instances = []

    def __init__(self, callback, device, channels, samplerate, kind, **kwargs):
        self.callback, self.device, self.kind = callback, device, kind
        self.channels, self.samplerate = channels, samplerate
        self.kwargs = kwargs
        self.latency = (0.005, 0.01) if kind == "duplex" else 0.008
        self._stop = threading.Event()
        self._thread = None
        FakeStream.instances.append(self)

    def start(self):
        def run():
            t = np.arange(PERIOD)
            n = 0
            while not self._stop.is_set():
                tone = (0.1 * np.sin(2 * np.pi * 220 * (t + n) / RATE)).astype(np.float32)
                n += PERIOD
                if self.kind == "duplex":
                    cin, cout = self.channels
                    out = np.zeros((PERIOD, cout), dtype=np.float32)
                    self.callback(np.repeat(tone[:, None], cin, axis=1), out, PERIOD, None, None)
                elif self.kind == "input":
                    self.callback(np.repeat(tone[:, None], self.channels, axis=1), PERIOD, None, None)
                else:
                    self.callback(np.zeros((PERIOD, self.channels), dtype=np.float32), PERIOD, None, None)
                time.sleep(PERIOD / RATE)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def close(self):
        self.stop()


class StartStopTests(unittest.TestCase):
    """start() and stop() against a fake sounddevice: stream choice, the worker
    thread, and that converted audio reaches the output."""

    def _run(self, hosts, mode, conversion_thread=None, on_convert=None):
        import sounddevice as sd

        from rvc.realtime import ring_transport

        devices = {
            0: {"name": "mic", "hostapi": hosts[0], "max_input_channels": 2,
                "max_output_channels": 0, "default_samplerate": 48000.0},
            1: {"name": "out", "hostapi": hosts[1], "max_input_channels": 0,
                "max_output_channels": 2, "default_samplerate": 48000.0},
        }
        apis = ["Windows WASAPI", "Windows WDM-KS"]
        FakeStream.instances = []
        clock = FakeClock()
        callbacks = IdentityCallbacks(clock, 5.0)

        def convert(data, *a):
            if on_convert is not None:
                on_convert()
            return np.asarray(data) * 0.5, 0.0, [0, 0, 0], None

        callbacks.change_voice = convert
        audio = RingAudio(callbacks, block_frame=240 * 48, stream_mode=mode,
                          warmup_timings_ms=[5.0] * 5, algorithmic_delay_ms=150.0,
                          conversion_thread=conversion_thread)
        patches = [
            patch.object(sd, "query_devices", side_effect=lambda i: devices[i]),
            patch.object(sd, "query_hostapis", side_effect=lambda i: {"name": apis[i]}),
            patch.object(sd, "check_input_settings", return_value=None),
            patch.object(sd, "check_output_settings", return_value=None),
            patch.object(sd, "WasapiSettings", side_effect=lambda **kw: ("wasapi", kw)),
            patch.object(sd, "Stream", side_effect=lambda callback, device, channels, samplerate, **kw:
                         FakeStream(callback, device, channels, samplerate, "duplex", **kw)),
            patch.object(sd, "InputStream", side_effect=lambda callback, device, channels, samplerate, **kw:
                         FakeStream(callback, device, channels, samplerate, "input", **kw)),
            patch.object(sd, "OutputStream", side_effect=lambda callback, device, channels, samplerate, **kw:
                         FakeStream(callback, device, channels, samplerate, "output", **kw)),
        ]
        for p in patches:
            p.start()
        try:
            switch = sys.getswitchinterval()
            audio.start(0, 1, exclusive_mode=False)
            time.sleep(1.2)
            status = audio.status_text()
            audio.stop()
            self.assertEqual(sys.getswitchinterval(), switch)
        finally:
            for p in patches:
                p.stop()
        return audio, status

    def test_same_host_auto_opens_one_duplex_stream(self):
        audio, status = self._run((0, 0), "auto")
        kinds = [s.kind for s in FakeStream.instances]
        self.assertEqual(kinds, ["duplex"])
        stream = FakeStream.instances[0]
        self.assertEqual(stream.kwargs["blocksize"], 0)
        self.assertEqual(stream.kwargs["latency"], "low")
        self.assertEqual(stream.channels, (2, 2))  # the legacy transport's channel counts
        self.assertFalse(audio.drift_enabled)
        self.assertGreaterEqual(callbacks_blocks(audio), 3)
        self.assertIn("ring duplex/Windows WASAPI", status)

    def test_separate_mode_opens_two_streams_with_drift_correction(self):
        audio, status = self._run((0, 0), "separate")
        self.assertEqual(sorted(s.kind for s in FakeStream.instances), ["input", "output"])
        self.assertTrue(audio.drift_enabled)
        self.assertIn("drift", status)

    def test_the_worker_runs_on_the_thread_that_warmed_up(self):
        # Compiled CUDA graphs are recorded per thread, so the conversion has to run
        # where the warm-up did or its first blocks record them again (~1 s).
        thread = ConversionThread()
        warmed_on = thread.call(threading.current_thread)
        seen = set()
        try:
            audio, _ = self._run((0, 0), "auto", conversion_thread=thread,
                                 on_convert=lambda: seen.add(threading.current_thread()))
            self.assertEqual(seen, {warmed_on})
            self.assertIsNone(audio._worker)
            # The thread outlives stop() and still takes work, e.g. a restart.
            self.assertEqual(thread.call(lambda: 7), 7)
        finally:
            thread.close()
        self.assertFalse(thread._thread.is_alive())

    def test_an_error_on_the_conversion_thread_reaches_the_caller(self):
        thread = ConversionThread()
        try:
            with self.assertRaises(ZeroDivisionError):
                thread.call(lambda: 1 / 0)
            self.assertEqual(thread.call(lambda: "still alive"), "still alive")
        finally:
            thread.close()

    def test_different_hosts_always_use_separate_streams(self):
        audio, status = self._run((0, 1), "auto")
        self.assertEqual(sorted(s.kind for s in FakeStream.instances), ["input", "output"])
        self.assertIn("Windows WASAPI->Windows WDM-KS", status)


def callbacks_blocks(audio):
    return len(audio.inference_times)


if __name__ == "__main__":
    unittest.main()
