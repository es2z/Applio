"""Non-blocking realtime audio transport.

PortAudio callbacks only copy samples. Model inference, resampling and elastic
buffer maintenance run on a dedicated worker so a long GUI Chunk Size never
becomes the host callback block size.
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from collections import deque

import numpy as np
import sounddevice as sd
import soxr

from rvc.realtime.devices import (
    AudioDeviceRef,
    get_audio_device_registry,
    list_audio_device,
)
from rvc.realtime.runtime import (
    INTERNAL_SAMPLE_RATE,
    DEFAULT_MAX_EXTRA_BUFFER_MS,
    ElasticBufferController,
    FloatRingBuffer,
    RuntimeAudioShape,
    ceil_to_grid,
)


CALLBACK_SCRATCH_FRAMES = 65_536
STREAM_HEARTBEAT_TIMEOUT_SECONDS = 3.0
STREAM_CLOCK_PROBE_SECONDS = 1.0
STREAM_RATE_TOLERANCE = 0.05
MAX_REPORTED_IO_LATENCY_MS = 1_000.0
WORKER_READ_FRAMES = 8_192
MAX_CLOCK_CORRECTION_PPM = 1_000.0
CLOCK_CORRECTION_GAIN_PPM_PER_MS = 40.0
INPUT_BACKLOG_GRACE_MS = 100
OUTPUT_QUEUE_GRACE_MS = 100
RECOVERY_CROSSFADE_MS = 10


def _is_wasapi(device: AudioDeviceRef) -> bool:
    return "WASAPI" in device.host_api


def _is_asio(device: AudioDeviceRef) -> bool:
    return "ASIO" in device.host_api


class Audio:
    def __init__(
        self,
        callbacks,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        input_audio_gain: float = 1.0,
        output_audio_gain: float = 1.0,
        monitor_audio_gain: float = 1.0,
        monitor: bool = False,
        runtime_shape: RuntimeAudioShape | None = None,
        max_extra_buffer_ms: int = DEFAULT_MAX_EXTRA_BUFFER_MS,
    ):
        self.callbacks = callbacks
        self.runtime_shape = runtime_shape or RuntimeAudioShape.create(512)
        self.input_audio_gain = input_audio_gain
        self.output_audio_gain = output_audio_gain
        self.monitor_audio_gain = monitor_audio_gain
        self.use_monitor = monitor
        self.max_extra_buffer_ms = max(0, int(max_extra_buffer_ms))
        self.f0_up_key = f0_up_key
        self.index_rate = index_rate
        self.protect = protect
        self.volume_envelope = volume_envelope
        self.f0_autotune = f0_autotune
        self.f0_autotune_strength = f0_autotune_strength
        self.proposed_pitch = proposed_pitch
        self.proposed_pitch_threshold = proposed_pitch_threshold

        self.stream = None
        self.input_stream = None
        self.output_stream = None
        self.monitor = None
        self.running = False
        self.last_error: str | None = None
        self.reconnect_in_progress = False
        self.reconnect_success = False
        self.transport_mode = "stopped"

        self.input_ring: FloatRingBuffer | None = None
        self.internal_input_ring: FloatRingBuffer | None = None
        self.output_ring: FloatRingBuffer | None = None
        self.monitor_ring: FloatRingBuffer | None = None
        self.input_rate = INTERNAL_SAMPLE_RATE
        self.output_rate = INTERNAL_SAMPLE_RATE
        self.monitor_rate = INTERNAL_SAMPLE_RATE
        self.input_resampler = None
        self.output_resampler = None
        self.monitor_resampler = None
        self._separate_stream_clocks = False
        self.output_clock_correction_ppm = 0.0
        self.device_latency_ms = 0.0
        self.input_device_index: int | None = None
        self.output_device_index: int | None = None
        self.observed_input_rate = 0.0
        self.observed_output_rate = 0.0
        self.open_warnings: list[str] = []

        self._worker = None
        self._input_event = threading.Event()
        self._input_heartbeat = threading.Event()
        self._output_heartbeat = threading.Event()
        self._worker_ready = threading.Event()
        self._input_callback_frames = 0
        self._output_callback_frames = 0
        self._capture_scratch = np.zeros(CALLBACK_SCRATCH_FRAMES, dtype=np.float32)
        self._playback_started = False
        self._startup_reserve_remaining = 0
        self._monitor_playback_started = False
        self._capture_discontinuity = threading.Event()
        self._replace_output_on_next_write = False
        self._last_rendered_sample = 0.0

        self.inference_times = deque(maxlen=512)
        self.latency = 0.0
        self.input_overflows = 0
        self.output_underflows = 0
        self.callback_status_events = 0
        self._last_output_silent = True
        self._pending_underflows = 0
        self._warmup_times: list[float] = []
        self.stale_input_dropped_frames = 0
        self.stale_output_dropped_frames = 0
        self.backlog_recoveries = 0
        self.elastic = ElasticBufferController(
            max_extra_buffer_ms=self.max_extra_buffer_ms
        )

    def configure_warmup(self, timings_ms: list[float]) -> None:
        self._warmup_times = [float(value) for value in timings_ms if value >= 0]
        if self._warmup_times:
            p50 = float(np.percentile(self._warmup_times, 50))
            p95 = float(np.percentile(self._warmup_times, 95))
            reserve = max(10, ceil_to_grid(max(0.0, p95 - p50)))
            self.elastic = ElasticBufferController(
                reserve, max_extra_buffer_ms=self.max_extra_buffer_ms
            )

    def _process(self, samples: np.ndarray):
        return self.callbacks.change_voice(
            samples * self.input_audio_gain,
            self.f0_up_key,
            self.index_rate,
            self.protect,
            self.volume_envelope,
            self.f0_autotune,
            self.f0_autotune_strength,
            self.proposed_pitch,
            self.proposed_pitch_threshold,
        )

    def _capture_callback(self, indata: np.ndarray, frames: int, status) -> None:
        self._input_callback_frames += frames
        self._input_heartbeat.set()
        if status:
            self.callback_status_events += 1
        if not self.running or self.input_ring is None:
            return
        if frames <= len(self._capture_scratch):
            mono = self._capture_scratch[:frames]
            if indata.shape[1] == 1:
                mono[:] = indata[:, 0]
            else:
                np.sum(indata, axis=1, out=mono)
                mono /= indata.shape[1]
        else:
            # This is outside normal host callback sizes. Preserve transport
            # rather than writing past the preallocated callback scratch.
            mono = indata[:, 0]
        overwritten_before = self.input_ring.overwritten_samples
        written = self.input_ring.write(
            mono, blocking=False, overflow_policy="drop_oldest"
        )
        overwritten = self.input_ring.overwritten_samples - overwritten_before
        if written < frames or overwritten > 0:
            self.input_overflows += 1
            lost_native = overwritten + max(0, frames - written)
            self.stale_input_dropped_frames += round(
                lost_native * INTERNAL_SAMPLE_RATE / self.input_rate
            )
            self._capture_discontinuity.set()
        self._input_event.set()

    def input_callback(self, indata, frames, times, status):
        self._capture_callback(indata, frames, status)

    def _render_callback(self, outdata: np.ndarray, frames: int, status) -> None:
        self._output_callback_frames += frames
        self._output_heartbeat.set()
        outdata.fill(0)
        if status:
            self.callback_status_events += 1
        if not self.running or self.output_ring is None:
            return

        if not self._playback_started:
            if self.output_ring.available_approx <= 0:
                return
            if self._startup_reserve_remaining > 0:
                consumed = min(frames, self._startup_reserve_remaining)
                self._startup_reserve_remaining -= consumed
                if consumed == frames:
                    return
                target = outdata[consumed:, 0]
            else:
                self._playback_started = True
                target = outdata[:, 0]
        else:
            target = outdata[:, 0]

        read = self.output_ring.read_into(target, blocking=False)
        if read < len(target):
            self.output_underflows += 1
            # Do not acquire the elastic-controller lock in the callback. The
            # inference worker turns this signal into an additive 10 ms step.
            self._pending_underflows += 1
        if read:
            self._last_rendered_sample = float(target[read - 1])
        target *= self.output_audio_gain
        for channel in range(1, outdata.shape[1]):
            outdata[:, channel] = outdata[:, 0]

    def output_callback(self, outdata, frames, times, status):
        self._render_callback(outdata, frames, status)

    def audio_stream_callback(self, indata, outdata, frames, times, status):
        self._capture_callback(indata, frames, status)
        self._render_callback(outdata, frames, status)

    def audio_queue(self, outdata, frames, times, status):
        outdata.fill(0)
        if status:
            self.callback_status_events += 1
        if not self.running or self.monitor_ring is None:
            return
        read = self.monitor_ring.read_into(outdata[:, 0], blocking=False)
        if read:
            outdata[:, 0] *= self.monitor_audio_gain
        for channel in range(1, outdata.shape[1]):
            outdata[:, channel] = outdata[:, 0]

    @staticmethod
    def _device_channels(device: AudioDeviceRef, selected_channel: int) -> int:
        if selected_channel >= 0 and _is_asio(device):
            return 1
        return max(1, min(2, device.channels))

    @staticmethod
    def _extra_settings(
        device: AudioDeviceRef, exclusive: bool, selected_channel: int
    ):
        if _is_wasapi(device):
            return sd.WasapiSettings(
                exclusive=bool(exclusive), auto_convert=not bool(exclusive)
            )
        if _is_asio(device) and selected_channel >= 0:
            return sd.AsioSettings(channel_selectors=[selected_channel])
        return None

    @staticmethod
    def _supported_rate(
        input_device: AudioDeviceRef, output_device: AudioDeviceRef
    ) -> int:
        candidates = []
        for value in (
            INTERNAL_SAMPLE_RATE,
            input_device.default_samplerate,
            output_device.default_samplerate,
            44_100,
        ):
            rate = int(round(value))
            if rate not in candidates:
                candidates.append(rate)
        for rate in candidates:
            try:
                sd.check_input_settings(
                    device=input_device.index,
                    channels=max(1, min(2, input_device.max_input_channels)),
                    dtype=np.float32,
                    samplerate=rate,
                )
                sd.check_output_settings(
                    device=output_device.index,
                    channels=max(1, min(2, output_device.max_output_channels)),
                    dtype=np.float32,
                    samplerate=rate,
                )
                return rate
            except Exception:
                continue
        raise RuntimeError("Input and output devices have no compatible sample rate")

    @staticmethod
    def _can_use_native_duplex(
        input_device: AudioDeviceRef, output_device: AudioDeviceRef
    ) -> bool:
        """Return true only for one PortAudio device with both directions.

        Sharing a host API does not imply sharing a hardware sample clock. For
        example, a MOTU WASAPI capture endpoint and a VB-Audio WASAPI render
        endpoint are independent devices and must not be forced into one
        PortAudio full-duplex stream.
        """
        return bool(
            input_device.index == output_device.index
            and input_device.host_api_index == output_device.host_api_index
            and input_device.max_input_channels > 0
            and output_device.max_output_channels > 0
        )

    @staticmethod
    def _validate_observed_rate(
        direction: str, device_index: int, requested_rate: int, observed_rate: float
    ) -> None:
        relative_error = abs(float(observed_rate) - requested_rate) / requested_rate
        if relative_error > STREAM_RATE_TOLERANCE:
            raise RuntimeError(
                f"PA {device_index} {direction} device clock mismatch: requested "
                f"{requested_rate} Hz, observed approximately {observed_rate:.0f} Hz"
            )

    def _validate_reported_io_latency(self) -> None:
        if self.device_latency_ms > MAX_REPORTED_IO_LATENCY_MS:
            raise RuntimeError(
                "Audio driver reported an abnormal I/O latency of "
                f"{self.device_latency_ms:.1f} ms; the stream was closed instead "
                "of starting with multi-second delay"
            )

    def _probe_stream_clocks(self) -> None:
        """Reject streams whose callbacks do not match their requested rates."""
        if not self._input_heartbeat.wait(STREAM_HEARTBEAT_TIMEOUT_SECONDS):
            raise RuntimeError("Input stream opened but produced no callbacks")
        if not self._output_heartbeat.wait(STREAM_HEARTBEAT_TIMEOUT_SECONDS):
            raise RuntimeError("Output stream opened but requested no callbacks")

        input_start = self._input_callback_frames
        output_start = self._output_callback_frames
        started = time.perf_counter()
        time.sleep(STREAM_CLOCK_PROBE_SECONDS)
        elapsed = max(time.perf_counter() - started, 1e-9)
        self.observed_input_rate = (
            self._input_callback_frames - input_start
        ) / elapsed
        self.observed_output_rate = (
            self._output_callback_frames - output_start
        ) / elapsed

        self._validate_observed_rate(
            "input",
            int(self.input_device_index),
            self.input_rate,
            self.observed_input_rate,
        )
        self._validate_observed_rate(
            "output",
            int(self.output_device_index),
            self.output_rate,
            self.observed_output_rate,
        )

        # The probe deliberately runs before inference. Do not feed its captured
        # audio into the model when the real session begins.
        for ring in (self.input_ring, self.internal_input_ring, self.output_ring):
            if ring is not None:
                ring.clear()
        self._input_event.clear()
        self._capture_discontinuity.clear()
        self._playback_started = False
        self._startup_reserve_remaining = round(
            self.elastic.total_reserve_ms * self.output_rate / 1000
        )

    def _create_buffers(self, *, separate_stream_clocks: bool) -> None:
        self._separate_stream_clocks = separate_stream_clocks
        internal_grace_frames = round(
            INPUT_BACKLOG_GRACE_MS * INTERNAL_SAMPLE_RATE / 1000
        )
        internal_capacity = (
            self.runtime_shape.context_frames
            + internal_grace_frames
            + WORKER_READ_FRAMES
        )
        native_context_frames = math.ceil(
            self.runtime_shape.context_frames
            * self.input_rate
            / INTERNAL_SAMPLE_RATE
        )
        input_capacity = max(
            native_context_frames
            + round(INPUT_BACKLOG_GRACE_MS * self.input_rate / 1000)
            + WORKER_READ_FRAMES,
            WORKER_READ_FRAMES * 2,
        )
        native_hop_frames = math.ceil(
            self.runtime_shape.hop_frames
            * self.output_rate
            / INTERNAL_SAMPLE_RATE
        )
        maximum_reserve_ms = (
            self.elastic.base_reserve_ms
            + self.elastic.max_extra_buffer_ms
            + OUTPUT_QUEUE_GRACE_MS
        )
        output_capacity = max(
            native_hop_frames
            + round(maximum_reserve_ms * self.output_rate / 1000)
            + WORKER_READ_FRAMES,
            WORKER_READ_FRAMES * 2,
        )
        self.input_ring = FloatRingBuffer(input_capacity)
        self.internal_input_ring = FloatRingBuffer(internal_capacity)
        self.output_ring = FloatRingBuffer(output_capacity)
        if self.use_monitor:
            self.monitor_ring = FloatRingBuffer(output_capacity)

        self.input_resampler = (
            soxr.ResampleStream(
                self.input_rate,
                INTERNAL_SAMPLE_RATE,
                1,
                dtype="float32",
                quality="HQ",
            )
            if self.input_rate != INTERNAL_SAMPLE_RATE
            else None
        )
        self.output_resampler = (
            soxr.ResampleStream(
                INTERNAL_SAMPLE_RATE,
                self.output_rate,
                1,
                dtype="float32",
                quality="HQ",
                vr=separate_stream_clocks,
            )
            if self.output_rate != INTERNAL_SAMPLE_RATE or separate_stream_clocks
            else None
        )
        self.monitor_resampler = (
            soxr.ResampleStream(
                INTERNAL_SAMPLE_RATE,
                self.monitor_rate,
                1,
                dtype="float32",
                quality="HQ",
            )
            if self.use_monitor and self.monitor_rate != INTERNAL_SAMPLE_RATE
            else None
        )

        self._startup_reserve_remaining = round(
            self.elastic.total_reserve_ms * self.output_rate / 1000
        )

    def _fade_from_last_rendered(self, audio: np.ndarray) -> np.ndarray:
        """Join a realtime catch-up block without a hard sample discontinuity."""
        result = np.asarray(audio, dtype=np.float32).copy()
        fade_frames = min(
            len(result), round(RECOVERY_CROSSFADE_MS * self.output_rate / 1000)
        )
        if fade_frames:
            ramp = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
            result[:fade_frames] = (
                self._last_rendered_sample * (1.0 - ramp)
                + result[:fade_frames] * ramp
            )
        return result

    def _flush_model_history(self) -> None:
        vc = getattr(self.callbacks, "vc", None)
        if vc is None:
            return
        if getattr(vc, "sola_buffer", None) is not None:
            vc.sola_buffer.zero_()
        model = getattr(vc, "vc_model", None)
        if model is not None:
            model.flush_buffers()

    def _update_output_clock_ratio(self, output_frames: int) -> None:
        """Compensate independent input/output device-clock drift.

        This changes only the native-rate resampling ratio (bounded to ±1000
        ppm); it never changes Chunk, Hop, or model tensor shapes.
        """
        if not self._separate_stream_clocks or self.output_resampler is None:
            self.output_clock_correction_ppm = 0.0
            return
        available = self.output_ring.available if self.output_ring is not None else 0
        target = round(self.elastic.total_reserve_ms * self.output_rate / 1000)
        error_ms = (target - available) / self.output_rate * 1000
        requested = float(
            np.clip(
                error_ms * CLOCK_CORRECTION_GAIN_PPM_PER_MS,
                -MAX_CLOCK_CORRECTION_PPM,
                MAX_CLOCK_CORRECTION_PPM,
            )
        )
        # Low-pass the control value so inference jitter does not become pitch
        # modulation. set_io_ratio slews the remaining transition per block.
        self.output_clock_correction_ppm = (
            0.9 * self.output_clock_correction_ppm + 0.1 * requested
        )
        adjusted_rate = self.output_rate * (
            1.0 + self.output_clock_correction_ppm / 1_000_000
        )
        self.output_resampler.set_io_ratio(
            INTERNAL_SAMPLE_RATE,
            adjusted_rate,
            slew_len=max(1, int(output_frames)),
        )

    def _write_processed(self, output: np.ndarray) -> None:
        self._update_output_clock_ratio(len(output))
        output_native = (
            self.output_resampler.resample_chunk(output, last=False)
            if self.output_resampler is not None
            else output
        )
        replace = False
        if self.output_ring is not None:
            capacity_recovery = (
                self.output_ring.available + len(output_native)
                > self.output_ring.capacity
            )
            replace = self._replace_output_on_next_write or capacity_recovery
            if replace:
                if capacity_recovery and not self._replace_output_on_next_write:
                    self.backlog_recoveries += 1
                dropped = self.output_ring.available
                output_native = self._fade_from_last_rendered(output_native)
                self.output_ring.replace(output_native)
                self.stale_output_dropped_frames += dropped
                self._replace_output_on_next_write = False
            else:
                overwritten_before = self.output_ring.overwritten_samples
                self.output_ring.write(
                    output_native, overflow_policy="drop_oldest"
                )
                self.stale_output_dropped_frames += (
                    self.output_ring.overwritten_samples - overwritten_before
                )
        if self.use_monitor and self.monitor_ring is not None:
            monitor_native = (
                self.monitor_resampler.resample_chunk(output, last=False)
                if self.monitor_resampler is not None
                else output
            )
            if replace:
                self.monitor_ring.replace(monitor_native)
            else:
                self.monitor_ring.write(
                    monitor_native, overflow_policy="drop_oldest"
                )

    def _worker_loop(self) -> None:
        initial = True
        self._worker_ready.set()
        try:
            while self.running:
                self._input_event.wait(0.05)
                self._input_event.clear()
                if self.input_ring is None or self.internal_input_ring is None:
                    continue

                available = self.input_ring.available
                while available > 0:
                    raw = self.input_ring.read(min(available, WORKER_READ_FRAMES))
                    if not len(raw):
                        break
                    internal = (
                        self.input_resampler.resample_chunk(raw, last=False)
                        if self.input_resampler is not None
                        else raw
                    )
                    overwritten_before = self.internal_input_ring.overwritten_samples
                    self.internal_input_ring.write(
                        internal, overflow_policy="drop_oldest"
                    )
                    overwritten = (
                        self.internal_input_ring.overwritten_samples
                        - overwritten_before
                    )
                    if overwritten:
                        self.stale_input_dropped_frames += overwritten
                        self._capture_discontinuity.set()
                    available = self.input_ring.available

                discontinuity = self._capture_discontinuity.is_set()
                if discontinuity:
                    self._capture_discontinuity.clear()
                backlog_limit = self.runtime_shape.context_frames + round(
                    INPUT_BACKLOG_GRACE_MS * INTERNAL_SAMPLE_RATE / 1000
                )
                backlog = self.internal_input_ring.available
                if discontinuity or backlog > backlog_limit:
                    if backlog >= self.runtime_shape.context_frames:
                        dropped = self.internal_input_ring.trim_to_latest(
                            self.runtime_shape.context_frames
                        )
                    else:
                        # A callback was lost before a complete fresh context was
                        # available. Do not join audio from opposite sides of the
                        # gap inside the model history.
                        dropped = self.internal_input_ring.available
                        self.internal_input_ring.clear()
                    self.stale_input_dropped_frames += dropped
                    self.backlog_recoveries += 1
                    self._flush_model_history()
                    self._replace_output_on_next_write = True
                    initial = True

                required = (
                    self.runtime_shape.context_frames
                    if initial
                    else self.runtime_shape.hop_frames
                )
                while self.internal_input_ring.available >= required and self.running:
                    block = self.internal_input_ring.read(required)
                    started = time.perf_counter()
                    out_wav, _, perf, _ = self._process(block)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    if perf and len(perf) > 1:
                        elapsed_ms = float(perf[1])
                    self.latency = elapsed_ms
                    self.inference_times.append(elapsed_ms)

                    silent = bool(len(out_wav) == 0 or np.max(np.abs(out_wav)) < 1e-6)
                    self.elastic.record_inference(
                        elapsed_ms, self.runtime_shape.effective_hop_ms
                    )
                    if self._pending_underflows:
                        self._pending_underflows = 0
                        self.elastic.record_underflow()
                    shrink_frames = self.elastic.observe_silence(silent)
                    if shrink_frames < 0 and silent:
                        # Removing silence is click-free and lets the real queue
                        # follow the reduced 10 ms reserve target immediately.
                        remove = min(-shrink_frames, max(0, len(out_wav) - 1))
                        out_wav = out_wav[remove:]

                    self._last_output_silent = silent
                    self._write_processed(np.asarray(out_wav, dtype=np.float32))
                    initial = False
                    required = self.runtime_shape.hop_frames
        except Exception as error:
            self.last_error = f"Realtime worker failed: {type(error).__name__}: {error}"
            print(self.last_error)
            print(traceback.format_exc())
            self.running = False

    def _open_monitor(
        self,
        device: AudioDeviceRef,
        selected_channel: int,
        exclusive_mode: bool,
    ) -> None:
        channels = self._device_channels(device, selected_channel)
        kwargs = dict(
            callback=self.audio_queue,
            latency="low",
            dtype=np.float32,
            device=device.index,
            blocksize=0,
            samplerate=self.monitor_rate,
            channels=channels,
        )
        try:
            self.monitor = sd.OutputStream(
                **kwargs,
                extra_settings=self._extra_settings(
                    device, exclusive_mode, selected_channel
                ),
            )
        except sd.PortAudioError:
            if not (exclusive_mode and _is_wasapi(device)):
                raise
            self.open_warnings.append(
                f"monitor PA {device.index} rejected WASAPI exclusive; using shared"
            )
            self.monitor = sd.OutputStream(
                **kwargs,
                extra_settings=self._extra_settings(device, False, selected_channel),
            )
        self.monitor.start()

    def _open_duplex(
        self,
        input_device: AudioDeviceRef,
        output_device: AudioDeviceRef,
        exclusive_mode: bool,
        input_channel: int,
        output_channel: int,
    ) -> None:
        stream_rate = self._supported_rate(input_device, output_device)
        self.input_rate = self.output_rate = stream_rate
        self._create_buffers(separate_stream_clocks=False)
        kwargs = dict(
            callback=self.audio_stream_callback,
            latency="low",
            dtype=np.float32,
            device=(input_device.index, output_device.index),
            blocksize=0,
            samplerate=stream_rate,
            channels=(
                self._device_channels(input_device, input_channel),
                self._device_channels(output_device, output_channel),
            ),
        )
        try:
            self.stream = sd.Stream(
                **kwargs,
                extra_settings=(
                    self._extra_settings(input_device, exclusive_mode, input_channel),
                    self._extra_settings(
                        output_device, exclusive_mode, output_channel
                    ),
                ),
            )
        except sd.PortAudioError:
            if not (
                exclusive_mode
                and (_is_wasapi(input_device) or _is_wasapi(output_device))
            ):
                raise
            self.open_warnings.append(
                "duplex WASAPI exclusive was rejected; using shared"
            )
            self.stream = sd.Stream(
                **kwargs,
                extra_settings=(
                    self._extra_settings(input_device, False, input_channel),
                    self._extra_settings(output_device, False, output_channel),
                ),
            )
        self.stream.start()
        input_latency, output_latency = self.stream.latency
        self.device_latency_ms = float(input_latency + output_latency) * 1000
        self.transport_mode = f"duplex/{input_device.host_api}"
        self._validate_reported_io_latency()

    def _open_separate(
        self,
        input_device: AudioDeviceRef,
        output_device: AudioDeviceRef,
        exclusive_mode: bool,
        input_channel: int,
        output_channel: int,
    ) -> None:
        self.input_rate = int(round(input_device.default_samplerate))
        self.output_rate = int(round(output_device.default_samplerate))
        self._create_buffers(separate_stream_clocks=True)
        input_kwargs = dict(
            callback=self.input_callback,
            latency="low",
            dtype=np.float32,
            device=input_device.index,
            blocksize=0,
            samplerate=self.input_rate,
            channels=self._device_channels(input_device, input_channel),
        )
        try:
            self.input_stream = sd.InputStream(
                **input_kwargs,
                extra_settings=self._extra_settings(
                    input_device, exclusive_mode, input_channel
                ),
            )
        except sd.PortAudioError:
            if not (exclusive_mode and _is_wasapi(input_device)):
                raise
            self.open_warnings.append(
                f"input PA {input_device.index} rejected WASAPI exclusive; using shared"
            )
            self.input_stream = sd.InputStream(
                **input_kwargs,
                extra_settings=self._extra_settings(input_device, False, input_channel),
            )

        output_kwargs = dict(
            callback=self.output_callback,
            latency="low",
            dtype=np.float32,
            device=output_device.index,
            blocksize=0,
            samplerate=self.output_rate,
            channels=self._device_channels(output_device, output_channel),
        )
        try:
            self.output_stream = sd.OutputStream(
                **output_kwargs,
                extra_settings=self._extra_settings(
                    output_device, exclusive_mode, output_channel
                ),
            )
        except sd.PortAudioError:
            if self.input_stream is not None:
                self.input_stream.close()
                self.input_stream = None
            if not (exclusive_mode and _is_wasapi(output_device)):
                raise
            self.open_warnings.append(
                f"output PA {output_device.index} rejected WASAPI exclusive; using shared"
            )
            # Recreate input too: some WASAPI drivers invalidate a not-yet-started
            # stream after another exclusive open fails in the same process.
            self.input_stream = sd.InputStream(
                **input_kwargs,
                extra_settings=self._extra_settings(input_device, False, input_channel),
            )
            self.output_stream = sd.OutputStream(
                **output_kwargs,
                extra_settings=self._extra_settings(output_device, False, output_channel),
            )
        self.output_stream.start()
        self.input_stream.start()
        self.device_latency_ms = float(
            self.input_stream.latency + self.output_stream.latency
        ) * 1000
        self.transport_mode = (
            f"separate/{input_device.host_api}->{output_device.host_api}"
        )
        self._validate_reported_io_latency()

    def start(
        self,
        input_device: AudioDeviceRef,
        output_device: AudioDeviceRef,
        output_monitor: AudioDeviceRef | None = None,
        exclusive_mode: bool = False,
        asio_input_channel: int = -1,
        asio_output_channel: int = -1,
        asio_output_monitor_channel: int = -1,
        **_legacy,
    ) -> None:
        self.stop()
        if input_device.direction != "input" or output_device.direction != "output":
            raise ValueError("Audio device direction does not match the selected role")

        self.running = True
        self.last_error = None
        self.open_warnings.clear()
        self.input_device_index = input_device.index
        self.output_device_index = output_device.index
        self.observed_input_rate = 0.0
        self.observed_output_rate = 0.0
        self._input_callback_frames = 0
        self._output_callback_frames = 0
        self.stale_input_dropped_frames = 0
        self.stale_output_dropped_frames = 0
        self.backlog_recoveries = 0
        self._input_heartbeat.clear()
        self._output_heartbeat.clear()
        self._worker_ready.clear()
        self._capture_discontinuity.clear()
        self._playback_started = False
        self._replace_output_on_next_write = False
        self._last_rendered_sample = 0.0
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="realtime-inference",
            daemon=True,
        )

        native_duplex = self._can_use_native_duplex(input_device, output_device)
        try:
            if native_duplex:
                try:
                    # A single stream is safe only when PortAudio exposes one
                    # actual device with both input and output directions.
                    self._open_duplex(
                        input_device,
                        output_device,
                        exclusive_mode,
                        asio_input_channel,
                        asio_output_channel,
                    )
                except Exception as duplex_error:
                    if self.stream is not None:
                        self.stream.close()
                        self.stream = None
                    print(
                        f"[Audio] Duplex open failed ({duplex_error}); using separate streams"
                    )
                    self._open_separate(
                        input_device,
                        output_device,
                        exclusive_mode,
                        asio_input_channel,
                        asio_output_channel,
                    )
            else:
                self._open_separate(
                    input_device,
                    output_device,
                    exclusive_mode,
                    asio_input_channel,
                    asio_output_channel,
                )

            if output_monitor is not None:
                self.monitor_rate = int(round(output_monitor.default_samplerate))
                # Recreate the monitor resampler/ring now that its native rate is known.
                output_capacity_ms = (
                    self.output_ring.capacity / self.output_rate * 1000
                    if self.output_ring is not None
                    else self.runtime_shape.effective_hop_ms
                )
                self.monitor_ring = FloatRingBuffer(
                    max(round(output_capacity_ms * self.monitor_rate / 1000), 4096)
                )
                self.monitor_resampler = (
                    soxr.ResampleStream(
                        INTERNAL_SAMPLE_RATE,
                        self.monitor_rate,
                        1,
                        dtype="float32",
                        quality="HQ",
                    )
                    if self.monitor_rate != INTERNAL_SAMPLE_RATE
                    else None
                )
                self._open_monitor(
                    output_monitor,
                    asio_output_monitor_channel,
                    exclusive_mode,
                )

            self._probe_stream_clocks()
            self._worker.start()
            if not self._worker_ready.wait(STREAM_HEARTBEAT_TIMEOUT_SECONDS):
                raise RuntimeError("Realtime inference worker did not start")
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self.running = False
        self._input_event.set()
        for name in ("stream", "input_stream", "output_stream", "monitor"):
            stream = getattr(self, name, None)
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
                setattr(self, name, None)
        worker = self._worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        self._worker = None
        for ring in (
            self.input_ring,
            self.internal_input_ring,
            self.output_ring,
            self.monitor_ring,
        ):
            if ring is not None:
                ring.clear()
        self.transport_mode = "stopped"

    def clear_buffers(self) -> None:
        for ring in (
            self.input_ring,
            self.internal_input_ring,
            self.output_ring,
            self.monitor_ring,
        ):
            if ring is not None:
                ring.clear()
        self._flush_model_history()

    def status_text(self) -> str:
        if self.last_error:
            return f"Error: {self.last_error}"
        values = list(self.inference_times)
        p50 = float(np.percentile(values, 50)) if values else 0.0
        p95 = float(np.percentile(values, 95)) if values else 0.0
        compile_parts = []
        compile_warnings = []
        for status in self.callbacks.compile_statuses():
            rebuilt = "/cache-rebuilt" if status.cache_rebuilt else ""
            compile_parts.append(f"{status.component}={status.backend}{rebuilt}")
            if status.error:
                compile_warnings.append(
                    f"{status.component}: {status.error[:160]}"
                )
        compile_text = ", ".join(compile_parts) if compile_parts else "eager"
        overload = ", OVERLOADED" if self.elastic.overloaded else ""
        open_warning = (
            f" | {'; '.join(self.open_warnings)}" if self.open_warnings else ""
        )
        pipeline_lower_bound = (
            self.runtime_shape.effective_chunk_ms
            + p95
            + self.elastic.total_reserve_ms
            + self.device_latency_ms
        )
        raw_input_ms = (
            self.input_ring.available / self.input_rate * 1000
            if self.input_ring is not None
            else 0.0
        )
        internal_input_ms = (
            self.internal_input_ring.available / INTERNAL_SAMPLE_RATE * 1000
            if self.internal_input_ring is not None
            else 0.0
        )
        output_queue_ms = (
            self.output_ring.available / self.output_rate * 1000
            if self.output_ring is not None
            else 0.0
        )
        stale_input_ms = (
            self.stale_input_dropped_frames / INTERNAL_SAMPLE_RATE * 1000
        )
        stale_output_ms = (
            self.stale_output_dropped_frames / self.output_rate * 1000
            if self.output_rate
            else 0.0
        )
        route_text = (
            f"route PA{self.input_device_index}@{self.input_rate}Hz"
            f"->PA{self.output_device_index}@{self.output_rate}Hz "
            f"(clock {self.observed_input_rate:.0f}/{self.observed_output_rate:.0f}Hz)"
        )
        return (
            f"{self.transport_mode} | {route_text} "
            f"| Chunk {self.runtime_shape.effective_chunk_ms:.1f} ms "
            f"| Hop {self.runtime_shape.effective_hop_ms:.1f} ms "
            f"| infer p50/p95 {p50:.1f}/{p95:.1f} ms "
            f"| reserve {self.elastic.total_reserve_ms} ms "
            f"| I/O {self.device_latency_ms:.1f} ms "
            f"| lower-bound est. {pipeline_lower_bound:.1f} ms "
            f"| queue in/out {raw_input_ms + internal_input_ms:.1f}/"
            f"{output_queue_ms:.1f} ms "
            f"| drift {self.output_clock_correction_ppm:+.0f} ppm "
            f"| xruns in/out {self.input_overflows}/{self.output_underflows} "
            f"| catch-up {self.backlog_recoveries} "
            f"(dropped in/out {stale_input_ms:.1f}/{stale_output_ms:.1f} ms) "
            f"| compile {compile_text}{overload}{open_warning}"
            + (
                f" | compile warning {'; '.join(compile_warnings)}"
                if compile_warnings
                else ""
            )
        )


# Compatibility aliases for older imports.
ServerAudioDevice = AudioDeviceRef
get_input_audio_device_registry = get_audio_device_registry
