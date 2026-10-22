"""The ring transport: `transport = "ring"` in rvc/realtime/options.py.

The legacy transport (rvc/realtime/audio.py) opens the devices with blocksize = Chunk
Size and converts inside the callback, so a block that has just been converted is
handed to the device as the *next* buffer to play: on top of the chunk itself, the
output waits about one more chunk. WDM-KS output in the separate-stream path hides
part of that, which is why the reference setup used WASAPI -> WDM-KS.

Here the devices run with their own small buffers (blocksize 0, latency "low") and the
callbacks only copy samples into and out of ring buffers. A worker thread converts a
block as soon as one has arrived and writes it to the output ring, which holds only
enough to cover the processing time varying (rvc/realtime/ring_buffer.py,
ReserveController). With two device clocks (separate streams) the drift between them
is taken up by a variable-ratio resampler on the output, bounded to +-1000 ppm.

What the conversion sees is exactly what the legacy transport hands it: blocks of
Chunk Size samples at 48 kHz, mono, input gain applied. The sound is only changed
where the reserve has to be corrected (quiet output only, never voice) and, with
separate streams, by the resampler (flat to 21 kHz, -0.4 dB at 22 kHz).

Ported from the discarded new_main_testversion branch (rvc/realtime/audio.py, 29eae3d),
keeping this branch's device indices, start() signature and status attributes.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from collections import deque

import numpy as np
import sounddevice as sd

from rvc.realtime.ring_buffer import FloatRingBuffer, ReserveController

INTERNAL_RATE = 48000
STARTUP_TIMEOUT_S = 3.0
QUIET_RMS = 1e-3  # -60 dBFS: only output this quiet is ever trimmed or padded
MONITOR_SLACK_MS = 200
CORRECTION_FADE = 96  # samples
WORKER_READ_FRAMES = 8192
SWITCH_INTERVAL_S = 0.001


def _soxr():
    import soxr

    return soxr


def query_device(index):
    """Name, host API, channels and default rate of one PortAudio device, without
    opening anything (list_audio_device() test-opens every device)."""
    info = sd.query_devices(index)
    host_api = sd.query_hostapis(info["hostapi"])["name"]
    return {
        "index": int(index),
        "name": info["name"],
        "host_api": host_api,
        "max_input_channels": int(info["max_input_channels"]),
        "max_output_channels": int(info["max_output_channels"]),
        "default_samplerate": float(info["default_samplerate"]),
    }


def _extra_settings(device, exclusive, asio_channel):
    if "WASAPI" in device["host_api"]:
        return sd.WasapiSettings(exclusive=bool(exclusive), auto_convert=not exclusive)
    if "ASIO" in device["host_api"] and asio_channel is not None and asio_channel >= 0:
        return sd.AsioSettings(channel_selectors=[int(asio_channel)])
    return None


def _channels(device, kind, asio_channel):
    """The same channel counts the legacy transport opens with."""
    if "ASIO" in device["host_api"] and asio_channel is not None and asio_channel >= 0:
        return 1
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    return max(1, device[key])


def _quietest_start(audio, frames):
    best_start, best_energy = 0, np.inf
    stride = max(1, frames // 2)
    for start in range(0, max(1, len(audio) - frames + 1), stride):
        segment = audio[start : start + frames]
        energy = float(np.dot(segment, segment))
        if energy < best_energy:
            best_start, best_energy = start, energy
    return best_start


def pad_quiet(audio, frames):
    """Lengthen quiet audio by `frames` samples: play its quietest stretch twice,
    crossfading from what would have followed into the repeat."""
    if frames <= 0 or len(audio) == 0:
        return audio
    frames = min(frames, len(audio))
    start = _quietest_start(audio, frames)
    end = start + frames
    segment = audio[start:end].copy()
    fade = min(CORRECTION_FADE, frames // 4, len(audio) - end)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        segment[:fade] = audio[end : end + fade] * (1 - ramp) + segment[:fade] * ramp
    return np.concatenate((audio[:end], segment, audio[end:])).astype(np.float32)


def trim_quiet(audio, frames):
    """Shorten quiet audio by `frames` samples: cut its quietest stretch out,
    crossfading from the audio before the cut into the audio after it."""
    if frames <= 0 or len(audio) <= frames:
        return audio
    start = _quietest_start(audio, frames)
    end = start + frames
    fade = min(CORRECTION_FADE, start)
    if fade <= 0:
        return np.concatenate((audio[:start], audio[end:])).astype(np.float32)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    joined = audio[start - fade : start] * (1 - ramp) + audio[end - fade : end] * ramp
    return np.concatenate((audio[: start - fade], joined, audio[end:])).astype(np.float32)


class RingAudio:
    """Drop-in for rvc.realtime.audio.Audio with the ring transport."""

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
        block_frame: int = 4096,
        stream_mode: str = "auto",
        warmup_timings_ms=(),
        algorithmic_delay_ms: float = 0.0,
    ):
        self.callbacks = callbacks
        self.f0_up_key = f0_up_key
        self.index_rate = index_rate
        self.protect = protect
        self.volume_envelope = volume_envelope
        self.f0_autotune = f0_autotune
        self.f0_autotune_strength = f0_autotune_strength
        self.proposed_pitch = proposed_pitch
        self.proposed_pitch_threshold = proposed_pitch_threshold
        self.input_audio_gain = input_audio_gain
        self.output_audio_gain = output_audio_gain
        self.monitor_audio_gain = monitor_audio_gain
        self.use_monitor = monitor
        self.block_frame = int(block_frame)
        self.stream_mode = stream_mode
        self.warmup_timings_ms = list(warmup_timings_ms)
        self.algorithmic_delay_ms = float(algorithmic_delay_ms)

        # Attributes the Realtime tab reads, as on the legacy Audio.
        self.latency = 0.0
        self.last_error = None
        self.reconnect_in_progress = False
        self.reconnect_success = False
        self.running = False

        self.stream = self.input_stream = self.output_stream = self.monitor = None
        self.transport_mode = "stopped"
        self.open_warnings = []
        self.input_rate = self.output_rate = self.monitor_rate = INTERNAL_RATE
        self.input_resampler = self.output_resampler = self.monitor_resampler = None
        self.drift_enabled = False
        self.device_latency_ms = 0.0
        self.fatal_error = None

        self.input_ring = self.internal_ring = self.output_ring = self.monitor_ring = None
        self.controller = ReserveController(self.warmup_timings_ms)
        self.inference_times = deque(maxlen=512)
        self.queue_delays = deque(maxlen=32)
        self.first_elapsed_ms = None
        self.startup_padding_ms = 0.0
        self.input_overflows = 0
        self.output_underflows = 0
        self.callback_status_events = 0
        self._pending_underflows = 0
        self._pending_overflow = False
        self._primed = False
        self._playback_started = False
        self._monitor_started = False
        self._worker = None
        self._input_event = threading.Event()
        self._input_heartbeat = threading.Event()
        self._output_heartbeat = threading.Event()
        self._switch_interval = None
        self._scratch = np.zeros(65536, dtype=np.float32)
        self.clock = time.perf_counter  # replaced in tests
        self._render_at, self._render_frames = None, 0

    # ------------------------------------------------------------------ callbacks
    def _capture(self, indata, frames, status):
        self._input_heartbeat.set()
        if status:
            self.callback_status_events += 1
            if getattr(status, "input_overflow", False):
                self.input_overflows += 1
                self._pending_overflow = True
        if not self.running or self.input_ring is None:
            return
        if indata.shape[1] == 1:
            mono = indata[:, 0]
        elif frames <= len(self._scratch):
            mono = self._scratch[:frames]
            np.mean(indata, axis=1, out=mono)
        else:
            mono = indata.mean(axis=1)
        if self.input_ring.write(mono, blocking=False) < frames:
            self.input_overflows += 1
            self._pending_overflow = True
        self._input_event.set()

    def _render(self, outdata, frames, status):
        self._output_heartbeat.set()
        outdata.fill(0)
        if status:
            self.callback_status_events += 1
        if not self.running or self.output_ring is None:
            return
        self._render_at, self._render_frames = self.clock(), frames
        if not self._playback_started:
            if self.output_ring.available_approx <= 0:
                return
            self._playback_started = True
        target = outdata[:, 0]
        if self.output_ring.read_into(target, blocking=False) < frames:
            self.output_underflows += 1
            self._pending_underflows += 1
        if self.output_audio_gain != 1.0:
            target *= self.output_audio_gain
        for channel in range(1, outdata.shape[1]):
            outdata[:, channel] = target

    def input_callback(self, indata, frames, times, status):
        self._capture(indata, frames, status)

    def output_callback(self, outdata, frames, times, status):
        self._render(outdata, frames, status)

    def duplex_callback(self, indata, outdata, frames, times, status):
        self._capture(indata, frames, status)
        self._render(outdata, frames, status)

    def monitor_callback(self, outdata, frames, times, status):
        outdata.fill(0)
        if not self.running or self.monitor_ring is None:
            return
        if not self._monitor_started:
            if self.monitor_ring.available_approx <= 0:
                return
            self._monitor_started = True
        target = outdata[:, 0]
        self.monitor_ring.read_into(target, blocking=False)
        if self.monitor_audio_gain != 1.0:
            target *= self.monitor_audio_gain
        for channel in range(1, outdata.shape[1]):
            outdata[:, channel] = target

    # --------------------------------------------------------------------- worker
    def _process(self, block):
        return self.callbacks.change_voice(
            block * self.input_audio_gain,
            self.f0_up_key,
            self.index_rate,
            self.protect,
            self.volume_envelope,
            self.f0_autotune,
            self.f0_autotune_strength,
            self.proposed_pitch,
            self.proposed_pitch_threshold,
        )

    def _deliver(self, out, elapsed_ms):
        """Write one converted block, correcting the output reserve if it is quiet."""
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        level_ms = self.queued_ms()
        if self._pending_underflows:
            self._pending_underflows = 0
            self.controller.record_underflow()
        if not self._primed:
            self._primed = True
            self.first_elapsed_ms = elapsed_ms
            self.startup_padding_ms = self.controller.startup_padding_ms(elapsed_ms)
            pad = int(round(self.startup_padding_ms * INTERNAL_RATE / 1000))
            out = np.concatenate((np.zeros(pad, dtype=np.float32), out))
            self.queue_delays.append(elapsed_ms + self.startup_padding_ms)
        else:
            self.queue_delays.append(elapsed_ms + level_ms)
            quiet = float(np.sqrt(np.mean(np.square(out)))) < QUIET_RMS if len(out) else True
            correction_ms = self.controller.update(
                elapsed_ms, level_ms, quiet, drift=self.drift_enabled,
                block_s=self.block_frame / INTERNAL_RATE,
            )
            if correction_ms:
                frames = int(round(correction_ms * INTERNAL_RATE / 1000))
                frames = int(np.clip(frames, -(len(out) // 2), len(out) // 2))
                out = pad_quiet(out, frames) if frames > 0 else trim_quiet(out, -frames)
        self._write_output(out)

    def queued_ms(self):
        """How long until a sample written now is handed to the device: what the
        output ring holds, plus the rest of the device period that is playing.

        The ring is emptied a whole device period at a time, so the ring alone jumps
        by a period depending on when it is looked at; with two clocks that phase
        slips slowly and the drift correction would chase it.
        """
        queued = self.output_ring.available / self.output_rate * 1000.0
        if self._render_at is not None and self._playback_started:
            period = self._render_frames / self.output_rate
            remaining = self._render_at + period - self.clock()
            queued += min(max(remaining, 0.0), period) * 1000.0
        return queued

    def _write_output(self, out):
        if self.output_resampler is not None:
            if self.drift_enabled:
                rate = self.output_rate * (1.0 + self.controller.drift_ppm / 1e6)
                self.output_resampler.set_io_ratio(
                    INTERNAL_RATE, rate, slew_len=max(1, len(out))
                )
            native = self.output_resampler.resample_chunk(out)
        else:
            native = out
        self.output_ring.write(native)
        if self.monitor_ring is not None:
            monitor = (
                self.monitor_resampler.resample_chunk(out)
                if self.monitor_resampler is not None
                else out
            )
            self.monitor_ring.write(monitor)
            # The monitor runs on its own clock and nothing is listening for timing
            # there, so it is only kept from drifting late.
            limit = int(
                (self.block_frame / INTERNAL_RATE + MONITOR_SLACK_MS / 1000)
                * self.monitor_rate
            )
            excess = self.monitor_ring.available - limit
            if excess > 0:
                self.monitor_ring.discard(excess)

    def pump(self):
        """Move captured audio to 48 kHz and convert every whole block there is."""
        while True:
            raw = self.input_ring.read(min(self.input_ring.available, WORKER_READ_FRAMES))
            if not len(raw):
                break
            internal = (
                self.input_resampler.resample_chunk(raw)
                if self.input_resampler is not None
                else raw
            )
            self.internal_ring.write(internal)
        while self.running and self.internal_ring.available >= self.block_frame:
            block = self.internal_ring.read(self.block_frame)
            if self._pending_overflow:
                # Samples were dropped, so streaming state no longer lines up.
                self._pending_overflow = False
                self.callbacks.reset_stream()
            started = self.clock()
            out_wav = self._process(block)[0]
            elapsed_ms = (self.clock() - started) * 1000.0
            self.latency = elapsed_ms
            self.inference_times.append(elapsed_ms)
            self._deliver(out_wav, elapsed_ms)

    def _worker_loop(self):
        try:
            while self.running:
                self._input_event.wait(0.05)
                self._input_event.clear()
                if self.input_ring is not None:
                    self.pump()
        except Exception as error:
            self.fatal_error = f"Realtime worker failed: {type(error).__name__}: {error}"
            self.last_error = self.fatal_error
            print(self.fatal_error)
            print(traceback.format_exc())
            self.running = False

    # ---------------------------------------------------------------- open / close
    def _supported(self, device, kind, rate, channels, extra):
        check = sd.check_input_settings if kind == "input" else sd.check_output_settings
        try:
            check(device=device["index"], channels=channels, dtype="float32",
                  samplerate=rate, extra_settings=extra)
            return True
        except Exception:
            return False

    def _rate_for(self, device, kind, channels, extra):
        for rate in (INTERNAL_RATE, int(round(device["default_samplerate"]))):
            if self._supported(device, kind, rate, channels, extra):
                return rate
        return int(round(device["default_samplerate"]))

    def _allocate(self, separate_clocks):
        soxr = _soxr()
        self.input_ring = FloatRingBuffer(max(int(self.input_rate * 8), WORKER_READ_FRAMES * 4))
        self.internal_ring = FloatRingBuffer(max(self.block_frame * 4, INTERNAL_RATE * 8))
        self.output_ring = FloatRingBuffer(
            int(self.output_rate * (self.block_frame / INTERNAL_RATE + 8))
        )
        self.input_resampler = (
            soxr.ResampleStream(self.input_rate, INTERNAL_RATE, 1, dtype="float32", quality="HQ")
            if self.input_rate != INTERNAL_RATE
            else None
        )
        self.drift_enabled = False
        self.output_resampler = None
        if separate_clocks:
            try:
                self.output_resampler = soxr.ResampleStream(
                    INTERNAL_RATE, self.output_rate, 1, dtype="float32", quality="HQ", vr=True
                )
                self.drift_enabled = True
            except TypeError:
                self.open_warnings.append("soxr has no variable-rate mode; no drift correction")
        if self.output_resampler is None and self.output_rate != INTERNAL_RATE:
            self.output_resampler = soxr.ResampleStream(
                INTERNAL_RATE, self.output_rate, 1, dtype="float32", quality="HQ"
            )

    def _open_duplex(self, inp, out, exclusive, in_ch, out_ch):
        channels = (_channels(inp, "input", in_ch), _channels(out, "output", out_ch))

        def open_with(excl):
            extras = (_extra_settings(inp, excl, in_ch), _extra_settings(out, excl, out_ch))
            rate = INTERNAL_RATE
            for candidate in (INTERNAL_RATE, int(round(inp["default_samplerate"])),
                              int(round(out["default_samplerate"]))):
                if self._supported(inp, "input", candidate, channels[0], extras[0]) and \
                        self._supported(out, "output", candidate, channels[1], extras[1]):
                    rate = candidate
                    break
            self.input_rate = self.output_rate = rate
            self._allocate(separate_clocks=False)
            return sd.Stream(
                callback=self.duplex_callback, latency="low", dtype="float32",
                device=(inp["index"], out["index"]), blocksize=0, samplerate=rate,
                channels=channels, extra_settings=extras,
            )

        try:
            self.stream = open_with(exclusive)
        except sd.PortAudioError:
            if not (exclusive and ("WASAPI" in inp["host_api"] or "WASAPI" in out["host_api"])):
                raise
            self.open_warnings.append("duplex WASAPI exclusive was rejected; using shared")
            self.stream = open_with(False)
        self.stream.start()
        in_latency, out_latency = self.stream.latency
        self.device_latency_ms = float(in_latency + out_latency) * 1000
        self.transport_mode = f"ring duplex/{inp['host_api']}"

    def _open_separate(self, inp, out, exclusive, in_ch, out_ch):
        in_channels = _channels(inp, "input", in_ch)
        out_channels = _channels(out, "output", out_ch)

        def open_input(excl):
            extra = _extra_settings(inp, excl, in_ch)
            self.input_rate = self._rate_for(inp, "input", in_channels, extra)
            return sd.InputStream(
                callback=self.input_callback, latency="low", dtype="float32",
                device=inp["index"], blocksize=0, samplerate=self.input_rate,
                channels=in_channels, extra_settings=extra,
            )

        def open_output(excl):
            extra = _extra_settings(out, excl, out_ch)
            self.output_rate = self._rate_for(out, "output", out_channels, extra)
            return sd.OutputStream(
                callback=self.output_callback, latency="low", dtype="float32",
                device=out["index"], blocksize=0, samplerate=self.output_rate,
                channels=out_channels, extra_settings=extra,
            )

        try:
            self.input_stream = open_input(exclusive)
        except sd.PortAudioError:
            if not (exclusive and "WASAPI" in inp["host_api"]):
                raise
            self.open_warnings.append("input WASAPI exclusive was rejected; using shared")
            self.input_stream = open_input(False)
        try:
            self.output_stream = open_output(exclusive)
        except sd.PortAudioError:
            if not (exclusive and "WASAPI" in out["host_api"]):
                raise
            self.open_warnings.append("output WASAPI exclusive was rejected; using shared")
            self.output_stream = open_output(False)
        self._allocate(separate_clocks=True)
        self.output_stream.start()
        self.input_stream.start()
        self.device_latency_ms = float(
            self.input_stream.latency + self.output_stream.latency
        ) * 1000
        self.transport_mode = f"ring separate/{inp['host_api']}->{out['host_api']}"

    def _open_monitor(self, device, exclusive, channel):
        soxr = _soxr()
        channels = _channels(device, "output", channel)

        def open_with(excl):
            extra = _extra_settings(device, excl, channel)
            self.monitor_rate = self._rate_for(device, "output", channels, extra)
            return sd.OutputStream(
                callback=self.monitor_callback, latency="low", dtype="float32",
                device=device["index"], blocksize=0, samplerate=self.monitor_rate,
                channels=channels, extra_settings=extra,
            )

        try:
            self.monitor = open_with(exclusive)
        except sd.PortAudioError:
            if not (exclusive and "WASAPI" in device["host_api"]):
                raise
            self.open_warnings.append("monitor WASAPI exclusive was rejected; using shared")
            self.monitor = open_with(False)
        self.monitor_ring = FloatRingBuffer(int(self.monitor_rate * 8))
        self.monitor_resampler = (
            soxr.ResampleStream(INTERNAL_RATE, self.monitor_rate, 1, dtype="float32", quality="HQ")
            if self.monitor_rate != INTERNAL_RATE
            else None
        )
        self.monitor.start()

    def start(
        self,
        input_device_id: int,
        output_device_id: int,
        output_monitor_id: int = None,
        exclusive_mode: bool = False,
        asio_input_channel: int = -1,
        asio_output_channel: int = -1,
        asio_output_monitor_channel: int = -1,
        read_chunk_size: int = None,
    ):
        """Same signature as the legacy Audio.start. read_chunk_size is not needed:
        the block comes from the VoiceChanger."""
        self.stop()
        inp = query_device(input_device_id)
        out = query_device(output_device_id)
        self.running = True
        self.last_error = self.fatal_error = None
        self.open_warnings = []
        self._input_heartbeat.clear()
        self._output_heartbeat.clear()
        self._primed = self._playback_started = self._monitor_started = False
        self._render_at, self._render_frames = None, 0
        self.controller = ReserveController(self.warmup_timings_ms)
        self.inference_times.clear()
        self.queue_delays.clear()
        # The callbacks need the GIL within one device period (a few ms), while the
        # worker runs Python between CUDA calls; switch threads more often than 5 ms.
        self._switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(min(self._switch_interval, SWITCH_INTERVAL_S))
        try:
            same_host = inp["host_api"] == out["host_api"]
            if self.stream_mode == "auto" and same_host:
                try:
                    self._open_duplex(inp, out, exclusive_mode, asio_input_channel, asio_output_channel)
                except Exception as error:
                    if self.stream is not None:
                        self.stream.close()
                        self.stream = None
                    print(f"[Ring transport] Duplex open failed ({error}); using separate streams")
                    self._open_separate(inp, out, exclusive_mode, asio_input_channel, asio_output_channel)
            else:
                self._open_separate(inp, out, exclusive_mode, asio_input_channel, asio_output_channel)
            if self.use_monitor and output_monitor_id is not None:
                self._open_monitor(query_device(output_monitor_id), exclusive_mode,
                                   asio_output_monitor_channel)
            self._worker = threading.Thread(
                target=self._worker_loop, name="realtime-ring-worker", daemon=True
            )
            self._worker.start()
            if not self._input_heartbeat.wait(STARTUP_TIMEOUT_S):
                raise RuntimeError("Input stream opened but produced no callbacks")
            if not self._output_heartbeat.wait(STARTUP_TIMEOUT_S):
                raise RuntimeError("Output stream opened but requested no callbacks")
            print(f"[Ring transport] {self.transport_mode}, in {self.input_rate} Hz, "
                  f"out {self.output_rate} Hz, device latency {self.device_latency_ms:.1f} ms"
                  + (f" ({'; '.join(self.open_warnings)})" if self.open_warnings else ""))
        except Exception:
            self.stop()
            raise

    def stop(self):
        self.running = False
        self._input_event.set()
        for name in ("stream", "input_stream", "output_stream", "monitor"):
            stream = getattr(self, name, None)
            if stream is not None:
                for action in (stream.stop, stream.close):
                    try:
                        action()
                    except Exception:
                        pass
                setattr(self, name, None)
        worker = self._worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=5.0)
        self._worker = None
        for ring in (self.input_ring, self.internal_ring, self.output_ring, self.monitor_ring):
            if ring is not None:
                ring.clear()
        if self._switch_interval is not None:
            sys.setswitchinterval(self._switch_interval)
            self._switch_interval = None
        self.transport_mode = "stopped"

    def clear_buffers(self):
        for ring in (self.input_ring, self.internal_ring, self.output_ring, self.monitor_ring):
            if ring is not None:
                ring.clear()

    # --------------------------------------------------------------------- status
    def estimated_latency_ms(self):
        """Input to output, the same for every sample of a block: one chunk to collect
        it, the conversion window's own delay, the time from the block being complete
        to its first sample playing (processing + what the output ring still held),
        and the devices. The middle term is the median over recent blocks."""
        block_ms = self.block_frame / INTERNAL_RATE * 1000
        queued = float(np.median(np.asarray(self.queue_delays))) if self.queue_delays else 0.0
        return block_ms + self.algorithmic_delay_ms + queued + self.device_latency_ms

    def status_text(self):
        if self.fatal_error:
            return f"Error: {self.fatal_error}"
        values = np.asarray(self.inference_times) if self.inference_times else np.zeros(1)
        p50, p95 = float(np.percentile(values, 50)), float(np.percentile(values, 95))
        level = (
            self.output_ring.available / self.output_rate * 1000 if self.output_ring else 0.0
        )
        text = (
            f"{self.transport_mode} | Chunk {self.block_frame / INTERNAL_RATE * 1000:.0f} ms"
            f" | process p50/p95 {p50:.1f}/{p95:.1f} ms"
            f" | reserve {self.controller.reserve_ms} ms (ring {level:.0f} ms)"
            f" | window {self.algorithmic_delay_ms:.0f} ms | I/O {self.device_latency_ms:.1f} ms"
            f" | est. total {self.estimated_latency_ms():.0f} ms"
            f" | xruns in/out {self.input_overflows}/{self.output_underflows}"
            f" | quiet corrections {self.controller.adjustments}"
        )
        if self.drift_enabled:
            text += f" | drift {self.controller.drift_ppm:+.0f} ppm"
        runtime = getattr(getattr(self.callbacks, "vc", None), "vc_model", None)
        fcn = getattr(runtime, "fcn_session", None)
        if fcn is not None:
            text += f" | FCN holdback: {fcn.holdback_ms:.2f} ms"
        if self.open_warnings:
            text += " | " + "; ".join(self.open_warnings)
        return text
