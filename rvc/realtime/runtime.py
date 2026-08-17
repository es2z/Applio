"""Runtime-only configuration and buffering primitives for realtime audio.

The GUI chunk size is deliberately kept separate from the PortAudio callback size.
All shape-affecting values in :class:`RuntimeAudioShape` are immutable for the
lifetime of one Start/Stop session.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np


INTERNAL_SAMPLE_RATE = 48_000
MODEL_FRAME_SIZE = 128
ADAPT_GRID_MS = 10
TARGET_INFER_UTILIZATION = 0.70

BUFFER_STEP_MS = 10
MIN_OUTPUT_RESERVE_MS = 10
MISSES_BEFORE_BUFFER_STEP = 2
MISS_WINDOW_MS = 5_000
SILENCE_BEFORE_SHRINK_MS = 1_000
STABLE_BEFORE_SHRINK_MS = 5_000
BUFFER_ADJUST_COOLDOWN_MS = 2_000
DEFAULT_MAX_EXTRA_BUFFER_MS = 200
OVERLOAD_DETECTION_MS = 30_000
SILENCE_FLUSH_MS = 1_000

ProcessingMode = Literal["fixed_chunk", "overlap"]
HopMode = Literal["auto", "manual"]


def ceil_to_grid(value_ms: float, grid_ms: int = ADAPT_GRID_MS) -> int:
    return max(grid_ms, int(math.ceil(value_ms / grid_ms) * grid_ms))


@dataclass(frozen=True)
class RuntimeAudioShape:
    requested_chunk_ms: float
    context_frames: int
    effective_chunk_ms: float
    hop_frames: int
    effective_hop_ms: float
    processing_mode: ProcessingMode
    hop_mode: HopMode

    @classmethod
    def create(
        cls,
        requested_chunk_ms: float,
        processing_mode: ProcessingMode = "fixed_chunk",
        hop_mode: HopMode = "auto",
        manual_hop_ms: float = 320.0,
        measured_p95_ms: float | None = None,
    ) -> "RuntimeAudioShape":
        chunk_units = max(
            1,
            round(
                float(requested_chunk_ms)
                * INTERNAL_SAMPLE_RATE
                / 1000
                / MODEL_FRAME_SIZE
            ),
        )
        context_frames = chunk_units * MODEL_FRAME_SIZE
        effective_chunk_ms = context_frames / INTERNAL_SAMPLE_RATE * 1000

        if processing_mode not in ("fixed_chunk", "overlap"):
            raise ValueError(f"Unknown processing mode: {processing_mode}")
        if hop_mode not in ("auto", "manual"):
            raise ValueError(f"Unknown hop mode: {hop_mode}")

        if processing_mode == "fixed_chunk":
            hop_frames = context_frames
            hop_mode = "auto"
        else:
            if hop_mode == "auto":
                # Before a measured warmup exists, a conservative third of the
                # context gives the caller a stable shape for the first warmup.
                candidate_ms = (
                    float(measured_p95_ms) / TARGET_INFER_UTILIZATION
                    if measured_p95_ms and measured_p95_ms > 0
                    else max(ADAPT_GRID_MS, effective_chunk_ms / 3)
                )
            else:
                candidate_ms = float(manual_hop_ms)

            if candidate_ms <= 0:
                raise ValueError("Overlap hop must be greater than zero")
            if candidate_ms > effective_chunk_ms + 1e-6:
                if hop_mode == "manual":
                    raise ValueError(
                        "Manual overlap hop cannot be longer than Chunk Size"
                    )
                candidate_ms = effective_chunk_ms

            if effective_chunk_ms >= ADAPT_GRID_MS:
                candidate_ms = min(
                    effective_chunk_ms, ceil_to_grid(candidate_ms, ADAPT_GRID_MS)
                )
            hop_frames = max(
                1,
                round(candidate_ms * INTERNAL_SAMPLE_RATE / 1000),
            )
            hop_frames = min(hop_frames, context_frames)

        return cls(
            requested_chunk_ms=float(requested_chunk_ms),
            context_frames=context_frames,
            effective_chunk_ms=effective_chunk_ms,
            hop_frames=hop_frames,
            effective_hop_ms=hop_frames / INTERNAL_SAMPLE_RATE * 1000,
            processing_mode=processing_mode,
            hop_mode=hop_mode,
        )


class FloatRingBuffer:
    """Bounded single-channel float32 ring buffer.

    PortAudio callbacks use non-blocking lock acquisition. A contended callback
    records a dropped operation instead of waiting on the model worker.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Ring capacity must be positive")
        self.capacity = int(capacity)
        self._data = np.zeros(self.capacity, dtype=np.float32)
        self._read = 0
        self._write = 0
        self._size = 0
        self._lock = threading.Lock()
        self.dropped_writes = 0
        self.overwritten_samples = 0
        self.underrun_reads = 0

    @property
    def available(self) -> int:
        with self._lock:
            return self._size

    @property
    def available_approx(self) -> int:
        """Return a lock-free snapshot suitable for a PortAudio callback.

        The value is only used as a readiness hint. The subsequent non-blocking
        read remains authoritative, so a slightly stale value is harmless.
        """
        return self._size

    @property
    def free(self) -> int:
        with self._lock:
            return self.capacity - self._size

    def clear(self) -> None:
        with self._lock:
            self._read = self._write = self._size = 0

    def _discard_locked(self, count: int) -> int:
        discarded = min(max(0, int(count)), self._size)
        self._read = (self._read + discarded) % self.capacity
        self._size -= discarded
        return discarded

    def discard(self, count: int) -> int:
        """Discard up to ``count`` oldest samples and return the actual count."""
        with self._lock:
            return self._discard_locked(count)

    def trim_to_latest(self, count: int) -> int:
        """Keep only the newest ``count`` samples and return discarded samples."""
        with self._lock:
            return self._discard_locked(self._size - max(0, int(count)))

    def write(
        self,
        samples: np.ndarray,
        *,
        blocking: bool = True,
        overflow_policy: Literal["drop_newest", "drop_oldest"] = "drop_newest",
    ) -> int:
        source = np.asarray(samples, dtype=np.float32).reshape(-1)
        if overflow_policy not in ("drop_newest", "drop_oldest"):
            raise ValueError(f"Unknown overflow policy: {overflow_policy}")
        acquired = self._lock.acquire(blocking=blocking)
        if not acquired:
            self.dropped_writes += len(source)
            return 0
        try:
            if overflow_policy == "drop_oldest" and len(source) > self.capacity:
                # The newest part of an oversized callback is the only part that
                # can still satisfy realtime semantics.
                skipped = len(source) - self.capacity
                source = source[skipped:]
                self.dropped_writes += skipped
            if overflow_policy == "drop_oldest":
                overwritten = self._discard_locked(
                    max(0, self._size + len(source) - self.capacity)
                )
                self.overwritten_samples += overwritten

            count = min(len(source), self.capacity - self._size)
            if count <= 0:
                self.dropped_writes += len(source)
                return 0
            first = min(count, self.capacity - self._write)
            self._data[self._write : self._write + first] = source[:first]
            second = count - first
            if second:
                self._data[:second] = source[first : first + second]
            self._write = (self._write + count) % self.capacity
            self._size += count
            if count < len(source):
                self.dropped_writes += len(source) - count
            return count
        finally:
            self._lock.release()

    def replace(self, samples: np.ndarray) -> int:
        """Atomically replace queued audio with the newest supplied samples."""
        source = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(source) > self.capacity:
            skipped = len(source) - self.capacity
            source = source[skipped:]
            self.dropped_writes += skipped
        with self._lock:
            self.overwritten_samples += self._size
            self._read = self._write = self._size = 0
            count = len(source)
            if count:
                self._data[:count] = source
                self._write = count % self.capacity
                self._size = count
            return count

    def read_into(self, destination: np.ndarray, *, blocking: bool = True) -> int:
        target = np.asarray(destination).reshape(-1)
        acquired = self._lock.acquire(blocking=blocking)
        if not acquired:
            self.underrun_reads += len(target)
            return 0
        try:
            count = min(len(target), self._size)
            first = min(count, self.capacity - self._read)
            target[:first] = self._data[self._read : self._read + first]
            second = count - first
            if second:
                target[first:count] = self._data[:second]
            self._read = (self._read + count) % self.capacity
            self._size -= count
            if count < len(target):
                self.underrun_reads += len(target) - count
            return count
        finally:
            self._lock.release()

    def read(self, count: int) -> np.ndarray:
        result = np.zeros(int(count), dtype=np.float32)
        read = self.read_into(result)
        return result[:read]


class ElasticBufferController:
    """Additive, never-exponential elastic latency controller."""

    def __init__(
        self,
        base_reserve_ms: int = MIN_OUTPUT_RESERVE_MS,
        max_extra_buffer_ms: int = DEFAULT_MAX_EXTRA_BUFFER_MS,
    ):
        self.base_reserve_ms = max(MIN_OUTPUT_RESERVE_MS, int(base_reserve_ms))
        self.max_extra_buffer_ms = max(0, int(max_extra_buffer_ms))
        self.extra_buffer_ms = 0
        self._misses: deque[float] = deque()
        self._inference: deque[tuple[float, float]] = deque()
        self._last_miss_at = 0.0
        self._last_adjust_at = 0.0
        self._silent_since: float | None = None
        self._lock = threading.Lock()
        self.overloaded = False
        self.backlog_ms = 0.0

    @property
    def total_reserve_ms(self) -> int:
        with self._lock:
            return self.base_reserve_ms + self.extra_buffer_ms

    def record_inference(self, duration_ms: float, cadence_ms: float) -> int:
        """Record one run and return frames to insert (+) or remove (-)."""
        now = time.monotonic()
        with self._lock:
            self._inference.append((now, float(duration_ms)))
            overload_cutoff = now - OVERLOAD_DETECTION_MS / 1000
            while self._inference and self._inference[0][0] < overload_cutoff:
                self._inference.popleft()
            self.backlog_ms = max(
                0.0, self.backlog_ms + float(duration_ms) - float(cadence_ms)
            )
            if self._inference and now - self._inference[0][0] >= (
                OVERLOAD_DETECTION_MS / 1000 * 0.9
            ):
                average = sum(value for _, value in self._inference) / len(
                    self._inference
                )
                self.overloaded = average >= cadence_ms and self.backlog_ms > 0
            if duration_ms > cadence_ms:
                return self._record_miss_locked(now)
            return 0

    def record_underflow(self) -> int:
        now = time.monotonic()
        with self._lock:
            # A real underflow is already audible, so it earns one immediate,
            # but still additive, 10 ms safety step.
            self._last_miss_at = now
            return self._grow_locked(now)

    def _record_miss_locked(self, now: float) -> int:
        self._last_miss_at = now
        cutoff = now - MISS_WINDOW_MS / 1000
        self._misses.append(now)
        while self._misses and self._misses[0] < cutoff:
            self._misses.popleft()
        if len(self._misses) < MISSES_BEFORE_BUFFER_STEP:
            return 0
        self._misses.clear()
        return self._grow_locked(now)

    def _grow_locked(self, now: float) -> int:
        if now - self._last_adjust_at < BUFFER_ADJUST_COOLDOWN_MS / 1000:
            return 0
        old = self.extra_buffer_ms
        self.extra_buffer_ms = min(
            self.max_extra_buffer_ms, self.extra_buffer_ms + BUFFER_STEP_MS
        )
        if self.extra_buffer_ms == old:
            return 0
        self._last_adjust_at = now
        return int(BUFFER_STEP_MS * INTERNAL_SAMPLE_RATE / 1000)

    def observe_silence(self, silent: bool) -> int:
        """Return the number of silent frames that may be removed."""
        now = time.monotonic()
        with self._lock:
            if not silent:
                self._silent_since = None
                return 0
            if self._silent_since is None:
                self._silent_since = now
                return 0
            if self.extra_buffer_ms <= 0:
                return 0
            if now - self._silent_since < SILENCE_BEFORE_SHRINK_MS / 1000:
                return 0
            if now - self._last_miss_at < STABLE_BEFORE_SHRINK_MS / 1000:
                return 0
            if now - self._last_adjust_at < BUFFER_ADJUST_COOLDOWN_MS / 1000:
                return 0
            self.extra_buffer_ms = max(0, self.extra_buffer_ms - BUFFER_STEP_MS)
            self._last_adjust_at = now
            return -int(BUFFER_STEP_MS * INTERNAL_SAMPLE_RATE / 1000)
