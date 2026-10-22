"""Primitives for the ring transport (rvc/realtime/ring_transport.py).

Ported from the discarded new_main_testversion branch (rvc/realtime/runtime.py,
29eae3d), where the same transport made WASAPI -> WASAPI as low latency as WASAPI ->
WDM-KS. Only the ring buffer survives unchanged; the latency control is rewritten
around the level the output ring actually holds (see ReserveController).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np

BUFFER_STEP_MS = 10
BASE_RESERVE_MS = 10
MAX_EXTRA_RESERVE_MS = 500
UNDERFLOW_COOLDOWN_S = 2.0
STABLE_BEFORE_SHRINK_S = 5.0
QUIET_BEFORE_SHRINK_S = 1.0
ADJUST_THRESHOLD_MS = 10.0
ERROR_WINDOW = 9
MAX_DRIFT_PPM = 1000.0
# PI control of the output resampling ratio from the reserve error. The ring level
# integrates the ratio error (1 ppm moves it 1e-3 ms per second), so with these gains
# the loop is critically damped at 0.03 rad/s: a ~30 s settling, slow enough that
# measurement noise does not reach the pitch (10 ms of error is 600 ppm, 1 cent).
DRIFT_KP_PPM_PER_MS = 60.0
DRIFT_KI_PPM_PER_MS_S = 0.9


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
        self.underrun_reads = 0

    @property
    def available(self) -> int:
        with self._lock:
            return self._size

    @property
    def available_approx(self) -> int:
        """Lock-free snapshot for a PortAudio callback. Only a readiness hint: the
        non-blocking read that follows is authoritative."""
        return self._size

    def clear(self) -> None:
        with self._lock:
            self._read = self._write = self._size = 0

    def write(self, samples: np.ndarray, *, blocking: bool = True) -> int:
        source = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not self._lock.acquire(blocking=blocking):
            self.dropped_writes += len(source)
            return 0
        try:
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

    def read_into(self, destination: np.ndarray, *, blocking: bool = True) -> int:
        target = np.asarray(destination).reshape(-1)
        if not self._lock.acquire(blocking=blocking):
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

    def discard(self, count: int) -> int:
        """Drop up to `count` of the oldest samples."""
        with self._lock:
            count = min(int(count), self._size)
            self._read = (self._read + count) % self.capacity
            self._size -= count
            return count


def ceil_to_step(value_ms: float, step_ms: int = BUFFER_STEP_MS) -> int:
    return int(math.ceil(max(0.0, value_ms) / step_ms) * step_ms)


class ReserveController:
    """Keeps the output ring just deep enough for the processing time to vary.

    The worker hands in, for every converted block, how long the conversion took
    (`elapsed_ms`) and how much the output ring still held when it finished
    (`level_ms`). A block that took p ms leaves level = constant - p, so the quantity
    held steady is `level + p`: the controller aims it at the recent p95 of the
    processing time plus a reserve, and reports the error.

    - The reserve is 10 ms, plus 10 ms for every underflow (at most one step per 2 s),
      back down by 10 ms after 5 s without one and 1 s of quiet output.
    - The error is corrected by trimming or padding quiet output (never voice), and,
      with two device clocks, also by a resampling ratio (`drift_ppm`) bounded to
      +-1000 ppm (1.7 cents), PI-controlled so a steady drift leaves no steady error.
    """

    def __init__(self, initial_timings_ms=(), clock=time.monotonic):
        self.clock = clock
        self.timings = deque((float(t) for t in initial_timings_ms), maxlen=256)
        self.extra_ms = 0
        self.errors = deque(maxlen=ERROR_WINDOW)
        self.drift_ppm = 0.0
        self._drift_integral = 0.0  # ms * s
        self._last_underflow = -math.inf
        self._last_change = -math.inf
        self._quiet_since = None
        self.underflows = 0
        self.adjustments = 0

    @property
    def reserve_ms(self) -> int:
        return BASE_RESERVE_MS + self.extra_ms

    @property
    def p95_ms(self) -> float:
        if not self.timings:
            return 0.0
        return float(np.percentile(np.asarray(self.timings), 95))

    def startup_padding_ms(self, first_elapsed_ms: float) -> float:
        """Silence to play before the first block, so later blocks that take up to the
        p95 still arrive in time."""
        self.timings.append(float(first_elapsed_ms))
        return max(0.0, self.p95_ms - first_elapsed_ms) + self.reserve_ms

    def record_underflow(self) -> None:
        now = self.clock()
        self.underflows += 1
        if now - self._last_change >= UNDERFLOW_COOLDOWN_S:
            self.extra_ms = min(MAX_EXTRA_RESERVE_MS, self.extra_ms + BUFFER_STEP_MS)
            self._last_change = now
        self._last_underflow = now

    def update(self, elapsed_ms: float, level_ms: float, quiet: bool, drift: bool,
               block_s: float = 1.0):
        """Record one block. Returns the correction in ms to apply to this block's
        output if it is quiet: positive pads, negative trims, 0 leaves it alone."""
        now = self.clock()
        self.timings.append(float(elapsed_ms))
        error = level_ms + elapsed_ms - self.p95_ms - self.reserve_ms
        self.errors.append(error)
        median = float(np.median(np.asarray(self.errors)))

        if drift:
            limit = MAX_DRIFT_PPM / DRIFT_KI_PPM_PER_MS_S
            self._drift_integral = float(
                np.clip(self._drift_integral + median * block_s, -limit, limit)
            )
            self.drift_ppm = float(np.clip(
                -(DRIFT_KP_PPM_PER_MS * median + DRIFT_KI_PPM_PER_MS_S * self._drift_integral),
                -MAX_DRIFT_PPM,
                MAX_DRIFT_PPM,
            ))

        if not quiet:
            self._quiet_since = None
            return 0.0
        if self._quiet_since is None:
            self._quiet_since = now
        if (
            self.extra_ms > 0
            and now - self._quiet_since >= QUIET_BEFORE_SHRINK_S
            and now - self._last_underflow >= STABLE_BEFORE_SHRINK_S
            and now - self._last_change >= UNDERFLOW_COOLDOWN_S
        ):
            self.extra_ms -= BUFFER_STEP_MS
            self._last_change = now
            median += BUFFER_STEP_MS  # the target just moved down by one step
        if abs(median) <= ADJUST_THRESHOLD_MS:
            return 0.0
        self.adjustments += 1
        self.errors.clear()
        return -median
