"""Primitives for the ring transport (rvc/realtime/ring_transport.py).

Ported from the discarded new_main_testversion branch (rvc/realtime/runtime.py,
29eae3d), where the same transport made WASAPI -> WASAPI as low latency as WASAPI ->
WDM-KS. The ring buffer is now lock free and the latency control is rewritten around
the level the output ring actually holds (see ReserveController).
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

BUFFER_STEP_MS = 10
BASE_RESERVE_MS = 30
MAX_EXTRA_RESERVE_MS = 500
UNDERFLOW_COOLDOWN_S = 2.0
STABLE_BEFORE_SHRINK_S = 30.0
QUIET_BEFORE_SHRINK_S = 1.0
ADJUST_THRESHOLD_MS = 10.0
ERROR_WINDOW = 9
# How long the slowest block seen is remembered. A slow block that is forgotten
# before the next one arrives costs an underflow, an audible click, every time.
NEED_WINDOW_S = 120.0
MAX_DRIFT_PPM = 1000.0
# PI control of the output resampling ratio from the reserve error. The ring level
# integrates the ratio error (1 ppm moves it 1e-3 ms per second), so with these gains
# the loop is critically damped at 0.03 rad/s: a ~30 s settling, slow enough that
# measurement noise does not reach the pitch (10 ms of error is 600 ppm, 1 cent).
DRIFT_KP_PPM_PER_MS = 60.0
DRIFT_KI_PPM_PER_MS_S = 0.9


class FloatRingBuffer:
    """Bounded single-channel float32 ring buffer for one producer and one consumer.

    Lock free: the producer only ever advances `_written` and the consumer only
    `_consumed`, each after its copy is done, so neither side waits for the other.
    The lock this replaced made a PortAudio callback that met the worker inside it
    give up and play (or drop) a whole device period of silence: a click.

    `write` belongs to the producer; `read_into`, `read`, `discard` and `clear` to the
    consumer. `blocking` is accepted for the old signature and has no effect.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Ring capacity must be positive")
        self.capacity = int(capacity)
        self._data = np.zeros(self.capacity, dtype=np.float32)
        self._written = 0  # total samples ever written, producer only
        self._consumed = 0  # total samples ever consumed, consumer only
        self.dropped_writes = 0
        self.underrun_reads = 0

    @property
    def available(self) -> int:
        consumed = self._consumed
        return self._written - consumed

    available_approx = available

    def clear(self) -> None:
        self._consumed = self._written

    def write(self, samples: np.ndarray, *, blocking: bool = True) -> int:
        source = np.asarray(samples, dtype=np.float32).reshape(-1)
        written = self._written
        count = min(len(source), self.capacity - (written - self._consumed))
        if count <= 0:
            self.dropped_writes += len(source)
            return 0
        start = written % self.capacity
        first = min(count, self.capacity - start)
        self._data[start : start + first] = source[:first]
        second = count - first
        if second:
            self._data[:second] = source[first : first + second]
        self._written = written + count
        if count < len(source):
            self.dropped_writes += len(source) - count
        return count

    def read_into(self, destination: np.ndarray, *, blocking: bool = True) -> int:
        target = np.asarray(destination).reshape(-1)
        consumed = self._consumed
        count = min(len(target), self._written - consumed)
        start = consumed % self.capacity
        first = min(count, self.capacity - start)
        target[:first] = self._data[start : start + first]
        second = count - first
        if second:
            target[first:count] = self._data[:second]
        self._consumed = consumed + count
        if count < len(target):
            self.underrun_reads += len(target) - count
        return count

    def read(self, count: int) -> np.ndarray:
        result = np.zeros(int(count), dtype=np.float32)
        read = self.read_into(result)
        return result[:read]

    def discard(self, count: int) -> int:
        """Drop up to `count` of the oldest samples."""
        consumed = self._consumed
        count = max(0, min(int(count), self._written - consumed))
        self._consumed = consumed + count
        return count


def ceil_to_step(value_ms: float, step_ms: int = BUFFER_STEP_MS) -> int:
    return int(math.ceil(max(0.0, value_ms) / step_ms) * step_ms)


class ReserveController:
    """Keeps the output ring deep enough that no block arrives after it ran dry.

    The worker hands in, for every converted block, how long the conversion took
    (`elapsed_ms`) and how much the output ring still held when it finished
    (`level_ms`). A block that took p ms leaves level = constant - p, so the quantity
    held steady is x = level + p. Its recent median is the operating point; a block
    whose x came out below it arrived late for some other reason (the input device,
    the worker waking up), and that lateness adds to what the block needed.

    - What a block needed is its processing time plus that lateness. The target is
      the most any block needed in the last 120 s plus a reserve, so a slow block
      that came once is planned for until it has not come for two minutes. (Aiming
      at the p95 instead, as before, left one block in twenty with only the reserve
      between it and an underflow, and every underflow is a click.)
    - The reserve is `margin_ms` (30 ms), plus 10 ms for every underflow (at most one
      step per 2 s), back down by 10 ms after 30 s without one and 1 s of quiet output.
    - The error is corrected by trimming or padding quiet output (never voice), and,
      with two device clocks, also by a resampling ratio (`drift_ppm`) bounded to
      +-1000 ppm (1.7 cents), PI-controlled so a steady drift leaves no steady error.
    """

    def __init__(self, initial_timings_ms=(), clock=time.monotonic,
                 margin_ms: float = BASE_RESERVE_MS):
        self.clock = clock
        self.margin_ms = float(margin_ms)
        now = clock()
        self.needs = deque(((now, float(t)) for t in initial_timings_ms), maxlen=4096)
        self.timings = deque((float(t) for t in initial_timings_ms), maxlen=256)
        self.levels = deque(maxlen=ERROR_WINDOW)
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
    def reserve_ms(self) -> float:
        return self.margin_ms + self.extra_ms

    @property
    def p95_ms(self) -> float:
        if not self.timings:
            return 0.0
        return float(np.percentile(np.asarray(self.timings), 95))

    @property
    def max_ms(self) -> float:
        return max(self.timings) if self.timings else 0.0

    @property
    def need_ms(self) -> float:
        """The most any block needed within the window."""
        return max((need for _, need in self.needs), default=0.0)

    @property
    def target_ms(self) -> float:
        return self.need_ms + self.reserve_ms

    def _remember(self, now, need):
        self.needs.append((now, float(need)))
        while self.needs and now - self.needs[0][0] > NEED_WINDOW_S:
            self.needs.popleft()

    def startup_padding_ms(self, first_elapsed_ms: float) -> float:
        """Silence to play before the first block, so later blocks that take up to the
        slowest one seen still arrive in time."""
        self.timings.append(float(first_elapsed_ms))
        self._remember(self.clock(), first_elapsed_ms)
        return max(0.0, self.need_ms - first_elapsed_ms) + self.reserve_ms

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
        x = level_ms + elapsed_ms
        self.levels.append(x)
        operating = float(np.median(np.asarray(self.levels)))
        self._remember(now, elapsed_ms + max(0.0, operating - x))
        median = operating - self.target_ms
        self.errors.append(median)

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
        self.levels.clear()
        return -median
