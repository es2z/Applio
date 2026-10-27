"""Opt-in realtime options for latency and quality experiments.

Every default reproduces the realtime path as it was before these options existed, so
nothing changes until a setting is switched on. Each option is independent and can be
A/B'd on its own. What each one does and why is in docs/realtime-latency-quality.md.

The options travel as one immutable object from the Realtime tab (and templates)
through AudioCallbacks and VoiceChanger down to the pipeline; it is fixed for the life
of one Start/Stop session because several of them change tensor shapes.
"""

from dataclasses import asdict, dataclass, fields

# Where the converted block is taken from inside the conversion window.
#   "legacy": the generator renders the whole window and the oldest part is played, so
#             Extra Conversion Size works as look-ahead and adds to the latency.
#   "trim":   the oldest Extra Conversion Size is past context only; the generator
#             renders the newest part (plus a flow margin) and that is played.
OUTPUT_POSITIONS = ("legacy", "trim")

# 48 kHz -> 16 kHz input conversion: per block ("stateless") or continuous ("stateful").
INPUT_RESAMPLES = ("stateless", "stateful")

# Which voicing decides `protect`: the F0 handed to the model ("f0", which for the
# gaps-filled methods is every frame voiced) or the predictor's voicing before the gap
# fill ("unfilled").
PROTECT_VOICINGS = ("f0", "unfilled")

# Output level: "legacy" multiplies the whole rendered window by sqrt(RMS) of the newest
# input block; "aligned" follows the input RMS of the same moments it is applied to.
GAIN_MODES = ("legacy", "aligned")

# APEX-GAN noise envelope normaliser: the mean over the rendered window ("window",
# what training and every earlier version do) or a fixed reference ("fixed").
APEX_NOISE_ENVELOPES = ("window", "fixed")

# Audio transport: callbacks of Chunk Size that run inference ("legacy") or small host
# callbacks around ring buffers with inference on its own thread ("ring").
TRANSPORTS = ("legacy", "ring")

# Ring transport stream layout: one duplex stream when both devices share a host API
# ("auto", what new_main_testversion did) or always two streams with clock drift
# compensation ("separate").
RING_STREAM_MODES = ("auto", "separate")

# Where the generator output is clipped to +-1: right after the generator, before the
# input-level gain ("generator", as always), or only the final output ("output"). The
# gain is sqrt(input RMS), well under 1, so with "output" a generator peak above 1 is
# scaled down instead of flattened first.
CLIP_STAGES = ("generator", "output")

# Synthesizer.infer's latent temperature, which "latent_noise_scale" defaults to.
DEFAULT_LATENT_NOISE_SCALE = 0.66666

# The 48 kHz device rate and the SOLA search every layout carries (core.py), needed to
# turn seconds into the window the "trim" layout builds.
_AUDIO_SAMPLE_RATE = 48000
SOLA_SEARCH_S = 0.01

# One read chunk is 128 samples at 48 kHz; the time-locked options need a block of whole
# 10 ms frames, i.e. a multiple of 480 samples, which is a multiple of 15 read chunks
# (40 ms).
_READ_CHUNK_SAMPLES = 128
_TIME_LOCKED_READ_CHUNK_STEP = 15


def _choice(value, allowed, default):
    return value if value in allowed else default


def _number(value, default, low=None, high=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if value != value:  # NaN
        return default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _latent_scale(value):
    """A slider with 0.01 steps shows 0.66666 as 0.67; that is the default, not a change."""
    value = _number(value, DEFAULT_LATENT_NOISE_SCALE, 0.0, 1.0)
    if abs(value - DEFAULT_LATENT_NOISE_SCALE) < 0.005:
        return DEFAULT_LATENT_NOISE_SCALE
    return value


@dataclass(frozen=True)
class RealtimeOptions:
    output_position: str = "legacy"
    lookahead_s: float = 0.0
    input_resample: str = "stateless"
    protect_voicing: str = "f0"
    gain_mode: str = "legacy"
    apex_noise_envelope: str = "window"
    apex_noise_voiced_ratio: float = 0.65
    transport: str = "legacy"
    ring_stream_mode: str = "auto"
    time_locked_noise: bool = False
    phase_continuity: bool = False
    f0_cache: bool = False
    f0_cache_margin_ms: float = 100.0
    latent_noise_scale: float = DEFAULT_LATENT_NOISE_SCALE
    input_highpass_hz: float = 0.0
    clip_stage: str = "generator"
    gate_fade_ms: float = 0.0
    gate_hold: bool = False
    # Stage 5: what makes the sound independent of Chunk Size. Both 0 = as always.
    #   context_window_s: "trim" only. The whole conversion window (past context +
    #       block + crossfade + SOLA + look-ahead) is held at this length and Extra
    #       Conversion Size is worked out from it, so lowering Chunk Size hands the time
    #       to past context instead of shrinking what the embedder and the F0 see.
    #   gain_span_s: the input-level gain's RMS is taken over this span instead of
    #       block + crossfade, so a shorter chunk does not make the level pump.
    context_window_s: float = 0.0
    gain_span_s: float = 0.0

    @classmethod
    def from_dict(cls, data):
        """Tolerant constructor: unknown keys are ignored, bad values fall back."""
        data = dict(data or {})
        default = cls()
        return cls(
            output_position=_choice(
                data.get("output_position"), OUTPUT_POSITIONS, default.output_position
            ),
            lookahead_s=_number(data.get("lookahead_s"), default.lookahead_s, 0.0, 1.0),
            input_resample=_choice(
                data.get("input_resample"), INPUT_RESAMPLES, default.input_resample
            ),
            protect_voicing=_choice(
                data.get("protect_voicing"), PROTECT_VOICINGS, default.protect_voicing
            ),
            gain_mode=_choice(data.get("gain_mode"), GAIN_MODES, default.gain_mode),
            apex_noise_envelope=_choice(
                data.get("apex_noise_envelope"),
                APEX_NOISE_ENVELOPES,
                default.apex_noise_envelope,
            ),
            apex_noise_voiced_ratio=_number(
                data.get("apex_noise_voiced_ratio"),
                default.apex_noise_voiced_ratio,
                0.0,
                1.0,
            ),
            transport=_choice(data.get("transport"), TRANSPORTS, default.transport),
            ring_stream_mode=_choice(
                data.get("ring_stream_mode"), RING_STREAM_MODES, default.ring_stream_mode
            ),
            time_locked_noise=bool(data.get("time_locked_noise", default.time_locked_noise)),
            phase_continuity=bool(data.get("phase_continuity", default.phase_continuity)),
            f0_cache=bool(data.get("f0_cache", default.f0_cache)),
            f0_cache_margin_ms=_number(
                data.get("f0_cache_margin_ms"), default.f0_cache_margin_ms, 0.0, 300.0
            ),
            latent_noise_scale=_latent_scale(data.get("latent_noise_scale")),
            input_highpass_hz=_number(
                data.get("input_highpass_hz"), default.input_highpass_hz, 0.0, 200.0
            ),
            clip_stage=_choice(data.get("clip_stage"), CLIP_STAGES, default.clip_stage),
            gate_fade_ms=_number(data.get("gate_fade_ms"), default.gate_fade_ms, 0.0, 200.0),
            gate_hold=bool(data.get("gate_hold", default.gate_hold)),
            context_window_s=_number(
                data.get("context_window_s"), default.context_window_s, 0.0, 10.0
            ),
            gain_span_s=_number(data.get("gain_span_s"), default.gain_span_s, 0.0, 5.0),
        )

    @classmethod
    def from_values(cls, *values):
        """Build from the UI controls, in the order of FIELD_ORDER."""
        return cls.from_dict(dict(zip(FIELD_ORDER, values)))

    def to_dict(self):
        return asdict(self)

    def values(self):
        """The fields in FIELD_ORDER, for Gradio updates."""
        data = self.to_dict()
        return [data[name] for name in FIELD_ORDER]

    @property
    def trims_output(self):
        return self.output_position == "trim"

    @property
    def needs_realtime_synthesis(self):
        """True when the generator call cannot be the plain Synthesizer.infer."""
        return (
            self.trims_output
            or self.time_locked_noise
            or self.phase_continuity
            or self.apex_noise_envelope != "window"
            or self.latent_noise_scale != DEFAULT_LATENT_NOISE_SCALE
        )

    @property
    def needs_whole_frames(self):
        """True when a block has to be a whole number of 10 ms frames (40 ms steps)."""
        return self.time_locked_noise or self.phase_continuity or self.f0_cache

    def changed(self):
        """Names of the options that differ from the defaults, for the status line."""
        default = RealtimeOptions()
        return [
            name
            for name in FIELD_ORDER
            if getattr(self, name) != getattr(default, name)
        ]


FIELD_ORDER = tuple(field.name for field in fields(RealtimeOptions))


def snap_read_chunk_size(read_chunk_size, options=None):
    """(read chunks, whether it moved).

    With the time-locked options on, Chunk Size is moved to the nearest multiple of 40 ms
    (at least 40 ms) instead of those options being switched off for the session, so a
    Chunk Size slider moved in 1 ms steps does not silently lose them. Anything else is
    returned as it is.
    """
    read_chunk_size = int(read_chunk_size)
    if options is None or not options.needs_whole_frames:
        return read_chunk_size, False
    step = _TIME_LOCKED_READ_CHUNK_STEP
    snapped = max(step, int(round(read_chunk_size / step)) * step)
    return snapped, snapped != read_chunk_size


def block_seconds(read_chunk_size):
    return int(read_chunk_size) * _READ_CHUNK_SAMPLES / _AUDIO_SAMPLE_RATE


def context_window_extra_frames(
    options, block_frame, crossfade_frame, sola_search_frame, sample_rate=_AUDIO_SAMPLE_RATE
):
    """Extra Conversion Size, in device samples, that makes the conversion window
    `context_window_s` long; None when the option is off or the layout is not "trim".

    The window is past context + block + crossfade + SOLA + look-ahead, so whatever the
    block gives up goes to past context. A window shorter than the rest leaves no past
    context (0), never a negative one.
    """
    if options is None or not options.trims_output or options.context_window_s <= 0:
        return None
    window = int(round(options.context_window_s * sample_rate))
    lookahead = int(round(options.lookahead_s * sample_rate))
    return max(0, window - block_frame - crossfade_frame - sola_search_frame - lookahead)


def chunk_independent_options(options, chunk_ms, crossfade_s, extra_s):
    """Options that keep what the current layout sounds like when Chunk Size is lowered
    afterwards (the Realtime tab's "keep this sound at a shorter chunk" button).

    Everything is measured from the layout as it is now:
    - Look-ahead: legacy plays the oldest block of the window, so its end still has
      Extra + crossfade + SOLA of future audio; "trim" with look-ahead = Extra gives
      every played sample at least that much, whatever the chunk. A layout that already
      trims keeps its look-ahead.
    - Conversion window: chunk + crossfade + SOLA + Extra (+ look-ahead when trimming),
      the length the embedder, the F0 predictor and the text encoder see now. Held, so a
      shorter chunk turns into past context.
    - Gain span: block + crossfade, what the gain is averaged over now.
    - Time-locked noise and sine phase: a shorter chunk means more joins; with these the
      two renders either side of a join agree, as they would inside one long block.
    - Stateful input resampling: no small discontinuity at each of those joins.
    Values already set (window, span) are kept. Everything else is left as it is.
    Returned as a dict for RealtimeOptions.from_dict, rounded to the UI's 0.01 s steps.
    """
    options = options or RealtimeOptions()
    data = options.to_dict()
    block = block_seconds(int(float(chunk_ms) * _AUDIO_SAMPLE_RATE / 1000 / _READ_CHUNK_SAMPLES))
    crossfade = float(crossfade_s)
    extra = float(extra_s)
    if options.trims_output:
        lookahead = options.lookahead_s
        window = block + crossfade + SOLA_SEARCH_S + extra + lookahead
    else:
        lookahead = extra
        window = block + crossfade + SOLA_SEARCH_S + extra
    if options.trims_output and options.context_window_s > 0:
        window = options.context_window_s
    span = options.gain_span_s if options.gain_span_s > 0 else block + crossfade
    data.update(
        output_position="trim",
        lookahead_s=round(min(1.0, lookahead), 2),
        context_window_s=round(window, 2),
        gain_span_s=round(span, 2),
        time_locked_noise=True,
        phase_continuity=True,
        input_resample="stateful",
    )
    return data
