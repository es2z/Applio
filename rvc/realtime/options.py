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


@dataclass(frozen=True)
class RealtimeOptions:
    output_position: str = "legacy"
    lookahead_s: float = 0.0
    input_resample: str = "stateless"
    protect_voicing: str = "f0"
    gain_mode: str = "legacy"
    apex_noise_envelope: str = "window"
    apex_noise_voiced_ratio: float = 0.65

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
        return self.trims_output or self.apex_noise_envelope != "window"

    def changed(self):
        """Names of the options that differ from the defaults, for the status line."""
        default = RealtimeOptions()
        return [
            name
            for name in FIELD_ORDER
            if getattr(self, name) != getattr(default, name)
        ]


FIELD_ORDER = tuple(field.name for field in fields(RealtimeOptions))
