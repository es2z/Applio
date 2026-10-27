"""Stage 5 opt-in options: a fixed conversion window and a fixed gain span, so that
lowering Chunk Size does not change what the model hears or how the level is set.

Uses the identity pipeline from test_realtime_options: the "conversion" returns the
window it is given, so layout and gain are all that is left to measure.
"""

import importlib.util
import os
import unittest

import numpy as np
import torch

from rvc.realtime.options import (
    FIELD_ORDER,
    RealtimeOptions,
    chunk_independent_options,
    context_window_extra_frames,
    snap_read_chunk_size,
)
from tests.test_realtime_options import make_changer, measure_delay_ms

REFERENCE = dict(crossfade=0.138, extra=0.22)  # the reference template


def preset(chunk_ms=1080, base=None, **layout):
    layout = {**REFERENCE, **layout}
    return RealtimeOptions.from_dict(
        chunk_independent_options(base or RealtimeOptions(), chunk_ms, layout["crossfade"], layout["extra"])
    )


class OptionTests(unittest.TestCase):
    def test_defaults_change_nothing(self):
        options = RealtimeOptions()
        self.assertEqual(options.context_window_s, 0.0)
        self.assertEqual(options.gain_span_s, 0.0)
        self.assertEqual(options.changed(), [])
        self.assertEqual(FIELD_ORDER[-2:], ("context_window_s", "gain_span_s"))
        changer = make_changer(options)
        self.assertIsNone(changer.context_window_extra_frame)
        self.assertIsNone(changer.vc_model.gain_span_16k)
        self.assertEqual(changer.layout_status(), "")

    def test_from_dict_clamps_and_round_trips(self):
        options = RealtimeOptions.from_dict({"context_window_s": 99, "gain_span_s": -1})
        self.assertEqual(options.context_window_s, 10.0)
        self.assertEqual(options.gain_span_s, 0.0)
        self.assertEqual(RealtimeOptions.from_dict(options.to_dict()), options)

    def test_the_preset_holds_what_the_current_layout_has(self):
        options = preset(1080)
        self.assertEqual(options.output_position, "trim")
        # Legacy's played block ends Extra + crossfade + SOLA before the newest input;
        # trim with look-ahead = Extra gives every played sample at least that.
        self.assertAlmostEqual(options.lookahead_s, 0.22)
        # 1080 + 138 + 10 + 220 ms, to the UI's 0.01 s.
        self.assertAlmostEqual(options.context_window_s, 1.45)
        self.assertAlmostEqual(options.gain_span_s, 1.22)
        self.assertTrue(options.time_locked_noise and options.phase_continuity)
        self.assertEqual(options.input_resample, "stateful")
        # Nothing that changes the voice itself.
        self.assertEqual(options.apex_noise_envelope, "window")
        self.assertFalse(options.f0_cache)
        self.assertEqual(options.gain_mode, "legacy")

    def test_the_preset_keeps_a_trimming_layout_and_set_values(self):
        base = RealtimeOptions(output_position="trim", lookahead_s=0.1, gain_mode="aligned")
        options = preset(480, base=base, extra=1.0)
        self.assertAlmostEqual(options.lookahead_s, 0.1)
        self.assertAlmostEqual(options.context_window_s, round(0.48 + 0.138 + 0.01 + 1.0 + 0.1, 2))
        self.assertEqual(options.gain_mode, "aligned")
        again = preset(240, base=options, extra=1.0)
        self.assertEqual(again.context_window_s, options.context_window_s)
        self.assertEqual(again.gain_span_s, options.gain_span_s)


class SnapTests(unittest.TestCase):
    def test_only_the_time_locked_options_move_the_chunk(self):
        self.assertEqual(snap_read_chunk_size(93), (93, False))
        self.assertEqual(snap_read_chunk_size(93, RealtimeOptions(output_position="trim")), (93, False))
        locked = RealtimeOptions(time_locked_noise=True)
        self.assertEqual(snap_read_chunk_size(93, locked), (90, True))  # 248 -> 240 ms
        self.assertEqual(snap_read_chunk_size(90, locked), (90, False))
        self.assertEqual(snap_read_chunk_size(1, locked), (15, True))  # never below 40 ms
        self.assertEqual(snap_read_chunk_size(98, RealtimeOptions(f0_cache=True)), (105, True))


class FixedWindowTests(unittest.TestCase):
    def test_the_window_and_the_delay_do_not_follow_the_chunk(self):
        options = preset(1080)
        sizes, delays, past = set(), set(), []
        for chunk_ms in (1080, 480, 240, 160):
            changer = make_changer(options, chunk_ms=chunk_ms, **REFERENCE)
            runtime = changer.vc_model
            sizes.add(runtime.convert_size_16k)
            delays.add(round(runtime.algorithmic_delay_ms, 3))
            past.append(runtime.skip_head)
            self.assertIn("conversion window 1450 ms", changer.layout_status())
        self.assertEqual(sizes, {23200})  # 1.45 s at 16 kHz
        # crossfade + SOLA + look-ahead, rounded up to 10 ms - the legacy 370 ms.
        self.assertEqual(delays, {370.0})
        # What the chunk gives up is past context.
        self.assertEqual(past, sorted(past))
        self.assertGreater(past[-1] - past[0], 80)

    def test_the_identity_output_is_the_input_delayed(self):
        changer = make_changer(preset(1080), chunk_ms=240, **REFERENCE)
        delay, corr = measure_delay_ms(changer)
        self.assertGreater(corr, 0.99)
        self.assertAlmostEqual(delay, 370.0 + 24 / 48.0, delta=1.0)  # + stateful resampler

    def test_legacy_ignores_it(self):
        options = RealtimeOptions(context_window_s=3.0)
        self.assertIsNone(context_window_extra_frames(options, 46080, 6624, 480))
        changer = make_changer(options)
        self.assertEqual(changer.extra_frame, int(0.22 * 48000))

    def test_a_window_too_short_leaves_no_past_context(self):
        options = RealtimeOptions(output_position="trim", context_window_s=0.2)
        self.assertEqual(context_window_extra_frames(options, 46080, 6624, 480), 0)


class GainSpanTests(unittest.TestCase):
    @staticmethod
    def gains(options, chunk_ms, seconds=20.0):
        """Per-block gain the Realtime class applied, read back through the identity."""
        changer = make_changer(options, chunk_ms=chunk_ms, **REFERENCE)
        runtime = changer.vc_model
        captured = {}
        convert = runtime.pipeline.voice_conversion

        def capture(*args):
            captured["model"] = out = convert(*args)
            return out.clone()

        runtime.pipeline.voice_conversion = capture
        block = changer.block_frame
        t = np.arange(int(seconds * 48000) // block * block) / 48000
        rng = np.random.default_rng(4)
        # Syllables: 4 Hz on/off bursts under a slower phrase envelope.
        envelope = (np.sin(2 * np.pi * 4 * t) > 0) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * t))
        x = (0.1 * envelope * rng.standard_normal(t.shape)).astype(np.float32)
        out = []
        for i in range(0, len(x), block):
            audio, _ = runtime.inference(x[i : i + block])
            model = captured["model"]
            out.append(float(torch.dot(audio, model) / torch.dot(model, model)))
        return np.array(out[4:])

    def test_a_span_of_block_plus_crossfade_is_the_legacy_gain(self):
        span = (240 + 138) / 1000.0
        legacy = self.gains(RealtimeOptions(), 240)
        fixed = self.gains(RealtimeOptions(gain_span_s=span), 240)
        np.testing.assert_array_equal(legacy, fixed)

    def test_a_fixed_span_stops_the_level_pumping_at_a_short_chunk(self):
        def wobble(gains):
            # Block-to-block change of the gain, in dB.
            return float(np.abs(np.diff(20 * np.log10(gains))).mean())

        long_chunk = wobble(self.gains(RealtimeOptions(), 1080))
        short_chunk = wobble(self.gains(RealtimeOptions(), 160))
        held = wobble(self.gains(RealtimeOptions(gain_span_s=1.22), 160))
        self.assertGreater(short_chunk, 1.5 * held)
        # Each block changes less than the 1080 ms chunk's did; there are just more of them.
        self.assertLess(held, long_chunk)

    def test_aligned_gain_uses_the_span(self):
        changer = make_changer(
            RealtimeOptions(output_position="trim", gain_mode="aligned", gain_span_s=0.05),
            chunk_ms=240, extra=1.0,
        )
        runtime = changer.vc_model
        runtime.convert_buffer.zero_()
        runtime.convert_buffer[-1600:] = 0.1  # only the newest 100 ms is loud
        short = runtime.aligned_gain(runtime.return_length * 480)
        runtime.gain_span_16k = 16000
        long = runtime.aligned_gain(runtime.return_length * 480)
        # A 50 ms span sees nothing of the loud part 200 ms back; a 1 s span does.
        self.assertEqual(short[0].item(), 0.0)
        self.assertGreater(long[0].item(), 0.0)


class StreamEvalPresetTests(unittest.TestCase):
    def load_tool(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "realtime_stream_eval.py")
        spec = importlib.util.spec_from_file_location("realtime_stream_eval", path)
        module = importlib.util.module_from_spec(spec)
        cwd = os.getcwd()
        try:
            spec.loader.exec_module(module)
        finally:
            os.chdir(cwd)
        return module

    def test_fixed_at_a_chunk_is_worked_out_from_the_template(self):
        tool = self.load_tool()
        name, body = tool.parse_preset("fixed@240")
        self.assertEqual(name, "fixed@240")
        settings = {"options": {}, "chunk_ms": 1080.0, "crossfade_s": 0.138, "extra_s": 0.22}
        options = tool.resolve_options(body, settings)
        self.assertEqual(options["chunk_ms"], 240.0)
        self.assertAlmostEqual(options["context_window_s"], 1.45)
        self.assertNotIn(tool.CHUNK_INDEPENDENT, options)
        name, body = tool.parse_preset("legacy@240")
        self.assertEqual(tool.resolve_options(body, settings), {"chunk_ms": 240.0})
        with self.assertRaises(SystemExit):
            tool.parse_preset("template@240")


if __name__ == "__main__":
    unittest.main()
