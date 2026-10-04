"""RingFormer+: the CodenameRingFormer generator against codename-ringformer's discriminator
plus the multi-scale sub-band CQT discriminator (MS-SB-CQT) the RingFormer paper uses.

Run from the repository root:
    env\\python.exe -m unittest tests.test_ringformer_plus -v

Real configs and real models, no mocks, in the style of tests/test_sifigan_plus.py. Kept
in its own file so that removing either RingFormer recipe later only touches what it has
to.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rvc.lib.algorithm.cqt_discriminator import (
    CQT_FILTERS,
    CQT_FMIN,
    ConstantQTransform,
    DiscriminatorCQT,
)
from rvc.lib.algorithm.discriminators import (
    DISCRIMINATOR_VERSIONS,
    MultiPeriodDiscriminator,
    describe_discriminator,
    discriminator_layout,
)
from rvc.lib.algorithm.generators.codename_ringformer import CodenameRingFormerGenerator
from rvc.lib.tools.pretrained_selector import pretrained_selector
from rvc.train.losses import discriminator_loss, feature_loss, generator_loss
from rvc.train.process.extract_model import extract_model
from rvc.train.reset_run import reset_training_run
from rvc.train.utils import (
    HParams,
    assert_resumable,
    describe_architecture_mismatch,
    save_checkpoint,
)
from rvc.train.vocoder_recipe import (
    RINGFORMER_PLUS,
    discriminator_version,
    generator_vocoder,
    pretrained_dir_name,
    sample_rate_error,
    uses_multiscale_mel_loss,
)
from rvc.train.warm_start import (
    CODENAME_RINGFORMER,
    HIFIGAN,
    detect_vocoder,
    normalize_weight_norm_keys,
    warm_start,
)

from tests.test_codename_ringformer import build_synthesizer, stock
from tests.test_sifigan import fake_f0, quiet
from tests.test_warm_start import save_legacy, snapshot

PLUS_VERSION = "codename-ringformer-cqt"
BASE_VERSION = "codename-ringformer"
SAMPLE_RATES = (32000, 40000, 48000)
CQT_SCALES = ((512, 9, 24), (256, 9, 36), (256, 9, 48))
# Sub-discriminator indices: scale, eight periods, three resolutions, three CQTs.
SHARED = range(12)
CQT_INDICES = (12, 13, 14)


def segment(sample_rate):
    return stock(sample_rate)["train"]["segment_size"]


def fresh_d(version, sample_rate=48000):
    return MultiPeriodDiscriminator(False, version=version, sample_rate=sample_rate)


def generator(vocoder, sample_rate=48000, speakers=1):
    return quiet(build_synthesizer, vocoder, sample_rate, 768, speakers)


def arch(vocoder, disc_version, sample_rate=48000):
    return {"vocoder": vocoder, "sample_rate": sample_rate, "disc_version": disc_version}


def sub_keys(state, index):
    return [key for key in state if key.startswith(f"discriminators.{index}.")]


class RecipeTest(unittest.TestCase):
    """What the recipe means, and that it only ever changes the discriminator."""

    def test_ringformer_plus_is_the_codename_ringformer_generator_against_the_cqt_layout(self):
        self.assertEqual(RINGFORMER_PLUS, "RingFormer+")
        self.assertEqual(generator_vocoder(RINGFORMER_PLUS), CODENAME_RINGFORMER)
        self.assertEqual(discriminator_version(RINGFORMER_PLUS), PLUS_VERSION)
        # CodenameRingFormer's losses: the single-scale mel loss, like its own recipe.
        self.assertFalse(uses_multiscale_mel_loss(RINGFORMER_PLUS))
        self.assertEqual(
            uses_multiscale_mel_loss(RINGFORMER_PLUS),
            uses_multiscale_mel_loss(CODENAME_RINGFORMER),
        )

    def test_every_sample_rate_is_accepted(self):
        for sample_rate in SAMPLE_RATES:
            self.assertIsNone(sample_rate_error(RINGFORMER_PLUS, sample_rate))

    def test_the_recipe_name_must_never_reach_synthesizer(self):
        """Synthesizer builds a HiFi-GAN decoder for any name it does not know."""
        unnormalised = generator(RINGFORMER_PLUS)
        self.assertEqual(detect_vocoder(unnormalised.state_dict()), HIFIGAN)
        normalised = generator(generator_vocoder(RINGFORMER_PLUS))
        self.assertEqual(detect_vocoder(normalised.state_dict()), CODENAME_RINGFORMER)
        self.assertIsInstance(normalised.dec, CodenameRingFormerGenerator)

    def test_the_pretrained_folder_has_no_plus_and_falls_back_to_hifigan(self):
        self.assertEqual(pretrained_dir_name(RINGFORMER_PLUS), "ringformer-plus")
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            pretraineds = Path(temp_dir) / "rvc" / "models" / "pretraineds"
            (pretraineds / "hifi-gan").mkdir(parents=True)
            for name in ("f0G40k.pth", "f0D40k.pth"):
                (pretraineds / "hifi-gan" / name).write_bytes(b"")
            os.chdir(temp_dir)
            try:
                g, d = quiet(pretrained_selector, RINGFORMER_PLUS, 40000)
                self.assertEqual(Path(g), Path("rvc/models/pretraineds/hifi-gan/f0G40k.pth"))
                self.assertEqual(Path(d), Path("rvc/models/pretraineds/hifi-gan/f0D40k.pth"))
                (pretraineds / "ringformer-plus").mkdir()
                for name in ("f0G40k.pth", "f0D40k.pth"):
                    (pretraineds / "ringformer-plus" / name).write_bytes(b"")
                g, d = quiet(pretrained_selector, RINGFORMER_PLUS, 40000)
                self.assertEqual(
                    Path(g), Path("rvc/models/pretraineds/ringformer-plus/f0G40k.pth")
                )
            finally:
                os.chdir(cwd)

    def test_the_cli_offers_it(self):
        import core

        with mock.patch.object(
            argparse.ArgumentParser,
            "parse_args",
            autospec=True,
            side_effect=lambda parser, *args, **kwargs: parser,
        ):
            parser = quiet(core.parse_arguments)
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        vocoder = next(
            item for item in subparsers.choices["train"]._actions if item.dest == "vocoder"
        )
        self.assertIn(RINGFORMER_PLUS, vocoder.choices)
        self.assertIn(CODENAME_RINGFORMER, vocoder.choices)


class ConstantQTransformTest(unittest.TestCase):
    """The CQT is nnAudio's CQT2010v2, carried over without its persistent buffers."""

    def test_it_agrees_with_librosa_on_tones(self):
        import librosa

        sample_rate = 96000
        t = np.arange(int(sample_rate * 3.0)) / sample_rate
        tones = (45, 95, 220, 7000)
        x = sum(0.2 * np.sin(2 * np.pi * f * t) for f in tones).astype(np.float32)
        for hop_length, n_octaves, bins_per_octave in ((512, 9, 24), (256, 9, 48)):
            with self.subTest(bins_per_octave=bins_per_octave):
                cqt = ConstantQTransform(
                    sample_rate, hop_length, n_octaves * bins_per_octave, bins_per_octave
                )
                ours = cqt(torch.from_numpy(x)[None, None])[0].norm(dim=-1).numpy()
                reference = np.abs(
                    librosa.cqt(
                        x,
                        sr=sample_rate,
                        hop_length=hop_length,
                        fmin=CQT_FMIN,
                        n_bins=n_octaves * bins_per_octave,
                        bins_per_octave=bins_per_octave,
                        pad_mode="constant",
                    )
                )
                frames = slice(ours.shape[1] // 4, 3 * ours.shape[1] // 4)
                freqs = CQT_FMIN * 2 ** (np.arange(ours.shape[0]) / bins_per_octave)
                for tone in tones:
                    k = int(np.argmin(np.abs(freqs - tone)))
                    ratio = ours[k, frames].mean() / reference[k, frames].mean()
                    self.assertAlmostEqual(ratio, 1.0, delta=0.01, msg=f"{tone} Hz")

    @unittest.skipUnless(importlib.util.find_spec("nnAudio"), "nnAudio is not installed")
    def test_it_is_bit_identical_to_nnaudio(self):
        from nnAudio.features.cqt import CQT2010v2

        for sample_rate in SAMPLE_RATES:
            for hop_length, n_octaves, bins_per_octave in CQT_SCALES:
                with self.subTest(sample_rate=sample_rate, bins_per_octave=bins_per_octave):
                    kwargs = dict(
                        hop_length=hop_length,
                        n_bins=n_octaves * bins_per_octave,
                        bins_per_octave=bins_per_octave,
                    )
                    reference = quiet(
                        CQT2010v2,
                        sr=2 * sample_rate,
                        output_format="Complex",
                        pad_mode="constant",
                        verbose=False,
                        **kwargs,
                    )
                    ours = ConstantQTransform(2 * sample_rate, **kwargs)
                    x = torch.randn(2, 1, 2 * segment(sample_rate)) * 0.3
                    self.assertTrue(torch.equal(reference(x), ours(x)))

    def test_nothing_is_saved_and_nothing_decimates_early(self):
        for sample_rate in SAMPLE_RATES:
            for hop_length, n_octaves, bins_per_octave in CQT_SCALES:
                cqt = ConstantQTransform(
                    2 * sample_rate, hop_length, n_octaves * bins_per_octave, bins_per_octave
                )
                self.assertEqual(cqt.state_dict(), {})
                # A hop of 256 or 512 over 9 octaves leaves no factor of two for early
                # decimation, so the transform behaves the same at every rate.
                self.assertEqual(cqt.downsample_factor, 1)
                self.assertEqual(cqt.hop_length, hop_length)

    def test_output_shapes_at_the_training_segment(self):
        # (bins, frames): 9 octaves of bins; frames = 2 * segment // hop + 1.
        expected = {
            32000: [(216, 51), (324, 101), (432, 101)],
            40000: [(216, 51), (324, 101), (432, 101)],
            48000: [(216, 68), (324, 136), (432, 136)],
        }
        for sample_rate, shapes in expected.items():
            x = torch.randn(1, 1, 2 * segment(sample_rate))
            got = [
                tuple(ConstantQTransform(2 * sample_rate, h, n * b, b)(x).shape[1:3])
                for h, n, b in CQT_SCALES
            ]
            self.assertEqual(got, shapes, sample_rate)


class DiscriminatorTest(unittest.TestCase):
    def test_the_layout_is_codename_ringformers_plus_three_cqts(self):
        expected = discriminator_layout(BASE_VERSION) + [
            ("Q", hop, octaves, bins, CQT_FMIN) for hop, octaves, bins in CQT_SCALES
        ]
        self.assertEqual(discriminator_layout(PLUS_VERSION), expected)
        for sample_rate in SAMPLE_RATES:
            net = fresh_d(PLUS_VERSION, sample_rate)
            self.assertEqual([describe_discriminator(d) for d in net.discriminators], expected)

    def test_it_needs_the_sample_rate_and_the_others_do_not(self):
        with self.assertRaises(ValueError):
            MultiPeriodDiscriminator(False, version=PLUS_VERSION)
        for version in DISCRIMINATOR_VERSIONS:
            if not DISCRIMINATOR_VERSIONS[version].get("cqts"):
                MultiPeriodDiscriminator(False, version=version)

    def test_only_the_cqt_layout_has_cqts_and_it_is_not_always_recorded(self):
        self.assertEqual(
            [v for v in DISCRIMINATOR_VERSIONS if DISCRIMINATOR_VERSIONS[v].get("cqts")],
            [PLUS_VERSION],
        )
        # No other version has a CQT discriminator, so the weights alone identify it.
        self.assertFalse(DISCRIMINATOR_VERSIONS[PLUS_VERSION].get("always_recorded", False))

    def test_the_same_seed_gives_the_shared_ones_and_the_rng_stream_unchanged(self):
        """What makes an A/B against CodenameRingFormer fair from the first step."""
        torch.manual_seed(1234)
        base = fresh_d(BASE_VERSION).state_dict()
        after_base = torch.rand(8)
        torch.manual_seed(1234)
        plus = fresh_d(PLUS_VERSION).state_dict()
        after_plus = torch.rand(8)
        for key in base:
            self.assertTrue(torch.equal(base[key], plus[key]), key)
        self.assertTrue(torch.equal(after_base, after_plus))
        # ...while the CQT discriminators themselves still follow the seed.
        torch.manual_seed(1234)
        again = fresh_d(PLUS_VERSION).state_dict()
        torch.manual_seed(4321)
        other = fresh_d(PLUS_VERSION).state_dict()
        key = "discriminators.12.convs.0.weight"
        self.assertTrue(torch.equal(plus[key], again[key]))
        self.assertFalse(torch.equal(plus[key], other[key]))

    def test_only_trained_weights_are_saved(self):
        state = fresh_d(PLUS_VERSION).state_dict()
        extra = set(state) - set(fresh_d(BASE_VERSION).state_dict())
        self.assertTrue(extra)
        self.assertTrue(all(key.split(".")[1] in ("12", "13", "14") for key in extra))
        self.assertFalse([k for k in extra if "kernel" in k or "lengths" in k or "lowpass" in k])
        # The same keys at every rate, although the CQT kernels' shapes are not.
        for sample_rate in (32000, 40000):
            other = fresh_d(PLUS_VERSION, sample_rate).state_dict()
            self.assertEqual(list(other), list(state))
            self.assertTrue(all(other[k].shape == state[k].shape for k in state))

    def test_weight_norm_sits_where_bigvgan_puts_it(self):
        d = fresh_d(PLUS_VERSION).discriminators[12]
        self.assertIsInstance(d, DiscriminatorCQT)
        normed = lambda m: hasattr(m, "parametrizations")
        self.assertFalse(any(normed(conv) for conv in d.conv_pres))
        self.assertFalse(normed(d.convs[0]))
        self.assertTrue(all(normed(conv) for conv in d.convs[1:]))
        self.assertTrue(normed(d.conv_post))
        self.assertEqual(d.convs[0].out_channels, CQT_FILTERS)
        self.assertEqual(len(d.conv_pres), 9)

    def _step(self, net, sample_rate, device, dtype=None):
        y = torch.randn(2, 1, segment(sample_rate), device=device) * 0.1
        y_hat = (torch.randn(2, 1, segment(sample_rate), device=device) * 0.1).requires_grad_(True)
        with torch.amp.autocast(
            device_type=device, enabled=dtype is not None, dtype=dtype or torch.float32
        ):
            y_d_r, y_d_g, fmap_r, fmap_g = net(y, y_hat)
        loss_disc, _, _ = discriminator_loss(y_d_r, y_d_g)
        loss_gen, _ = generator_loss(y_d_g)
        loss = loss_disc + loss_gen + feature_loss(fmap_r, fmap_g)
        loss.backward()
        self.assertEqual(len(y_d_r), 15)
        self.assertTrue(torch.isfinite(loss), f"loss is {loss}")
        self.assertTrue(torch.isfinite(y_hat.grad).all())
        for index in CQT_INDICES:
            self.assertEqual(len(fmap_r[index]), 5)
            for name, parameter in net.discriminators[index].named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_forward_and_backward_in_fp32_at_every_rate(self):
        for sample_rate in SAMPLE_RATES:
            with self.subTest(sample_rate=sample_rate):
                self._step(fresh_d(PLUS_VERSION, sample_rate), sample_rate, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_forward_and_backward_under_half_precision_autocast(self):
        dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for sample_rate in SAMPLE_RATES:
            for dtype in dtypes:
                with self.subTest(sample_rate=sample_rate, dtype=dtype):
                    self._step(
                        fresh_d(PLUS_VERSION, sample_rate).cuda(), sample_rate, "cuda", dtype
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_the_transform_runs_in_fp32_under_autocast(self):
        """Resample and CQT are convolutions, which autocast would run in half precision
        even on an fp32 input; they must come out exactly as they do without it."""
        d = fresh_d(PLUS_VERSION).discriminators[14].cuda()
        captured = []
        d.cqt.register_forward_hook(lambda module, inputs, output: captured.append(output))
        x = torch.randn(2, 1, segment(48000), device="cuda") * 0.1
        for dtype in (torch.float16, torch.bfloat16):
            for given in (x, x.to(dtype)):
                with self.subTest(autocast=dtype, input=given.dtype):
                    with torch.no_grad():
                        # The fp32 transform of exactly what the discriminator was given.
                        reference = d.cqt(d.resample(given.float()))
                        captured.clear()
                        with torch.amp.autocast(device_type="cuda", dtype=dtype):
                            d(given)
                    self.assertEqual(captured[0].dtype, torch.float32)
                    self.assertTrue(torch.equal(captured[0], reference))


class DiscriminatorWarmStartTest(unittest.TestCase):
    """Sub-discriminators are matched by what they look at; the CQT ones exist only here."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.sources = {}
        for name, version in (
            ("v2", "v2"),
            ("codename", BASE_VERSION),
            ("plus", PLUS_VERSION),
        ):
            net = fresh_d(version)
            path = os.path.join(cls.tmp.name, f"D_{name}.pth")
            quiet(save_legacy, net, path, {"disc_version": version})
            cls.sources[name] = (path, net.state_dict())

    def _warm(self, source, target_version):
        target = fresh_d(target_version)
        before = snapshot(target)
        transfer = quiet(warm_start, target, self.sources[source][0], "D")
        self.assertEqual(transfer.errors, [])
        return target.state_dict(), before, self.sources[source][1], transfer

    def _assert_inherited(self, loaded, saved, target_index, source_index):
        keys = sub_keys(loaded, target_index)
        self.assertTrue(keys)
        for key in keys:
            source_key = f"discriminators.{source_index}." + key.split(".", 2)[2]
            self.assertTrue(torch.equal(loaded[key], saved[source_key]), key)

    def _assert_fresh(self, loaded, before, index):
        keys = sub_keys(loaded, index)
        self.assertTrue(keys)
        for key in keys:
            self.assertTrue(torch.equal(loaded[key], before[key]), key)

    def test_from_codename_ringformer_only_the_cqts_start_fresh(self):
        loaded, before, saved, transfer = self._warm("codename", PLUS_VERSION)
        for index in SHARED:
            self._assert_inherited(loaded, saved, index, index)
        for index in CQT_INDICES:
            self._assert_fresh(loaded, before, index)
        self.assertEqual(transfer.dropped, {})
        reasons = set(transfer.reinitialised.values())
        self.assertEqual(
            reasons,
            {
                f"the pretrained discriminator has no CQT discriminator (hop {h}, {n} "
                f"octaves x {b} bins from {CQT_FMIN:g} Hz)"
                for h, n, b in CQT_SCALES
            },
        )
        inherited = sum(loaded[k].numel() for i in SHARED for k in sub_keys(loaded, i))
        total = sum(v.numel() for v in loaded.values())
        self.assertGreater(inherited / total, 0.99)

    def test_from_hifigan_the_scale_and_all_eight_periods_come(self):
        loaded, before, saved, transfer = self._warm("v2", PLUS_VERSION)
        for index in range(9):  # scale + periods 2 .. 37, the same order in both
            self._assert_inherited(loaded, saved, index, index)
        for index in (9, 10, 11, *CQT_INDICES):
            self._assert_fresh(loaded, before, index)
        self.assertEqual(transfer.dropped, {})

    def test_back_to_codename_ringformer_drops_only_the_cqts(self):
        loaded, _, saved, transfer = self._warm("plus", BASE_VERSION)
        self.assertEqual(transfer.reinitialised, {})
        for index in SHARED:
            self._assert_inherited(loaded, saved, index, index)
        dropped = set(transfer.dropped.values())
        self.assertEqual(len(dropped), 3)
        self.assertTrue(all(r.startswith("this model has no CQT discriminator") for r in dropped))

    def test_the_same_layout_loads_whole(self):
        loaded, _, saved, transfer = self._warm("plus", PLUS_VERSION)
        self.assertEqual(transfer.reinitialised, {})
        self.assertEqual(transfer.dropped, {})
        for key, value in saved.items():
            self.assertTrue(torch.equal(loaded[key], value), key)

    def test_the_layout_is_read_off_the_weights_without_a_record(self):
        path = os.path.join(self.tmp.name, "D_plus_unrecorded.pth")
        net = fresh_d(PLUS_VERSION)
        quiet(save_legacy, net, path, None)
        transfer = quiet(warm_start, fresh_d(PLUS_VERSION), path, "D")
        self.assertEqual(transfer.reinitialised, {})

    def test_a_record_that_contradicts_the_weights_stops_the_run(self):
        path = os.path.join(self.tmp.name, "D_lying.pth")
        quiet(save_legacy, fresh_d(BASE_VERSION), path, {"disc_version": PLUS_VERSION})
        with self.assertRaises(SystemExit):
            quiet(warm_start, fresh_d(PLUS_VERSION), path, "D")


class GeneratorWarmStartTest(unittest.TestCase):
    """The generator is CodenameRingFormer's, so it warm starts exactly as that does."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.paths = {}
        for name, vocoder, disc_version in (
            ("hifigan", HIFIGAN, "v2"),
            ("codename", CODENAME_RINGFORMER, BASE_VERSION),
            ("plus", CODENAME_RINGFORMER, PLUS_VERSION),
        ):
            path = os.path.join(cls.tmp.name, f"G_{name}.pth")
            quiet(save_legacy, generator(vocoder), path, arch(vocoder, disc_version))
            cls.paths[name] = path

    def _warm(self, source, recipe):
        target = generator(generator_vocoder(recipe))
        transfer = quiet(
            warm_start,
            target,
            self.paths[source],
            "G",
            target_identity=arch(generator_vocoder(recipe), discriminator_version(recipe)),
        )
        self.assertEqual(transfer.errors, [])
        saved = normalize_weight_norm_keys(
            torch.load(self.paths[source], map_location="cpu", weights_only=True)["model"]
        )
        return target.state_dict(), saved, transfer

    def test_between_the_two_recipes_everything_carries_over(self):
        for source, recipe in (("codename", RINGFORMER_PLUS), ("plus", CODENAME_RINGFORMER)):
            with self.subTest(source=source, recipe=recipe):
                state, saved, transfer = self._warm(source, recipe)
                self.assertEqual(transfer.reinitialised, {})
                self.assertEqual(transfer.dropped, {})
                for key in state:
                    self.assertTrue(torch.equal(state[key], saved[key].float()), key)

    def test_from_hifigan_it_inherits_what_codename_ringformer_does(self):
        _, _, plus = self._warm("hifigan", RINGFORMER_PLUS)
        _, _, base = self._warm("hifigan", CODENAME_RINGFORMER)
        self.assertEqual(set(plus.reinitialised), set(base.reinitialised))
        self.assertEqual(set(plus.dropped), set(base.dropped))
        self.assertTrue(plus.reinitialised)


class ResumeGuardTest(unittest.TestCase):
    def test_the_two_recipes_refuse_each_other(self):
        for was, now in ((BASE_VERSION, PLUS_VERSION), (PLUS_VERSION, BASE_VERSION)):
            with self.subTest(was=was, now=now):
                reason = describe_architecture_mismatch(
                    arch(CODENAME_RINGFORMER, was), arch(CODENAME_RINGFORMER, now)
                )
                self.assertIn(f"discriminator version {was} -> {now}", reason)
                self.assertNotIn("vocoder", reason)
        self.assertIsNone(
            describe_architecture_mismatch(
                arch(CODENAME_RINGFORMER, PLUS_VERSION), arch(CODENAME_RINGFORMER, PLUS_VERSION)
            )
        )

    def test_a_saved_codename_ringformer_checkpoint_is_refused_and_the_reset_suggested(self):
        with tempfile.TemporaryDirectory() as folder:
            quiet(
                save_legacy,
                generator(CODENAME_RINGFORMER),
                os.path.join(folder, "G_100.pth"),
                arch(CODENAME_RINGFORMER, BASE_VERSION),
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
                assert_resumable(
                    folder,
                    {},
                    arch(
                        generator_vocoder(RINGFORMER_PLUS),
                        discriminator_version(RINGFORMER_PLUS),
                    ),
                )
        self.assertIn(
            "discriminator version codename-ringformer -> codename-ringformer-cqt",
            output.getvalue(),
        )
        self.assertIn("RingFormer+", output.getvalue())
        self.assertIn("--reset_training", output.getvalue())


class ResetTest(unittest.TestCase):
    """A reset is how an existing folder moves between the two recipes: the generator is
    kept and only the discriminator is rebuilt."""

    def _folder(self, disc_version, sample_rate=48000):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / "branch"
        root.mkdir()
        config = stock(sample_rate, CODENAME_RINGFORMER)
        config["model"]["text_enc_hidden_dim"] = 768
        (root / "config.json").write_text(json.dumps(config))
        net_g = generator(CODENAME_RINGFORMER, sample_rate)
        net_d = fresh_d(disc_version, sample_rate)
        identity = arch(CODENAME_RINGFORMER, disc_version, sample_rate)
        for tag, net in (("G", net_g), ("D", net_d)):
            quiet(
                save_checkpoint,
                net,
                torch.optim.AdamW(net.parameters(), lr=1e-4),
                1e-4,
                840,
                str(root / f"{tag}_2333333.pth"),
                torch.amp.GradScaler(enabled=False),
                architecture_identity=identity,
            )
        return root, snapshot(net_g), snapshot(net_d)

    @staticmethod
    def _installed(root, tag):
        return torch.load(root / f"{tag}_0.pth", map_location="cpu", weights_only=True)

    def _check(self, was, recipe, sample_rate=48000):
        now = discriminator_version(recipe)
        root, source_g, source_d = self._folder(was, sample_rate)
        archive = quiet(reset_training_run, root, vocoder=recipe)

        generator_checkpoint = self._installed(root, "G")
        self.assertEqual(generator_checkpoint["vocoder"], CODENAME_RINGFORMER)
        self.assertEqual(generator_checkpoint["disc_version"], now)
        weights = normalize_weight_norm_keys(generator_checkpoint["model"])
        for key, value in source_g.items():
            self.assertTrue(torch.equal(weights[key], value), key)

        discriminator_checkpoint = self._installed(root, "D")
        self.assertEqual(discriminator_checkpoint["disc_version"], now)
        target = fresh_d(now, sample_rate)
        target.load_state_dict(discriminator_checkpoint["model"])
        loaded = target.state_dict()
        for index in SHARED:
            for key in sub_keys(loaded, index):
                self.assertTrue(torch.equal(loaded[key], source_d[key]), key)
        self.assertEqual(
            [describe_discriminator(d) for d in target.discriminators],
            discriminator_layout(now),
        )

        # The run that follows resumes from what the reset installed.
        quiet(
            assert_resumable,
            str(root),
            {},
            arch(generator_vocoder(recipe), now, sample_rate),
        )
        record = json.loads((archive / "reset.json").read_text())
        self.assertIsNone(record["generator_transfer"])
        self.assertIsNotNone(record["discriminator_transfer"])
        return discriminator_checkpoint

    def test_codename_ringformer_to_ringformer_plus(self):
        installed = self._check(BASE_VERSION, RINGFORMER_PLUS)
        self.assertTrue(sub_keys(installed["model"], 14))

    def test_ringformer_plus_to_codename_ringformer(self):
        installed = self._check(PLUS_VERSION, CODENAME_RINGFORMER)
        self.assertFalse(sub_keys(installed["model"], 12))

    def test_at_32k(self):
        """The reset builds the CQT discriminators at the folder's own sample rate."""
        self._check(BASE_VERSION, RINGFORMER_PLUS, sample_rate=32000)

    def test_the_same_recipe_leaves_the_discriminator_alone(self):
        root, _, source_d = self._folder(PLUS_VERSION)
        quiet(reset_training_run, root, vocoder=RINGFORMER_PLUS)
        discriminator_checkpoint = self._installed(root, "D")
        self.assertEqual(discriminator_checkpoint["disc_version"], PLUS_VERSION)
        for key, value in source_d.items():
            self.assertTrue(torch.equal(discriminator_checkpoint["model"][key], value), key)


class ExportTest(unittest.TestCase):
    """What a RingFormer+ run exports is a CodenameRingFormer model, loaded by the real
    inference code."""

    def test_the_exported_model_loads_through_voice_converter(self):
        from rvc.infer.infer import VoiceConverter

        config = stock(48000, CODENAME_RINGFORMER)
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "model_info.json"), "w") as f:
                json.dump({"embedder_model": "contentvec", "speakers_id": 3}, f)
            hps = HParams(**json.loads(json.dumps(config)))
            hps.model.text_enc_hidden_dim = 768
            hps.model.spk_embed_dim = 3
            # train.py builds and exports with generator_vocoder(training_recipe).
            vocoder = generator_vocoder(RINGFORMER_PLUS)
            net_g = generator(vocoder, speakers=3)
            path = os.path.join(directory, "m_1e_1s.pth")
            quiet(
                extract_model,
                net_g.state_dict(),
                48000,
                "m",
                path,
                1,
                1,
                hps,
                "info",
                vocoder,
            )
            self.assertTrue(os.path.isfile(path), "extract_model produced no file")
            cpt = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(cpt["vocoder"], CODENAME_RINGFORMER)
            self.assertEqual(
                (cpt["gen_istft_n_fft"], cpt["gen_istft_hop_size"]),
                (config["model"]["gen_istft_n_fft"], config["model"]["gen_istft_hop_size"]),
            )

            converter = quiet(VoiceConverter)
            converter.load_model(path)
            quiet(converter.setup_network)
        self.assertIsInstance(converter.net_g.dec, CodenameRingFormerGenerator)
        device = next(converter.net_g.parameters()).device
        length = 60
        with torch.no_grad():
            audio, _, _ = converter.net_g.infer(
                torch.randn(1, length, 768, device=device),
                torch.full((1,), length, dtype=torch.long, device=device),
                torch.randint(1, 255, (1, length), device=device),
                fake_f0(1, length).to(device),
                torch.zeros(1, dtype=torch.long, device=device),
            )
        self.assertEqual(audio.shape, (1, 1, length * config["data"]["hop_length"]))
        self.assertTrue(torch.isfinite(audio).all())


if __name__ == "__main__":
    unittest.main()
