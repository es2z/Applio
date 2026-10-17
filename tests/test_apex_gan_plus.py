"""APEX-GAN+: the APEX-GAN generator against resolution discriminators made for 48 kHz.

Run from the repository root:
    env\\python.exe -m unittest tests.test_apex_gan_plus -v

Real configs and real models, no mocks except where a test must prove training was never
started, in the style of tests/test_sifigan_plus.py. Kept in its own file so that removing
either APEX-GAN recipe later only touches what it has to.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rvc.lib.algorithm.discriminators import (
    DISCRIMINATOR_VERSIONS,
    describe_discriminator,
    discriminator_layout,
)
from rvc.lib.algorithm.generators.apex_gan import APEXGANGenerator
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
    APEX_GAN_PLUS,
    discriminator_version,
    generator_vocoder,
    pretrained_dir_name,
    sample_rate_error,
    uses_multiscale_mel_loss,
)
from rvc.train.warm_start import (
    APEX_GAN,
    HIFIGAN,
    detect_vocoder,
    normalize_weight_norm_keys,
    warm_start,
)

from tests.test_apex_gan import (
    SEGMENT,
    build_synthesizer,
    fake_f0,
    fresh_d,
    quiet,
    stock,
)
from tests.test_warm_start import save_legacy, snapshot

APEX_VERSION = "apex-gan"
PLUS_VERSION = "apex-gan-48k"
PLUS_RESOLUTIONS = ((2048, 240, 1200), (4096, 480, 2400), (1024, 100, 480))
# Sub-discriminator indices shared by apex-gan and apex-gan-48k.
PERIODS = range(0, 5)
RESOLUTIONS = range(5, 8)
SUBBANDS = range(8, 12)


def generator(vocoder, speakers=1):
    return build_synthesizer(vocoder, 48000, 768, speakers)


def arch(vocoder, disc_version):
    return {"vocoder": vocoder, "sample_rate": 48000, "disc_version": disc_version}


def sub_keys(state, index):
    return [key for key in state if key.startswith(f"discriminators.{index}.")]


class RecipeTest(unittest.TestCase):
    """What the recipe means, and that it only ever changes the discriminator."""

    def test_apex_gan_plus_is_the_apex_gan_generator_against_apex_gan_48k(self):
        self.assertEqual(APEX_GAN_PLUS, "APEX-GAN+")
        self.assertEqual(generator_vocoder(APEX_GAN_PLUS), APEX_GAN)
        self.assertEqual(discriminator_version(APEX_GAN_PLUS), PLUS_VERSION)
        # APEX-GAN's losses: the single-scale mel loss.
        self.assertFalse(uses_multiscale_mel_loss(APEX_GAN_PLUS))

    def test_apex_gan_itself_is_unchanged(self):
        self.assertEqual(generator_vocoder(APEX_GAN), APEX_GAN)
        self.assertEqual(discriminator_version(APEX_GAN), APEX_VERSION)
        self.assertEqual(pretrained_dir_name(APEX_GAN), "apex-gan")

    def test_the_recipe_name_must_never_reach_synthesizer(self):
        """Why train.py and reset_run.py normalise first: Synthesizer builds a HiFi-GAN
        decoder for any name it does not know, without a word."""
        unnormalised = generator(APEX_GAN_PLUS)
        self.assertEqual(detect_vocoder(unnormalised.state_dict()), HIFIGAN)
        normalised = generator(generator_vocoder(APEX_GAN_PLUS))
        self.assertEqual(detect_vocoder(normalised.state_dict()), APEX_GAN)
        self.assertIsInstance(normalised.dec, APEXGANGenerator)

    def test_the_pretrained_folder_has_no_plus_and_falls_back_to_hifigan(self):
        self.assertEqual(pretrained_dir_name(APEX_GAN_PLUS), "apex-gan-plus")
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            pretraineds = Path(temp_dir) / "rvc" / "models" / "pretraineds"
            (pretraineds / "hifi-gan").mkdir(parents=True)
            for name in ("f0G48k.pth", "f0D48k.pth"):
                (pretraineds / "hifi-gan" / name).write_bytes(b"")
            os.chdir(temp_dir)
            try:
                g, d = quiet(pretrained_selector, APEX_GAN_PLUS, 48000)
                self.assertEqual(Path(g), Path("rvc/models/pretraineds/hifi-gan/f0G48k.pth"))
                self.assertEqual(Path(d), Path("rvc/models/pretraineds/hifi-gan/f0D48k.pth"))
                (pretraineds / "apex-gan-plus").mkdir()
                for name in ("f0G48k.pth", "f0D48k.pth"):
                    (pretraineds / "apex-gan-plus" / name).write_bytes(b"")
                g, d = quiet(pretrained_selector, APEX_GAN_PLUS, 48000)
                self.assertEqual(
                    Path(g), Path("rvc/models/pretraineds/apex-gan-plus/f0G48k.pth")
                )
                self.assertEqual(
                    Path(d), Path("rvc/models/pretraineds/apex-gan-plus/f0D48k.pth")
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
        self.assertIn(APEX_GAN_PLUS, vocoder.choices)
        self.assertIn(APEX_GAN, vocoder.choices)


class SampleRateTest(unittest.TestCase):
    """48 kHz only, refused before anything touches the model folder."""

    def test_only_48k_is_accepted_for_apex_gans_reason(self):
        self.assertIsNone(sample_rate_error(APEX_GAN_PLUS, 48000))
        self.assertIsNone(sample_rate_error(APEX_GAN_PLUS, "48000"))
        for sample_rate in (32000, 40000):
            with self.subTest(sample_rate=sample_rate):
                message = sample_rate_error(APEX_GAN_PLUS, sample_rate)
                self.assertIn("APEX-GAN+ trains only at 48 kHz", message)
                self.assertIn(str(sample_rate), message)
                # The generator limits it first: APEX-GAN's decoder config is 48k only.
                self.assertIn("upsampling", message)
                self.assertEqual(
                    message.split(". ", 1)[1], sample_rate_error(APEX_GAN, sample_rate).split(". ", 1)[1]
                )
        # SiFi-GAN+'s reason is about its discriminator, and stays that way.
        self.assertIn("STFT resolutions", sample_rate_error("SiFi-GAN+", 40000))

    def test_core_refuses_before_touching_the_folder(self):
        import core

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            core, "logs_path", temp_dir
        ), mock.patch.object(
            core.subprocess, "run", side_effect=AssertionError("training was started")
        ):
            folder = Path(temp_dir) / "m"
            folder.mkdir()
            (folder / "config.json").write_text(json.dumps(stock(40000)), encoding="utf-8")
            before = (folder / "config.json").read_bytes()
            # The GUI's radio says 48 kHz, but the folder was extracted at 40 kHz, and
            # that is what train.py would train at.
            with self.assertRaises(ValueError) as caught:
                core.run_train_script(
                    model_name="m",
                    save_every_epoch=1,
                    save_only_latest=False,
                    save_every_weights=False,
                    total_epoch=1,
                    sample_rate=48000,
                    batch_size=1,
                    gpu=0,
                    overtraining_detector=False,
                    overtraining_threshold=50,
                    pretrained=False,
                    cleanup=False,
                    vocoder=APEX_GAN_PLUS,
                    reset_training=True,
                    learning_rate=5e-5,
                )
            self.assertIn("48 kHz", str(caught.exception))
            self.assertEqual((folder / "config.json").read_bytes(), before)
            self.assertEqual(sorted(os.listdir(temp_dir)), ["m"])

    def test_train_py_refuses_a_40k_folder(self):
        """The last line of defence when train.py is started directly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir) / "logs" / "m"
            folder.mkdir(parents=True)
            (folder / "config.json").write_text(json.dumps(stock(40000)), encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
            argv = ["m", "1", "1", "", "", "0", "1", "40000", "False", "False", "False",
                    "False", "0", "False", APEX_GAN_PLUS, "False"]
            result = subprocess.run(
                [sys.executable, str(REPO / "rvc" / "train" / "train.py"), *argv],
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=600,
            )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("APEX-GAN+ trains only at 48 kHz", result.stdout)


class DiscriminatorTest(unittest.TestCase):
    def test_the_layout_is_apex_gans_with_48k_resolutions(self):
        expected = (
            [("P", period) for period in (2, 3, 5, 7, 11)]
            + [("R", resolution, "hann") for resolution in PLUS_RESOLUTIONS]
        )
        layout = discriminator_layout(PLUS_VERSION)
        self.assertEqual(layout[:8], expected)
        # The sub-band discriminators are apex-gan's, unchanged.
        self.assertEqual(layout[8:], discriminator_layout(APEX_VERSION)[8:])
        self.assertEqual(len(layout), 12)
        net = fresh_d(PLUS_VERSION)
        self.assertEqual([describe_discriminator(d) for d in net.discriminators], layout)

    def test_the_resolutions_are_v3_48ks_and_codename_ringformers(self):
        """Hann included, so a SiFi-GAN+ or RingFormer D gives its resolution
        discriminators to this one."""
        def resolutions(version):
            return [d for d in discriminator_layout(version) if d[0] == "R"]

        self.assertEqual(resolutions(PLUS_VERSION), resolutions("v3-48k"))
        self.assertEqual(resolutions(PLUS_VERSION), resolutions("codename-ringformer"))

    def test_it_is_apex_gan_to_the_weights_and_something_else_to_the_warm_start(self):
        apex, plus = fresh_d(APEX_VERSION).state_dict(), fresh_d(PLUS_VERSION).state_dict()
        # Identical names and shapes: the weights alone cannot tell the two apart...
        self.assertEqual(list(apex), list(plus))
        self.assertTrue(all(apex[key].shape == plus[key].shape for key in apex))
        # ...but no resolution discriminator of one means the same as one of the other,
        # not even [2048, 240, 1200], which apex-gan looks at through a rectangular window.
        def resolutions(version):
            return {d for d in discriminator_layout(version) if d[0] == "R"}

        self.assertFalse(resolutions(APEX_VERSION) & resolutions(PLUS_VERSION))
        self.assertEqual(
            [d for d in discriminator_layout(APEX_VERSION) if d[0] != "R"],
            [d for d in discriminator_layout(PLUS_VERSION) if d[0] != "R"],
        )

    def test_only_the_two_later_layouts_are_always_recorded(self):
        recorded = {
            version
            for version, spec in DISCRIMINATOR_VERSIONS.items()
            if spec.get("always_recorded", False)
        }
        self.assertEqual(recorded, {"v3-48k", PLUS_VERSION})

    def test_the_same_seed_initialises_it_exactly_like_apex_gan(self):
        """What makes an A/B against APEX-GAN fair from the first step."""
        torch.manual_seed(1234)
        apex = fresh_d(APEX_VERSION).state_dict()
        torch.manual_seed(1234)
        plus = fresh_d(PLUS_VERSION).state_dict()
        for key in apex:
            self.assertTrue(torch.equal(apex[key], plus[key]), key)

    def test_spectrogram_shapes_at_48k(self):
        net = fresh_d(PLUS_VERSION)
        x = torch.randn(2, 1, SEGMENT)
        shapes = [
            tuple(net.discriminators[index].spectrogram(x).shape[1:])
            for index in RESOLUTIONS
        ]
        # The same pixel count as apex-gan's (513 x 144, 1025 x 72, 257 x 345): doubling
        # the FFT doubles the bins and doubling the hop halves the frames.
        self.assertEqual(shapes, [(1025, 72), (2049, 36), (513, 172)])

    def _step(self, net, device, dtype=None):
        y = torch.randn(2, 1, SEGMENT, device=device) * 0.3
        y_hat = (torch.randn(2, 1, SEGMENT, device=device) * 0.3).requires_grad_(True)
        with torch.amp.autocast(
            device_type=device, enabled=dtype is not None, dtype=dtype or torch.float32
        ):
            y_d_r, y_d_g, fmap_r, fmap_g = net(y, y_hat)
        loss_disc, _, _ = discriminator_loss(y_d_r, y_d_g)
        loss_gen, _ = generator_loss(y_d_g)
        loss = loss_disc + loss_gen + feature_loss(fmap_r, fmap_g)
        loss.backward()
        self.assertEqual(len(y_d_r), 12)
        self.assertTrue(torch.isfinite(loss), f"loss is {loss}")
        self.assertTrue(torch.isfinite(y_hat.grad).all())
        for name, parameter in net.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_forward_and_backward_in_fp32(self):
        torch.manual_seed(0)
        self._step(fresh_d(PLUS_VERSION), "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_forward_and_backward_under_half_precision_autocast(self):
        dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                self._step(fresh_d(PLUS_VERSION).cuda(), "cuda", dtype)


class DiscriminatorWarmStartTest(unittest.TestCase):
    """Sub-discriminators are matched by what they look at: apex-gan and apex-gan-48k share
    their periods and sub-band discriminators, and apex-gan-48k's resolution
    discriminators are v3-48k's and codename-ringformer's."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.sources = {}
        for name, version, recorded in (
            ("apex", APEX_VERSION, True),
            ("apex-unrecorded", APEX_VERSION, False),
            ("plus", PLUS_VERSION, True),
            ("v2", "v2", True),
            ("v3", "v3", True),
            ("v3-48k", "v3-48k", True),
            ("codename", "codename-ringformer", True),
            ("codename-cqt", "codename-ringformer-cqt", True),
        ):
            net = fresh_d(version)
            path = os.path.join(cls.tmp.name, f"D_{name}.pth")
            quiet(save_legacy, net, path, {"disc_version": version} if recorded else None)
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

    def test_apex_gan_gives_the_periods_and_sub_bands(self):
        for source in ("apex", "apex-unrecorded"):
            with self.subTest(source=source):
                loaded, before, saved, transfer = self._warm(source, PLUS_VERSION)
                for index in (*PERIODS, *SUBBANDS):
                    self._assert_inherited(loaded, saved, index, index)
                for index in RESOLUTIONS:
                    self._assert_fresh(loaded, before, index)
                dropped = set(transfer.dropped.values())
                self.assertEqual(len(dropped), 3)
                self.assertIn(
                    "this model has no STFT resolution [2048, 240, 1200] discriminator",
                    dropped,
                )
                self.assertIn(
                    "the pretrained discriminator has no STFT resolution "
                    "[2048, 240, 1200] (hann window) discriminator",
                    set(transfer.reinitialised.values()),
                )

    def test_back_to_apex_gan_keeps_the_periods_and_sub_bands(self):
        loaded, before, saved, transfer = self._warm("plus", APEX_VERSION)
        for index in (*PERIODS, *SUBBANDS):
            self._assert_inherited(loaded, saved, index, index)
        for index in RESOLUTIONS:
            self._assert_fresh(loaded, before, index)
        self.assertEqual(len(set(transfer.dropped.values())), 3)

    def test_the_same_layout_loads_whole(self):
        for source, version in (("plus", PLUS_VERSION), ("apex", APEX_VERSION)):
            with self.subTest(source=source):
                loaded, _, saved, transfer = self._warm(source, version)
                self.assertEqual(transfer.reinitialised, {})
                self.assertEqual(transfer.dropped, {})
                for key, value in saved.items():
                    self.assertTrue(torch.equal(loaded[key], value), key)

    def test_hifigan_gives_the_shared_periods(self):
        loaded, before, saved, transfer = self._warm("v2", PLUS_VERSION)
        for index in PERIODS:  # v2: scale at 0, periods 2 .. 11 at 1 .. 5
            self._assert_inherited(loaded, saved, index, index + 1)
        for index in (*RESOLUTIONS, *SUBBANDS):
            self._assert_fresh(loaded, before, index)
        self.assertEqual(
            set(transfer.dropped.values()),
            {
                "this model has no scale discriminator",
                "this model has no period 17 discriminator",
                "this model has no period 23 discriminator",
                "this model has no period 37 discriminator",
            },
        )

    def test_v3_gives_only_the_periods(self):
        loaded, before, saved, transfer = self._warm("v3", PLUS_VERSION)
        for index in PERIODS:
            self._assert_inherited(loaded, saved, index, index + 1)
        for index in (*RESOLUTIONS, *SUBBANDS):
            self._assert_fresh(loaded, before, index)
        dropped = set(transfer.dropped.values())
        self.assertIn("this model has no scale discriminator", dropped)
        self.assertEqual(len([r for r in dropped if "STFT resolution" in r]), 3)

    def test_sifigan_plus_gives_the_periods_and_resolutions(self):
        loaded, before, saved, transfer = self._warm("v3-48k", PLUS_VERSION)
        # v3-48k: scale at 0, periods at 1 .. 5, resolutions at 6 .. 8.
        for index in (*PERIODS, *RESOLUTIONS):
            self._assert_inherited(loaded, saved, index, index + 1)
        for index in SUBBANDS:
            self._assert_fresh(loaded, before, index)
        self.assertEqual(set(transfer.dropped.values()), {"this model has no scale discriminator"})

    def test_ringformer_gives_the_periods_and_resolutions(self):
        for source in ("codename", "codename-cqt"):
            with self.subTest(source=source):
                loaded, before, saved, transfer = self._warm(source, PLUS_VERSION)
                for index in PERIODS:
                    self._assert_inherited(loaded, saved, index, index + 1)
                # codename-ringformer holds periods 17, 23 and 37 at 6 .. 8 and the
                # resolution discriminators after them.
                for target_index, source_index in ((5, 9), (6, 10), (7, 11)):
                    self._assert_inherited(loaded, saved, target_index, source_index)
                for index in SUBBANDS:
                    self._assert_fresh(loaded, before, index)
                dropped = set(transfer.dropped.values())
                self.assertIn("this model has no scale discriminator", dropped)
                self.assertFalse([r for r in dropped if "STFT resolution" in r])

    def test_a_record_that_contradicts_the_weights_stops_the_run(self):
        path = os.path.join(self.tmp.name, "D_lying.pth")
        quiet(save_legacy, fresh_d("v2"), path, {"disc_version": PLUS_VERSION})
        with self.assertRaises(SystemExit):
            quiet(warm_start, fresh_d(PLUS_VERSION), path, "D")


class GeneratorWarmStartTest(unittest.TestCase):
    """The generator is APEX-GAN's, so it warm starts exactly as APEX-GAN does."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.paths = {}
        for name, vocoder, disc_version in (
            ("hifigan", HIFIGAN, "v2"),
            ("apex", APEX_GAN, APEX_VERSION),
            ("plus", APEX_GAN, PLUS_VERSION),
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

    def test_from_apex_gan_and_back_everything_carries_over(self):
        for source, recipe in (("apex", APEX_GAN_PLUS), ("plus", APEX_GAN)):
            with self.subTest(source=source, recipe=recipe):
                state, saved, transfer = self._warm(source, recipe)
                self.assertEqual(transfer.reinitialised, {})
                self.assertEqual(transfer.dropped, {})
                for key in state:
                    self.assertTrue(torch.equal(state[key], saved[key].float()), key)

    def test_from_hifigan_it_inherits_what_apex_gan_does(self):
        _, _, plus = self._warm("hifigan", APEX_GAN_PLUS)
        _, _, apex = self._warm("hifigan", APEX_GAN)
        self.assertTrue(plus.reinitialised)
        self.assertEqual(set(plus.reinitialised), set(apex.reinitialised))
        self.assertEqual(set(plus.loaded), set(apex.loaded))


class ResumeGuardTest(unittest.TestCase):
    def test_apex_gan_and_apex_gan_plus_refuse_each_other(self):
        for was, now in ((APEX_VERSION, PLUS_VERSION), (PLUS_VERSION, APEX_VERSION)):
            with self.subTest(was=was, now=now):
                reason = describe_architecture_mismatch(
                    arch(APEX_GAN, was), arch(APEX_GAN, now)
                )
                self.assertIn(f"discriminator version {was} -> {now}", reason)
                self.assertNotIn("vocoder", reason)
        self.assertIsNone(
            describe_architecture_mismatch(
                arch(APEX_GAN, PLUS_VERSION), arch(APEX_GAN, PLUS_VERSION)
            )
        )

    def test_a_saved_apex_gan_checkpoint_is_refused_and_the_reset_suggested(self):
        with tempfile.TemporaryDirectory() as folder:
            quiet(
                save_legacy,
                generator(APEX_GAN),
                os.path.join(folder, "G_100.pth"),
                arch(APEX_GAN, APEX_VERSION),
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
                assert_resumable(
                    folder,
                    {},
                    arch(
                        generator_vocoder(APEX_GAN_PLUS),
                        discriminator_version(APEX_GAN_PLUS),
                    ),
                )
        self.assertIn("discriminator version apex-gan -> apex-gan-48k", output.getvalue())
        self.assertIn("APEX-GAN <-> APEX-GAN+", output.getvalue())
        self.assertIn("--reset_training", output.getvalue())


class ResetTest(unittest.TestCase):
    """A reset is how an existing folder moves between APEX-GAN and APEX-GAN+: the
    generator is kept and only the discriminator is rebuilt."""

    def _folder(self, disc_version):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / "branch"
        root.mkdir()
        # What train.py leaves in an APEX-GAN folder: the stock config with APEX-GAN's
        # decoder keys written into it.
        config = stock(48000, APEX_GAN)
        config["model"]["text_enc_hidden_dim"] = 768
        (root / "config.json").write_text(json.dumps(config))
        net_g = generator(APEX_GAN, speakers=config["model"]["spk_embed_dim"])
        net_d = fresh_d(disc_version)
        for tag, net in (("G", net_g), ("D", net_d)):
            quiet(
                save_checkpoint,
                net,
                torch.optim.AdamW(net.parameters(), lr=1e-4),
                1e-4,
                840,
                str(root / f"{tag}_2333333.pth"),
                torch.amp.GradScaler(enabled=False),
                architecture_identity=arch(APEX_GAN, disc_version),
            )
        return root, snapshot(net_g), snapshot(net_d)

    @staticmethod
    def _installed(root, tag):
        return torch.load(root / f"{tag}_0.pth", map_location="cpu", weights_only=True)

    def _check(self, was, recipe):
        now = discriminator_version(recipe)
        root, source_g, source_d = self._folder(was)
        archive = quiet(reset_training_run, root, vocoder=recipe)

        generator_checkpoint = self._installed(root, "G")
        self.assertEqual(generator_checkpoint["vocoder"], APEX_GAN)
        self.assertEqual(generator_checkpoint["disc_version"], now)
        weights = normalize_weight_norm_keys(generator_checkpoint["model"])
        for key, value in source_g.items():
            self.assertTrue(torch.equal(weights[key], value), key)

        discriminator_checkpoint = self._installed(root, "D")
        self.assertEqual(discriminator_checkpoint["disc_version"], now)
        target = fresh_d(now)
        target.load_state_dict(discriminator_checkpoint["model"])
        loaded = target.state_dict()
        for index in (*PERIODS, *SUBBANDS):
            for key in sub_keys(loaded, index):
                self.assertTrue(torch.equal(loaded[key], source_d[key]), key)
        for index in RESOLUTIONS:
            for key in sub_keys(loaded, index):
                self.assertFalse(torch.equal(loaded[key], source_d[key]), key)
        self.assertEqual(
            [describe_discriminator(d) for d in target.discriminators],
            discriminator_layout(now),
        )

        # The run that follows resumes from what the reset installed.
        quiet(assert_resumable, str(root), {}, arch(generator_vocoder(recipe), now))
        record = json.loads((archive / "reset.json").read_text())
        self.assertIsNone(record["generator_transfer"])
        self.assertIsNotNone(record["discriminator_transfer"])

    def test_apex_gan_to_apex_gan_plus(self):
        self._check(APEX_VERSION, APEX_GAN_PLUS)

    def test_apex_gan_plus_to_apex_gan(self):
        self._check(PLUS_VERSION, APEX_GAN)

    def test_the_same_recipe_leaves_the_discriminator_alone(self):
        root, _, source_d = self._folder(PLUS_VERSION)
        quiet(reset_training_run, root, vocoder=APEX_GAN_PLUS)
        discriminator_checkpoint = self._installed(root, "D")
        self.assertEqual(discriminator_checkpoint["disc_version"], PLUS_VERSION)
        for key, value in source_d.items():
            self.assertTrue(torch.equal(discriminator_checkpoint["model"][key], value), key)


class ExportTest(unittest.TestCase):
    """What an APEX-GAN+ run exports is an APEX-GAN model, loaded by the real inference
    code."""

    def test_the_exported_model_loads_through_voice_converter(self):
        from rvc.infer.infer import VoiceConverter

        config = stock(48000, APEX_GAN)
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "model_info.json"), "w") as f:
                json.dump({"embedder_model": "contentvec", "speakers_id": 1}, f)
            hps = HParams(
                **dict(
                    config,
                    model=dict(config["model"], spk_embed_dim=1, text_enc_hidden_dim=768),
                )
            )
            # train.py builds and exports with generator_vocoder(training_recipe).
            vocoder = generator_vocoder(APEX_GAN_PLUS)
            path = os.path.join(directory, "m_1e_1s.pth")
            quiet(
                extract_model,
                generator(vocoder).state_dict(),
                48000,
                "m",
                path,
                1,
                1,
                hps,
                None,
                vocoder,
            )
            self.assertTrue(os.path.isfile(path), "extract_model produced no file")
            cpt = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(cpt["vocoder"], APEX_GAN)

            converter = quiet(VoiceConverter)
            converter.load_model(path)
            quiet(converter.setup_network)
        self.assertIsInstance(converter.net_g.dec, APEXGANGenerator)
        device = next(converter.net_g.parameters()).device
        frames = 50
        with torch.no_grad():
            audio, _, _ = converter.net_g.infer(
                torch.randn(1, frames, 768, device=device),
                torch.full((1,), frames, dtype=torch.long, device=device),
                torch.randint(1, 255, (1, frames), device=device),
                fake_f0(1, frames).to(device),
                torch.zeros(1, dtype=torch.long, device=device),
            )
        self.assertEqual(audio.shape, (1, 1, frames * 480))
        self.assertTrue(torch.isfinite(audio).all())


if __name__ == "__main__":
    unittest.main()
