"""SiFi-GAN+: the SiFi-GAN generator against resolution discriminators made for 48 kHz.

Run from the repository root:
    env\\python.exe -m unittest tests.test_sifigan_plus -v

Real configs and real models, no mocks except where a test must prove training was never
started, in the style of tests/test_sifigan.py. Kept in its own file so that removing
either SiFi-GAN recipe later only touches what it has to.
"""

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
    MultiPeriodDiscriminator,
    describe_discriminator,
    discriminator_layout,
)
from rvc.lib.algorithm.generators.sifigan import SiFiGANGenerator
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
    SIFIGAN_PLUS,
    discriminator_version,
    generator_vocoder,
    pretrained_dir_name,
    sample_rate_error,
    uses_multiscale_mel_loss,
)
from rvc.train.warm_start import (
    CODENAME_RINGFORMER,
    HIFIGAN,
    SIFIGAN,
    detect_vocoder,
    normalize_weight_norm_keys,
    warm_start,
)

from tests.test_codename_ringformer import build_synthesizer
from tests.test_sifigan import fake_f0, quiet, stock
from tests.test_warm_start import save_legacy, snapshot

PLUS_VERSION = "v3-48k"
SEGMENT = 17280  # 48 kHz stock segment_size
PLUS_RESOLUTIONS = ((2048, 240, 1200), (4096, 480, 2400), (1024, 100, 480))


def generator(vocoder, dim=768, speakers=1):
    return quiet(build_synthesizer, vocoder, 48000, dim, speakers)


def arch(vocoder, disc_version, **extra):
    return {"vocoder": vocoder, "sample_rate": 48000, "disc_version": disc_version, **extra}


def fresh_d(version):
    return MultiPeriodDiscriminator(False, version=version)


def sub_keys(state, index):
    return [key for key in state if key.startswith(f"discriminators.{index}.")]


class RecipeTest(unittest.TestCase):
    """What the recipe means, and that it only ever changes the discriminator."""

    def test_sifigan_plus_is_the_sifigan_generator_against_v3_48k(self):
        self.assertEqual(SIFIGAN_PLUS, "SiFi-GAN+")
        self.assertEqual(generator_vocoder(SIFIGAN_PLUS), SIFIGAN)
        self.assertEqual(discriminator_version(SIFIGAN_PLUS), PLUS_VERSION)
        # SiFi-GAN's losses, multi-scale mel included.
        self.assertTrue(uses_multiscale_mel_loss(SIFIGAN_PLUS))

    def test_the_other_recipes_are_unchanged(self):
        for recipe, version, multiscale in (
            ("HiFi-GAN", "v2", False),
            ("MRF HiFi-GAN", "v2", False),
            ("RefineGAN", "v3", True),
            ("SiFi-GAN", "v3", True),
            ("CodenameRingFormer", "codename-ringformer", False),
        ):
            with self.subTest(recipe=recipe):
                self.assertEqual(generator_vocoder(recipe), recipe)
                self.assertEqual(discriminator_version(recipe), version)
                self.assertEqual(uses_multiscale_mel_loss(recipe), multiscale)
                self.assertEqual(pretrained_dir_name(recipe), recipe.lower())

    def test_the_recipe_name_must_never_reach_synthesizer(self):
        """Why train.py and reset_run.py normalise first: Synthesizer builds a HiFi-GAN
        decoder for any name it does not know, without a word."""
        unnormalised = generator(SIFIGAN_PLUS)
        self.assertEqual(detect_vocoder(unnormalised.state_dict()), HIFIGAN)
        normalised = generator(generator_vocoder(SIFIGAN_PLUS))
        self.assertEqual(detect_vocoder(normalised.state_dict()), SIFIGAN)
        self.assertIsInstance(normalised.dec, SiFiGANGenerator)

    def test_the_pretrained_folder_has_no_plus_and_falls_back_to_hifigan(self):
        self.assertEqual(pretrained_dir_name(SIFIGAN_PLUS), "sifi-gan-plus")
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            pretraineds = Path(temp_dir) / "rvc" / "models" / "pretraineds"
            for folder in ("hifi-gan",):
                (pretraineds / folder).mkdir(parents=True)
                for name in ("f0G48k.pth", "f0D48k.pth"):
                    (pretraineds / folder / name).write_bytes(b"")
            os.chdir(temp_dir)
            try:
                g, d = quiet(pretrained_selector, SIFIGAN_PLUS, 48000)
                self.assertEqual(Path(g), Path("rvc/models/pretraineds/hifi-gan/f0G48k.pth"))
                self.assertEqual(Path(d), Path("rvc/models/pretraineds/hifi-gan/f0D48k.pth"))
                (pretraineds / "sifi-gan-plus").mkdir()
                for name in ("f0G48k.pth", "f0D48k.pth"):
                    (pretraineds / "sifi-gan-plus" / name).write_bytes(b"")
                g, d = quiet(pretrained_selector, SIFIGAN_PLUS, 48000)
                self.assertEqual(
                    Path(g), Path("rvc/models/pretraineds/sifi-gan-plus/f0G48k.pth")
                )
                self.assertEqual(
                    Path(d), Path("rvc/models/pretraineds/sifi-gan-plus/f0D48k.pth")
                )
            finally:
                os.chdir(cwd)


class SampleRateTest(unittest.TestCase):
    """48 kHz only, refused before anything touches the model folder."""

    def test_only_48k_is_accepted(self):
        self.assertIsNone(sample_rate_error(SIFIGAN_PLUS, 48000))
        self.assertIsNone(sample_rate_error(SIFIGAN_PLUS, "48000"))
        for sample_rate in (32000, 40000):
            with self.subTest(sample_rate=sample_rate):
                message = sample_rate_error(SIFIGAN_PLUS, sample_rate)
                self.assertIn("48 kHz", message)
                self.assertIn(str(sample_rate), message)
        for recipe in ("HiFi-GAN", "MRF HiFi-GAN", "RefineGAN", "SiFi-GAN", "CodenameRingFormer"):
            for sample_rate in (32000, 40000, 48000):
                self.assertIsNone(sample_rate_error(recipe, sample_rate))

    def _run_train_script(self, core, model_name, sample_rate, **kwargs):
        return core.run_train_script(
            model_name=model_name,
            save_every_epoch=1,
            save_only_latest=False,
            save_every_weights=False,
            total_epoch=1,
            sample_rate=sample_rate,
            batch_size=1,
            gpu=0,
            overtraining_detector=False,
            overtraining_threshold=50,
            pretrained=False,
            cleanup=False,
            vocoder=SIFIGAN_PLUS,
            **kwargs,
        )

    def test_core_refuses_before_touching_the_folder(self):
        import core

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            core, "logs_path", temp_dir
        ), mock.patch.object(
            core.subprocess, "run", side_effect=AssertionError("training was started")
        ):
            folder = Path(temp_dir) / "m"
            folder.mkdir()
            config = stock(40000)
            (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
            before = (folder / "config.json").read_bytes()
            # The GUI's radio says 48 kHz, but the folder was extracted at 40 kHz, and
            # that is what train.py would train at.
            with self.assertRaises(ValueError) as caught:
                self._run_train_script(
                    core, "m", 48000, reset_training=True, learning_rate=5e-5
                )
            self.assertIn("48 kHz", str(caught.exception))
            self.assertEqual((folder / "config.json").read_bytes(), before)
            self.assertEqual(sorted(os.listdir(temp_dir)), ["m"])

            # No folder config yet: the requested sample rate decides.
            with self.assertRaises(ValueError):
                self._run_train_script(core, "fresh", 40000)

    def test_train_py_refuses_a_40k_folder(self):
        """The last line of defence when train.py is started directly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir) / "logs" / "m"
            folder.mkdir(parents=True)
            (folder / "config.json").write_text(json.dumps(stock(40000)), encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
            argv = ["m", "1", "1", "", "", "0", "1", "40000", "False", "False", "False",
                    "False", "0", "False", SIFIGAN_PLUS, "False"]
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
        self.assertIn("SiFi-GAN+ trains only at 48 kHz", result.stdout)


class DiscriminatorTest(unittest.TestCase):
    def test_the_layout_is_v3s_periods_with_48k_resolutions(self):
        expected = (
            [("S",)]
            + [("P", period) for period in (2, 3, 5, 7, 11)]
            + [("R", resolution, "hann") for resolution in PLUS_RESOLUTIONS]
        )
        self.assertEqual(discriminator_layout(PLUS_VERSION), expected)
        net = fresh_d(PLUS_VERSION)
        self.assertEqual([describe_discriminator(d) for d in net.discriminators], expected)

    def test_it_is_v3_to_the_weights_and_something_else_to_the_warm_start(self):
        v3, plus = fresh_d("v3").state_dict(), fresh_d(PLUS_VERSION).state_dict()
        # Identical names and shapes: the weights alone cannot tell the two apart...
        self.assertEqual(list(v3), list(plus))
        self.assertTrue(all(v3[key].shape == plus[key].shape for key in v3))
        # ...but no resolution discriminator of one means the same as one of the other,
        # not even [2048, 240, 1200], which v3 looks at through a rectangular window.
        def resolutions(version):
            return {d for d in discriminator_layout(version) if d[0] == "R"}

        self.assertFalse(resolutions("v3") & resolutions(PLUS_VERSION))
        self.assertEqual(
            [d for d in discriminator_layout("v3") if d[0] != "R"],
            [d for d in discriminator_layout(PLUS_VERSION) if d[0] != "R"],
        )

    def test_only_the_new_layout_is_always_recorded(self):
        self.assertTrue(DISCRIMINATOR_VERSIONS[PLUS_VERSION]["always_recorded"])
        for version in ("v1", "v2", "v3", "codename-ringformer"):
            self.assertFalse(DISCRIMINATOR_VERSIONS[version].get("always_recorded", False))

    def test_the_same_seed_initialises_it_exactly_like_v3(self):
        """What makes an A/B against SiFi-GAN fair from the first step."""
        torch.manual_seed(1234)
        v3 = fresh_d("v3").state_dict()
        torch.manual_seed(1234)
        plus = fresh_d(PLUS_VERSION).state_dict()
        for key in v3:
            self.assertTrue(torch.equal(v3[key], plus[key]), key)

    def test_spectrogram_shapes_at_48k(self):
        net = fresh_d(PLUS_VERSION)
        x = torch.randn(2, 1, SEGMENT)
        shapes = [
            tuple(d.spectrogram(x).shape[1:])
            for d in net.discriminators
            if describe_discriminator(d)[0] == "R"
        ]
        # (n_fft // 2 + 1 bins, (len + 2 * pad - n_fft) // hop + 1 frames) with
        # pad = (n_fft - hop) / 2, comfortably below the 17280 samples reflect padding
        # needs.
        self.assertEqual(shapes, [(1025, 72), (2049, 36), (513, 172)])

    def _step(self, net, device, dtype=None):
        y = (torch.randn(2, 1, SEGMENT, device=device) * 0.1)
        y_hat = (torch.randn(2, 1, SEGMENT, device=device) * 0.1).requires_grad_(True)
        with torch.amp.autocast(
            device_type=device, enabled=dtype is not None, dtype=dtype or torch.float32
        ):
            y_d_r, y_d_g, fmap_r, fmap_g = net(y, y_hat)
        loss_disc, _, _ = discriminator_loss(y_d_r, y_d_g)
        loss_gen, _ = generator_loss(y_d_g)
        loss = loss_disc + loss_gen + feature_loss(fmap_r, fmap_g)
        loss.backward()
        self.assertEqual(len(y_d_r), 9)
        self.assertTrue(torch.isfinite(loss), f"loss is {loss}")
        self.assertTrue(torch.isfinite(y_hat.grad).all())
        for name, parameter in net.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_forward_and_backward_in_fp32(self):
        self._step(fresh_d(PLUS_VERSION), "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_forward_and_backward_under_half_precision_autocast(self):
        dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                self._step(fresh_d(PLUS_VERSION).cuda(), "cuda", dtype)


class DiscriminatorWarmStartTest(unittest.TestCase):
    """Sub-discriminators are matched by what they look at, and v3 and v3-48k only share
    their scale and period discriminators."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.sources = {}
        for name, version, recorded in (
            ("v2", "v2", True),
            ("v3", "v3", True),
            ("v3-unrecorded", "v3", False),
            ("plus", PLUS_VERSION, True),
            ("codename", "codename-ringformer", True),
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
        for key in sub_keys(loaded, index):
            self.assertTrue(torch.equal(loaded[key], before[key]), key)

    def test_v3_gives_the_scale_and_periods_and_no_resolution_discriminator(self):
        for source in ("v3", "v3-unrecorded"):
            with self.subTest(source=source):
                loaded, before, saved, transfer = self._warm(source, PLUS_VERSION)
                for index in range(6):  # scale + periods 2, 3, 5, 7, 11
                    self._assert_inherited(loaded, saved, index, index)
                for index in (6, 7, 8):
                    self._assert_fresh(loaded, before, index)
                dropped = set(transfer.dropped.values())
                self.assertEqual(len(dropped), 3)
                self.assertTrue(all("STFT resolution" in reason for reason in dropped))
                self.assertIn(
                    "this model has no STFT resolution [2048, 240, 1200] discriminator",
                    dropped,
                )
                self.assertIn(
                    "the pretrained discriminator has no STFT resolution "
                    "[2048, 240, 1200] (hann window) discriminator",
                    set(transfer.reinitialised.values()),
                )

    def test_v2_gives_the_scale_and_the_shared_periods(self):
        loaded, before, saved, transfer = self._warm("v2", PLUS_VERSION)
        for index in range(6):
            self._assert_inherited(loaded, saved, index, index)
        for index in (6, 7, 8):
            self._assert_fresh(loaded, before, index)
        for period in (17, 23, 37):
            self.assertTrue(
                any(f"period {period}" in reason for reason in transfer.dropped.values())
            )

    def test_codename_ringformer_gives_everything(self):
        """Its three resolution discriminators are this layout's, window included."""
        loaded, _, saved, transfer = self._warm("codename", PLUS_VERSION)
        self.assertEqual(transfer.reinitialised, {})
        for index in range(6):
            self._assert_inherited(loaded, saved, index, index)
        # codename-ringformer holds periods 17, 23 and 37 at 6..8 and the resolution
        # discriminators after them.
        for target_index, source_index in ((6, 9), (7, 10), (8, 11)):
            self._assert_inherited(loaded, saved, target_index, source_index)

    def test_back_to_v3_keeps_the_scale_and_periods(self):
        loaded, before, saved, _ = self._warm("plus", "v3")
        for index in range(6):
            self._assert_inherited(loaded, saved, index, index)
        for index in (6, 7, 8):
            self._assert_fresh(loaded, before, index)

    def test_the_same_layout_loads_whole(self):
        for source, version in (("plus", PLUS_VERSION), ("v3", "v3"), ("v3-unrecorded", "v3")):
            with self.subTest(source=source):
                loaded, _, saved, transfer = self._warm(source, version)
                self.assertEqual(transfer.reinitialised, {})
                self.assertEqual(transfer.dropped, {})
                for index in range(9):
                    self._assert_inherited(loaded, saved, index, index)

    def test_a_record_that_contradicts_the_weights_stops_the_run(self):
        path = os.path.join(self.tmp.name, "D_lying.pth")
        quiet(save_legacy, fresh_d("v2"), path, {"disc_version": PLUS_VERSION})
        with self.assertRaises(SystemExit):
            quiet(warm_start, fresh_d(PLUS_VERSION), path, "D")


class GeneratorWarmStartTest(unittest.TestCase):
    """The generator is SiFi-GAN's, so it warm starts exactly as SiFi-GAN does."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.paths = {}
        for name, vocoder, disc_version in (
            ("hifigan", HIFIGAN, "v2"),
            ("sifigan", SIFIGAN, "v3"),
            ("plus", SIFIGAN, PLUS_VERSION),
            ("codename", CODENAME_RINGFORMER, "codename-ringformer"),
        ):
            path = os.path.join(cls.tmp.name, f"G_{name}.pth")
            quiet(save_legacy, generator(vocoder), path, arch(vocoder, disc_version))
            cls.paths[name] = path

    def _warm(self, source, recipe):
        target = generator(generator_vocoder(recipe))
        before = snapshot(target)
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
        return target.state_dict(), before, saved, transfer

    def test_from_hifigan_only_sifigans_own_modules_start_fresh(self):
        state, before, _, transfer = self._warm("hifigan", SIFIGAN_PLUS)
        expected = {
            key
            for key in state
            if key.startswith(("dec.sn.", "dec.fn.downsamples."))
            or key == "dec.source_scales"
        }
        self.assertEqual(set(transfer.reinitialised), expected)
        for key in expected:
            self.assertTrue(torch.equal(state[key], before[key]), key)

    def test_from_sifigan_and_back_everything_carries_over(self):
        for source, recipe in (("sifigan", SIFIGAN_PLUS), ("plus", "SiFi-GAN")):
            with self.subTest(source=source, recipe=recipe):
                state, _, saved, transfer = self._warm(source, recipe)
                self.assertEqual(transfer.reinitialised, {})
                self.assertEqual(transfer.dropped, {})
                for key in state:
                    self.assertTrue(torch.equal(state[key], saved[key].float()), key)

    def test_from_codename_ringformer_only_the_encoders_flow_and_speakers_come(self):
        state, before, saved, transfer = self._warm("codename", SIFIGAN_PLUS)
        for key in state:
            if key.startswith("dec."):
                self.assertIn(key, transfer.reinitialised)
                self.assertTrue(torch.equal(state[key], before[key]), key)
            else:
                self.assertTrue(torch.equal(state[key], saved[key].float()), key)
        self.assertTrue(any("not ported" in note for note in transfer.notes))


class ResumeGuardTest(unittest.TestCase):
    def test_sifigan_and_sifigan_plus_refuse_each_other(self):
        for was, now in (("v3", PLUS_VERSION), (PLUS_VERSION, "v3")):
            with self.subTest(was=was, now=now):
                reason = describe_architecture_mismatch(
                    arch(SIFIGAN, was, sifigan_filter_resblock="rvc"),
                    arch(SIFIGAN, now, sifigan_filter_resblock="rvc"),
                )
                self.assertIn(f"discriminator version {was} -> {now}", reason)
        self.assertIsNone(
            describe_architecture_mismatch(arch(SIFIGAN, PLUS_VERSION), arch(SIFIGAN, PLUS_VERSION))
        )

    def test_a_saved_sifigan_checkpoint_is_refused_and_the_reset_suggested(self):
        with tempfile.TemporaryDirectory() as folder:
            quiet(
                save_legacy,
                generator(SIFIGAN),
                os.path.join(folder, "G_100.pth"),
                arch(SIFIGAN, "v3", sifigan_filter_resblock="rvc"),
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
                assert_resumable(
                    folder,
                    {},
                    arch(
                        generator_vocoder(SIFIGAN_PLUS),
                        discriminator_version(SIFIGAN_PLUS),
                        sifigan_filter_resblock="rvc",
                    ),
                )
        self.assertIn("discriminator version v3 -> v3-48k", output.getvalue())
        self.assertIn("--reset_training", output.getvalue())


class ResetTest(unittest.TestCase):
    """A reset is how an existing folder moves between SiFi-GAN and SiFi-GAN+: the
    generator is kept and only the discriminator is rebuilt."""

    def _folder(self, disc_version):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / "branch"
        root.mkdir()
        config = stock(48000)
        config["model"]["text_enc_hidden_dim"] = 768
        (root / "config.json").write_text(json.dumps(config))
        net_g, net_d = generator(SIFIGAN), fresh_d(disc_version)
        identity = arch(SIFIGAN, disc_version, sifigan_filter_resblock="rvc")
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

    def _check(self, was, recipe):
        now = discriminator_version(recipe)
        root, source_g, source_d = self._folder(was)
        archive = quiet(reset_training_run, root, vocoder=recipe)

        generator_checkpoint = self._installed(root, "G")
        self.assertEqual(generator_checkpoint["vocoder"], SIFIGAN)
        self.assertEqual(generator_checkpoint["disc_version"], now)
        weights = normalize_weight_norm_keys(generator_checkpoint["model"])
        for key, value in source_g.items():
            self.assertTrue(torch.equal(weights[key], value), key)

        discriminator_checkpoint = self._installed(root, "D")
        self.assertEqual(discriminator_checkpoint["disc_version"], now)
        target = fresh_d(now)
        before = snapshot(target)
        target.load_state_dict(discriminator_checkpoint["model"])
        loaded = target.state_dict()
        for index in range(6):
            for key in sub_keys(loaded, index):
                self.assertTrue(torch.equal(loaded[key], source_d[key]), key)
        for index in (6, 7, 8):
            for key in sub_keys(loaded, index):
                self.assertFalse(torch.equal(loaded[key], source_d[key]), key)
        self.assertEqual(
            [describe_discriminator(d) for d in target.discriminators],
            discriminator_layout(now),
        )

        # The run that follows resumes from what the reset installed.
        quiet(
            assert_resumable,
            str(root),
            {},
            arch(generator_vocoder(recipe), now, sifigan_filter_resblock="rvc"),
        )
        record = json.loads((archive / "reset.json").read_text())
        self.assertIsNone(record["generator_transfer"])
        self.assertIsNotNone(record["discriminator_transfer"])

    def test_sifigan_to_sifigan_plus(self):
        self._check("v3", SIFIGAN_PLUS)

    def test_sifigan_plus_to_sifigan(self):
        self._check(PLUS_VERSION, "SiFi-GAN")

    def test_the_same_recipe_leaves_the_discriminator_alone(self):
        root, _, source_d = self._folder(PLUS_VERSION)
        quiet(reset_training_run, root, vocoder=SIFIGAN_PLUS)
        discriminator_checkpoint = self._installed(root, "D")
        self.assertEqual(discriminator_checkpoint["disc_version"], PLUS_VERSION)
        for key, value in source_d.items():
            self.assertTrue(torch.equal(discriminator_checkpoint["model"][key], value), key)


class ExportTest(unittest.TestCase):
    """What a SiFi-GAN+ run exports is a SiFi-GAN model, loaded by the real inference code."""

    def test_the_exported_model_loads_through_voice_converter(self):
        from rvc.infer.infer import VoiceConverter

        config = stock(48000)
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "model_info.json"), "w") as f:
                json.dump({"embedder_model": "contentvec", "speakers_id": 3}, f)
            hps = HParams(**json.loads(json.dumps(config)))
            hps.model.text_enc_hidden_dim = 768
            hps.model.spk_embed_dim = 3
            # train.py builds and exports with generator_vocoder(training_recipe).
            vocoder = generator_vocoder(SIFIGAN_PLUS)
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
            self.assertEqual(cpt["vocoder"], "SiFi-GAN")
            self.assertEqual(cpt["sifigan_filter_resblock"], "rvc")

            converter = quiet(VoiceConverter)
            converter.load_model(path)
            quiet(converter.setup_network)
        self.assertIsInstance(converter.net_g.dec, SiFiGANGenerator)
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
