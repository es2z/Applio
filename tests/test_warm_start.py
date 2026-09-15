import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import torch

from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.tools.pretrained_selector import pretrained_selector
from rvc.train import lr_boost
from rvc.train.extract.preparing_files import (
    apply_generator_lr_boost_settings,
    read_generator_lr_boost_settings,
)
from rvc.train.utils import HParams, assert_resumable, load_pretrained, save_checkpoint
from rvc.train.warm_start import (
    detect_vocoder,
    infer_hifigan_sample_rate,
    normalize_weight_norm_keys,
)

REPO = Path(__file__).resolve().parents[1]
STOCK_PRETRAIN_G = REPO / "rvc" / "models" / "pretraineds" / "hifi-gan" / "f0G48k.pth"


def build_generator(vocoder, sample_rate=48000, text_enc_hidden_dim=768, speakers=1):
    with open(REPO / "rvc" / "configs" / f"{sample_rate}.json", encoding="utf-8") as f:
        config = json.load(f)
    model = dict(
        config["model"], text_enc_hidden_dim=text_enc_hidden_dim, spk_embed_dim=speakers
    )
    return Synthesizer(
        config["data"]["filter_length"] // 2 + 1,
        config["train"]["segment_size"] // config["data"]["hop_length"],
        **model,
        use_f0=True,
        sr=sample_rate,
        vocoder=vocoder,
    )


def save_legacy(net, path, identity=None):
    """Save the way train.py does, which writes weight_g / weight_v names."""
    save_checkpoint(
        net,
        torch.optim.AdamW(net.parameters()),
        1e-4,
        1,
        str(path),
        torch.amp.GradScaler(enabled=False),
        architecture_identity=identity,
    )


def identity(vocoder, sample_rate=48000):
    return {"vocoder": vocoder, "sample_rate": sample_rate}


def snapshot(net):
    return {key: value.detach().clone() for key, value in net.state_dict().items()}


class LegacyWeightNormNamesTest(unittest.TestCase):
    """The bug: every weight-normed layer of a pretrain was silently skipped."""

    def test_a_saved_generator_loads_whole(self):
        source, target = build_generator("HiFi-GAN"), build_generator("HiFi-GAN")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "G.pth"
            save_legacy(source, path)
            saved = torch.load(path, weights_only=True)["model"]
            self.assertTrue(any(key.endswith(".weight_g") for key in saved))
            transfer = load_pretrained(
                target, path, "G", verbose=False, target_identity=identity("HiFi-GAN")
            )
        self.assertEqual(transfer.reinitialised, {})
        loaded = target.state_dict()
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(loaded[key], value), key)

    def test_a_saved_discriminator_loads_whole(self):
        source, target = MultiPeriodDiscriminator(), MultiPeriodDiscriminator()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "D.pth"
            save_legacy(source, path)
            load_pretrained(target, path, "D", verbose=False)
        loaded = target.state_dict()
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(loaded[key], value), key)

    @unittest.skipUnless(STOCK_PRETRAIN_G.is_file(), "stock 48k pretrain not downloaded")
    def test_the_stock_pretrain_loads_everything_but_the_embedder_projection(self):
        # train.py sizes the speaker embedding from the pretrain, 109 for the stock one.
        target = build_generator("HiFi-GAN", text_enc_hidden_dim=1024, speakers=109)
        transfer = load_pretrained(
            target, STOCK_PRETRAIN_G, "G", verbose=False, target_identity=identity("HiFi-GAN")
        )
        self.assertEqual(list(transfer.reinitialised), ["enc_p.emb_phone.weight"])
        source = normalize_weight_norm_keys(
            torch.load(STOCK_PRETRAIN_G, map_location="cpu", weights_only=True)["model"]
        )
        loaded = target.state_dict()
        for key in transfer.loaded:
            self.assertTrue(torch.equal(loaded[key], source[key].float()), key)

    def test_the_sample_rate_is_read_off_a_legacy_hifigan(self):
        for sample_rate in (32000, 40000, 48000):
            net = build_generator("HiFi-GAN", sample_rate=sample_rate)
            self.assertEqual(infer_hifigan_sample_rate(net.state_dict()), sample_rate)


class _TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_p = torch.nn.Module()
        self.enc_p.emb_phone = torch.nn.Linear(4, 4)
        self.dec = torch.nn.Linear(4, 4)


class NothingIsSkippedSilentlyTest(unittest.TestCase):
    def test_a_source_tensor_with_no_place_in_the_model_stops_the_run(self):
        state = _TinyNet().state_dict()
        state["flow.extra.weight"] = torch.zeros(1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "G.pth")
            torch.save({"model": state}, path)
            with self.assertRaises(SystemExit):
                load_pretrained(_TinyNet(), path, "G", verbose=False)

    def test_a_missing_tensor_stops_the_run(self):
        state = _TinyNet().state_dict()
        del state["dec.bias"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "G.pth")
            torch.save({"model": state}, path)
            with self.assertRaises(SystemExit):
                load_pretrained(_TinyNet(), path, "G", verbose=False)

    def test_an_exported_inference_model_is_refused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "model_100e_1000s.pth")
            torch.save({"weight": _TinyNet().state_dict()}, path)
            with self.assertRaises(SystemExit):
                load_pretrained(_TinyNet(), path, "G", verbose=False)


class CrossVocoderGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hifigan = build_generator("HiFi-GAN")
        cls.refinegan = build_generator("RefineGAN")
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.hifigan_path = Path(cls.temp_dir.name) / "G_hifigan.pth"
        cls.refinegan_path = Path(cls.temp_dir.name) / "G_refinegan.pth"
        cls.legacy_refinegan_path = Path(cls.temp_dir.name) / "G_refinegan_legacy.pth"
        save_legacy(cls.hifigan, cls.hifigan_path)
        save_legacy(cls.refinegan, cls.refinegan_path, identity("RefineGAN"))
        save_legacy(cls.refinegan, cls.legacy_refinegan_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def assert_shared_parts_inherited(self, source, target):
        loaded = target.state_dict()
        for key, value in source.state_dict().items():
            if not key.startswith("dec."):
                self.assertTrue(torch.equal(loaded[key], value), key)

    def test_vocoders_are_recognised_from_their_keys(self):
        self.assertEqual(detect_vocoder(self.hifigan.state_dict()), "HiFi-GAN")
        self.assertEqual(detect_vocoder(self.refinegan.state_dict()), "RefineGAN")
        legacy = torch.load(self.hifigan_path, weights_only=True)["model"]
        self.assertEqual(detect_vocoder(legacy), "HiFi-GAN")

    def test_hifigan_to_refinegan_ports_residual_blocks_and_conv_post(self):
        target = build_generator("RefineGAN")
        before = snapshot(target)
        transfer = load_pretrained(
            target, self.hifigan_path, "G", verbose=False, target_identity=identity("RefineGAN")
        )
        self.assert_shared_parts_inherited(self.hifigan, target)
        loaded, source = target.state_dict(), self.hifigan.state_dict()
        for stage in range(4):
            for kernel in range(3):
                for name in ("convs1", "convs2"):
                    for j in range(3):
                        for leaf in (
                            "bias",
                            "parametrizations.weight.original0",
                            "parametrizations.weight.original1",
                        ):
                            self.assertTrue(
                                torch.equal(
                                    loaded[
                                        f"dec.upsample_conv_blocks.{stage}.blocks.{kernel}.1.{name}.{j}.{leaf}"
                                    ],
                                    source[
                                        f"dec.resblocks.{stage * 3 + kernel}.{name}.{j}.{leaf}"
                                    ],
                                )
                            )
        self.assertTrue(
            torch.allclose(
                target.dec.conv_post.weight, self.hifigan.dec.conv_post.weight, atol=1e-6
            )
        )
        for fresh in (
            "dec.mel_conv.parametrizations.weight.original1",
            "dec.cond.weight",
            "dec.upsample_conv_blocks.0.input_conv.weight",
        ):
            self.assertTrue(torch.equal(loaded[fresh], before[fresh]), fresh)
            self.assertIn(fresh, transfer.reinitialised)
        self.assertIn("dec.conv_pre.weight", transfer.dropped)

    def test_refinegan_to_hifigan_is_the_inverse(self):
        target = build_generator("HiFi-GAN")
        before = snapshot(target)
        load_pretrained(
            target, self.refinegan_path, "G", verbose=False, target_identity=identity("HiFi-GAN")
        )
        self.assert_shared_parts_inherited(self.refinegan, target)
        loaded, source = target.state_dict(), self.refinegan.state_dict()
        self.assertTrue(
            torch.equal(
                loaded["dec.resblocks.11.convs2.2.parametrizations.weight.original1"],
                source[
                    "dec.upsample_conv_blocks.3.blocks.2.1.convs2.2.parametrizations.weight.original1"
                ],
            )
        )
        self.assertTrue(
            torch.allclose(
                loaded["dec.conv_post.weight"], self.refinegan.dec.conv_post.weight, atol=1e-6
            )
        )
        self.assertTrue(torch.equal(loaded["dec.conv_pre.weight"], before["dec.conv_pre.weight"]))

    def test_no_port_when_the_source_sample_rate_is_unknown(self):
        # A RefineGAN decoder's shapes do not depend on the sample rate, so a checkpoint
        # that did not record one cannot prove it matches.
        target = build_generator("HiFi-GAN")
        before = snapshot(target)
        transfer = load_pretrained(
            target, self.legacy_refinegan_path, "G", verbose=False, target_identity=identity("HiFi-GAN")
        )
        self.assert_shared_parts_inherited(self.refinegan, target)
        loaded = target.state_dict()
        for key in loaded:
            if key.startswith("dec."):
                self.assertTrue(torch.equal(loaded[key], before[key]), key)
        self.assertTrue(any("not ported" in note for note in transfer.notes))

    def test_no_port_across_sample_rates(self):
        target = build_generator("RefineGAN")
        transfer = load_pretrained(
            target, self.hifigan_path, "G", verbose=False, target_identity=identity("RefineGAN", 40000)
        )
        self.assertFalse(any(key.startswith("dec.") for key in transfer.loaded))

    def test_the_same_vocoder_at_a_different_recorded_sample_rate_stops_the_run(self):
        with self.assertRaises(SystemExit):
            load_pretrained(
                build_generator("RefineGAN"),
                self.refinegan_path,
                "G",
                verbose=False,
                target_identity=identity("RefineGAN", 40000),
            )

    def test_a_wider_embedder_still_warm_starts_across_vocoders(self):
        target = build_generator("RefineGAN", text_enc_hidden_dim=1024)
        transfer = load_pretrained(
            target, self.hifigan_path, "G", verbose=False, target_identity=identity("RefineGAN")
        )
        self.assertIn("enc_p.emb_phone.weight", transfer.reinitialised)
        self.assertIn("flow.flows.0.enc.in_layers.0.parametrizations.weight.original1", transfer.loaded)


class DiscriminatorLayoutTest(unittest.TestCase):
    def test_v2_to_v3_matches_by_period_and_leaves_resolution_discriminators_new(self):
        source, target = MultiPeriodDiscriminator(version="v2"), MultiPeriodDiscriminator(version="v3")
        before = snapshot(target)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "D.pth"
            save_legacy(source, path)
            transfer = load_pretrained(target, path, "D", verbose=False)
        loaded, saved = target.state_dict(), source.state_dict()
        for index in range(6):  # scale + periods 2, 3, 5, 7, 11 sit at the same index
            for key in saved:
                if key.startswith(f"discriminators.{index}."):
                    self.assertTrue(torch.equal(loaded[key], saved[key]), key)
        for index in (6, 7, 8):
            for key in loaded:
                if key.startswith(f"discriminators.{index}."):
                    self.assertTrue(torch.equal(loaded[key], before[key]), key)
        self.assertTrue(
            any("period 37" in reason for reason in transfer.dropped.values())
        )

    def test_an_unknown_layout_stops_the_run(self):
        state = {"discriminators.0.convs.0.weight_v": torch.zeros(32, 1, 5, 1)}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "D.pth")
            torch.save({"model": state}, path)
            with self.assertRaises(SystemExit):
                load_pretrained(MultiPeriodDiscriminator(), path, "D", verbose=False)


class ResumeArchitectureGuardTest(unittest.TestCase):
    def _experiment(self, temp_dir, model, recorded=None):
        checkpoint = {"model": model, "iteration": 1}
        checkpoint.update(recorded or {})
        torch.save(checkpoint, os.path.join(temp_dir, "G_100.pth"))

    LEGACY_HIFIGAN = {
        "dec.ups.0.weight_v": torch.zeros(1),
        "dec.resblocks.0.convs1.0.weight_v": torch.zeros(1),
    }

    def test_resuming_into_a_different_vocoder_is_refused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._experiment(temp_dir, self.LEGACY_HIFIGAN)
            with self.assertRaises(SystemExit):
                assert_resumable(temp_dir, {}, {"vocoder": "RefineGAN", "disc_version": "v3"})

    def test_an_unstamped_hifigan_checkpoint_resumes_as_hifigan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._experiment(temp_dir, self.LEGACY_HIFIGAN)
            assert_resumable(
                temp_dir, {}, {"vocoder": "HiFi-GAN", "sample_rate": 48000, "disc_version": "v2"}
            )

    def test_a_recorded_discriminator_version_must_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._experiment(temp_dir, self.LEGACY_HIFIGAN, {"disc_version": "v3"})
            with self.assertRaises(SystemExit):
                assert_resumable(temp_dir, {}, {"vocoder": "HiFi-GAN", "disc_version": "v2"})


class GeneratorLrBoostTest(unittest.TestCase):
    GAMMA = 0.999875

    def _optimizer(self, lr=1e-4):
        return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=lr)

    def test_the_factor_covers_exactly_the_boost_epochs(self):
        factor = lr_boost.generator_lr_boost_factor
        self.assertEqual(factor(1, 3.0, 2), 3.0)
        self.assertEqual(factor(2, 3.0, 2), 3.0)
        self.assertEqual(factor(3, 3.0, 2), 1.0)
        self.assertEqual(factor(1, 3.0, 0), 1.0)

    def test_off_never_touches_the_optimizer(self):
        optimizer = self._optimizer()
        self.assertIsNone(lr_boost.scale_learning_rates(optimizer, 1.0))
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)

    def _run(self, epochs_to_run, boost_epochs, start_state=None, start_epoch=1):
        """Mirror train.py: scheduler built with last_epoch=epoch_str-2, boost per epoch."""
        optimizer = self._optimizer()
        if start_state is not None:
            optimizer.load_state_dict(start_state)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.GAMMA, last_epoch=start_epoch - 2
        )
        effective = {}
        for epoch in range(start_epoch, start_epoch + epochs_to_run):
            factor = lr_boost.generator_lr_boost_factor(epoch, 3.0, boost_epochs)
            saved = lr_boost.scale_learning_rates(optimizer, factor)
            effective[epoch] = optimizer.param_groups[0]["lr"]
            lr_boost.restore_learning_rates(optimizer, saved)
            state = optimizer.state_dict()  # what save_checkpoint would store
            scheduler.step()
        return effective, state

    def test_the_schedule_and_saved_state_are_untouched_by_the_boost(self):
        boosted, boosted_state = self._run(6, boost_epochs=3)
        plain, plain_state = self._run(6, boost_epochs=0)
        self.assertEqual(boosted_state["param_groups"], plain_state["param_groups"])
        for epoch in range(1, 7):
            expected = plain[epoch] * (3.0 if epoch <= 3 else 1.0)
            self.assertEqual(boosted[epoch], expected)

    def test_a_resume_continues_the_boost_period(self):
        # Compared against an unboosted run resumed the same way, because the resume
        # itself does not reproduce an uninterrupted ExponentialLR chain exactly.
        _, boosted_state = self._run(2, boost_epochs=4)
        _, plain_state = self._run(2, boost_epochs=0)
        self.assertEqual(boosted_state["param_groups"], plain_state["param_groups"])
        boosted, _ = self._run(4, 4, start_state=boosted_state, start_epoch=3)
        plain, _ = self._run(4, 0, start_state=plain_state, start_epoch=3)
        for epoch in range(3, 7):
            self.assertEqual(boosted[epoch], plain[epoch] * (3.0 if epoch <= 4 else 1.0))

    def test_invalid_settings_are_rejected(self):
        for multiplier, epochs in ((0, 1), (-1, 1), (math.nan, 1), (3.0, 1.5), (3.0, -1)):
            with self.assertRaises(ValueError):
                lr_boost.validate_generator_lr_boost(multiplier, epochs)

    def test_train_reads_the_config_section(self):
        self.assertEqual(lr_boost.read_generator_lr_boost(HParams(learning_rate=1e-4)), (3.0, 0))
        self.assertEqual(
            lr_boost.read_generator_lr_boost(
                HParams(g_lr_boost_multiplier=2.5, g_lr_boost_epochs=7)
            ),
            (2.5, 7),
        )


class GeneratorLrBoostSettingsTest(unittest.TestCase):
    def _config(self, temp_dir, train=None):
        path = Path(temp_dir) / "config.json"
        path.write_text(json.dumps({"train": train or {"learning_rate": 1e-4}}), encoding="utf-8")
        return path

    def test_unticked_on_a_config_without_a_boost_changes_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._config(temp_dir)
            before = path.read_bytes()
            apply_generator_lr_boost_settings(temp_dir, enabled=False, multiplier=3.0, epochs=10)
            self.assertEqual(path.read_bytes(), before)

    def test_ticking_writes_and_unticking_switches_off(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._config(temp_dir)
            apply_generator_lr_boost_settings(temp_dir, enabled=True, multiplier=2.0, epochs=5)
            self.assertEqual(
                read_generator_lr_boost_settings(temp_dir),
                {"enabled": True, "multiplier": 2.0, "epochs": 5},
            )
            apply_generator_lr_boost_settings(temp_dir, enabled=False, multiplier=2.0, epochs=5)
            train = json.loads(path.read_text(encoding="utf-8"))["train"]
            self.assertEqual(train["g_lr_boost_epochs"], 0)
            self.assertFalse(read_generator_lr_boost_settings(temp_dir)["enabled"])

    def test_enabled_with_zero_epochs_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._config(temp_dir)
            with self.assertRaises(ValueError):
                apply_generator_lr_boost_settings(temp_dir, enabled=True, multiplier=3.0, epochs=0)

    def test_the_cli_writes_only_what_it_was_given(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._config(temp_dir, {"g_lr_boost_multiplier": 2.0, "g_lr_boost_epochs": 4})
            apply_generator_lr_boost_settings(temp_dir, epochs=8)
            train = json.loads(path.read_text(encoding="utf-8"))["train"]
            self.assertEqual((train["g_lr_boost_multiplier"], train["g_lr_boost_epochs"]), (2.0, 8))


class PretrainedSelectorTest(unittest.TestCase):
    def test_refinegan_falls_back_to_the_hifigan_pretrain(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir) / "rvc" / "models" / "pretraineds" / "hifi-gan"
            folder.mkdir(parents=True)
            for name in ("f0G48k.pth", "f0D48k.pth"):
                (folder / name).write_bytes(b"")
            os.chdir(temp_dir)
            try:
                g, d = pretrained_selector("RefineGAN", 48000)
                self.assertEqual(Path(g), Path("rvc/models/pretraineds/hifi-gan/f0G48k.pth"))
                self.assertEqual(Path(d), Path("rvc/models/pretraineds/hifi-gan/f0D48k.pth"))
                self.assertEqual(pretrained_selector("RefineGAN", 32000), ("", ""))
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
