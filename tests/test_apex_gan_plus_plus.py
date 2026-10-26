"""APEX-GAN++: APEX-GAN+ with anti-aliased Snakes and three CQT discriminators.

Run from the repository root:
    env\\python.exe -m unittest tests.test_apex_gan_plus_plus -v

Real configs and real models, in the style of tests/test_apex_gan_plus.py. Kept in its own
file so that removing the recipe later only touches what it has to.
"""

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rvc.lib.algorithm.alias_free import (
    DownSample1d,
    UpSample1d,
    kaiser_sinc_filter1d,
)
from rvc.lib.algorithm.discriminators import (
    DISCRIMINATOR_VERSIONS,
    describe_discriminator,
    discriminator_layout,
)
from rvc.lib.algorithm.generators.apex_gan import APEXGANGenerator, ResBlock
from rvc.lib.algorithm.generators.apex_gan_snake import Snake
from rvc.train.process.extract_model import extract_model
from rvc.train.process.model_blender import model_blender
from rvc.train.reset_run import reset_training_run
from rvc.train.utils import (
    HParams,
    assert_resumable,
    describe_architecture_mismatch,
    save_checkpoint,
)
from rvc.train.vocoder_recipe import (
    APEX_GAN_PLUS,
    APEX_GAN_PLUS_PLUS,
    discriminator_version,
    generator_options,
    generator_vocoder,
    pretrained_dir_name,
    sample_rate_error,
    uses_multiscale_mel_loss,
)
from rvc.train.warm_start import APEX_GAN, HIFIGAN, normalize_weight_norm_keys, warm_start

from tests.test_apex_gan import SEGMENT, build_synthesizer, fake_f0, fresh_d, quiet, stock
from tests.test_apex_gan_plus import sub_keys
from tests.test_warm_start import save_legacy, snapshot

PLUS_VERSION = "apex-gan-48k"
PP_VERSION = "apex-gan-48k-cqt"
CQT_DESCRIPTORS = [d for d in discriminator_layout("codename-ringformer-cqt") if d[0] == "Q"]


def generator(recipe, speakers=1):
    return build_synthesizer(
        generator_vocoder(recipe), 48000, 768, speakers, **generator_options(recipe)
    )


def arch(recipe):
    return {
        "vocoder": generator_vocoder(recipe),
        "sample_rate": 48000,
        "disc_version": discriminator_version(recipe),
        **generator_options(recipe),
    }


def index_of(layout, descriptor):
    return layout.index(descriptor)


class RecipeTest(unittest.TestCase):
    def test_it_is_the_apex_gan_generator_anti_aliased_against_the_cqt_layout(self):
        self.assertEqual(APEX_GAN_PLUS_PLUS, "APEX-GAN++")
        self.assertEqual(generator_vocoder(APEX_GAN_PLUS_PLUS), APEX_GAN)
        self.assertEqual(discriminator_version(APEX_GAN_PLUS_PLUS), PP_VERSION)
        self.assertFalse(uses_multiscale_mel_loss(APEX_GAN_PLUS_PLUS))
        self.assertEqual(generator_options(APEX_GAN_PLUS_PLUS), {"apex_gan_antialias": True})

    def test_the_other_recipes_record_the_option_off_or_not_at_all(self):
        for recipe in (APEX_GAN, APEX_GAN_PLUS):
            self.assertEqual(generator_options(recipe), {"apex_gan_antialias": False})
        for recipe in (HIFIGAN, "SiFi-GAN+", "RingFormer+"):
            self.assertEqual(generator_options(recipe), {})

    def test_48k_only_and_its_own_pretrained_folder(self):
        self.assertIsNone(sample_rate_error(APEX_GAN_PLUS_PLUS, 48000))
        self.assertIn("APEX-GAN++ trains only at 48 kHz", sample_rate_error(APEX_GAN_PLUS_PLUS, 40000))
        self.assertEqual(pretrained_dir_name(APEX_GAN_PLUS_PLUS), "apex-gan-plus-plus")

    def test_the_cli_and_the_training_tab_offer_it(self):
        source = (REPO / "core.py").read_text(encoding="utf-8")
        self.assertIn('"APEX-GAN++",', source)
        tab = (REPO / "tabs" / "train" / "train.py").read_text(encoding="utf-8")
        self.assertIn('"APEX-GAN++",', tab)
        self.assertIn("- **APEX-GAN++**:", tab)
        for language in ("en_US", "ja_JA"):
            path = REPO / "assets" / "i18n" / "languages" / f"{language}.json"
            entries = json.loads(path.read_text(encoding="utf-8"))
            key = next(k for k in entries if k.startswith("Choose the vocoder for audio"))
            self.assertIn("- **APEX-GAN++**:", key, language)
            self.assertIn("APEX-GAN++", entries[key], language)


class AliasFreeTest(unittest.TestCase):
    def test_the_filter_is_symmetric_with_unit_dc_gain(self):
        filter_ = kaiser_sinc_filter1d(0.25, 0.3, 12).flatten()
        self.assertEqual(filter_.numel(), 12)
        self.assertAlmostEqual(float(filter_.sum()), 1.0, places=6)
        self.assertTrue(torch.allclose(filter_, filter_.flip(0)))

    def test_the_rates_and_a_low_tone_survive_the_round_trip(self):
        up, down = UpSample1d(2, 12), DownSample1d(2, 12)
        t = torch.arange(2048, dtype=torch.float64)
        x = torch.sin(2 * math.pi * 0.05 * t).float().view(1, 1, -1)
        y = up(x)
        self.assertEqual(y.shape[-1], 4096)
        z = down(y)
        self.assertEqual(z.shape, x.shape)
        self.assertLess(float((z - x)[..., 64:-64].abs().max()), 2e-3)
        constant = torch.full((1, 3, 100), 0.7)
        self.assertTrue(torch.allclose(down(up(constant)), constant, atol=1e-5))

    def test_it_suppresses_the_harmonic_a_snake_folds_back(self):
        """A Snake on a 0.3 fs tone makes a 0.6 fs harmonic, which aliases to 0.4 fs."""
        n = 4096
        t = torch.arange(n, dtype=torch.float64)
        x = (2.0 * torch.sin(2 * math.pi * 0.3 * t)).float().view(1, 1, -1)
        snake = Snake(1, init=1.0)
        with torch.no_grad():
            snake.alpha.fill_(3.0)
            plain = snake(x)
            aliased = DownSample1d(2, 12)(snake(UpSample1d(2, 12)(x)))
        window = torch.hann_window(n, dtype=torch.float64)

        def alias_level(signal):
            signal = signal.flatten().double()
            spectrum = torch.fft.rfft((signal - signal.mean()) * window).abs()
            bin_ = round(0.4 * n)
            return float(spectrum[bin_ - 2 : bin_ + 3].max())

        # Measured: 48.5 plain, 7.2 anti-aliased.
        self.assertLess(alias_level(aliased), alias_level(plain) / 4)

    def test_nothing_is_saved(self):
        block = ResBlock(8, 3, (1, 3, 5), antialias=True)
        self.assertEqual(
            list(block.state_dict()), list(ResBlock(8, 3, (1, 3, 5)).state_dict())
        )
        self.assertFalse([k for k in block.state_dict() if "filter" in k])


class GeneratorTest(unittest.TestCase):
    def test_the_weights_are_apex_gans_name_for_name_and_seed_for_seed(self):
        torch.manual_seed(0)
        plain = generator(APEX_GAN_PLUS).state_dict()
        torch.manual_seed(0)
        anti = generator(APEX_GAN_PLUS_PLUS).state_dict()
        self.assertEqual(list(plain), list(anti))
        for key in plain:
            self.assertTrue(torch.equal(plain[key], anti[key]), key)

    def test_same_weights_other_function_same_length(self):
        torch.manual_seed(0)
        plain, anti = generator(APEX_GAN_PLUS).eval(), generator(APEX_GAN_PLUS_PLUS).eval()
        anti.load_state_dict(plain.state_dict())
        self.assertTrue(anti.dec.antialias)
        self.assertFalse(plain.dec.antialias)
        frames = 36
        args = (
            torch.randn(1, frames, 768),
            torch.full((1,), frames, dtype=torch.long),
            torch.randint(1, 255, (1, frames)),
            fake_f0(1, frames),
            torch.zeros(1, dtype=torch.long),
        )
        outputs = []
        for net in (plain, anti):
            torch.manual_seed(1)
            with torch.no_grad():
                outputs.append(net.infer(*args)[0])
        for audio in outputs:
            self.assertEqual(audio.shape, (1, 1, SEGMENT))
            self.assertTrue(torch.isfinite(audio).all())
        self.assertFalse(torch.allclose(outputs[0], outputs[1]))

    def test_the_phase_reference_stays_on_the_unit_circle(self):
        decoder = generator(APEX_GAN_PLUS_PLUS).dec
        raw = torch.randn(2, 4, 50, 2)
        norm = raw.norm(dim=-1, keepdim=True)
        beta = (raw[..., 0] / norm[..., 0], raw[..., 1] / norm[..., 0])
        cos_beta, sin_beta = decoder.upsampled_beta(beta)
        self.assertEqual(cos_beta.shape, (2, 4, 100))
        radius = torch.sqrt(cos_beta**2 + sin_beta**2)
        self.assertTrue(torch.allclose(radius, torch.ones_like(radius), atol=1e-3))

    def _train_step(self, device, dtype=None):
        torch.manual_seed(0)
        net = generator(APEX_GAN_PLUS_PLUS).to(device).train()
        frames = 36
        with torch.amp.autocast(
            device_type=device, enabled=dtype is not None, dtype=dtype or torch.float32
        ):
            out = net(
                torch.randn(2, frames, 768, device=device),
                torch.full((2,), frames, dtype=torch.long, device=device),
                torch.randint(1, 255, (2, frames), device=device),
                fake_f0(2, frames).to(device),
                torch.randn(2, 1025, frames, device=device),
                torch.full((2,), frames, dtype=torch.long, device=device),
                torch.zeros(2, dtype=torch.long, device=device),
            )
        audio = out[0]
        self.assertEqual(audio.shape, (2, 1, SEGMENT))
        audio.float().square().mean().backward()
        missing = [
            name
            for name, parameter in net.dec.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing, [])
        for name, parameter in net.dec.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_every_decoder_parameter_learns_in_fp32(self):
        self._train_step("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_every_decoder_parameter_learns_under_fp16_autocast(self):
        self._train_step("cuda", torch.float16)


class DiscriminatorTest(unittest.TestCase):
    def test_the_layout_is_apex_gan_48ks_plus_the_three_cqts(self):
        layout = discriminator_layout(PP_VERSION)
        plus = discriminator_layout(PLUS_VERSION)
        self.assertEqual([d for d in layout if d[0] != "Q"], plus)
        self.assertEqual([d for d in layout if d[0] == "Q"], CQT_DESCRIPTORS)
        self.assertEqual(len(layout), 15)
        net = fresh_d(PP_VERSION)
        self.assertEqual([describe_discriminator(d) for d in net.discriminators], layout)
        self.assertFalse(DISCRIMINATOR_VERSIONS[PP_VERSION].get("always_recorded", False))

    def test_the_same_seed_initialises_the_shared_ones_like_apex_gan_plus(self):
        """What makes an A/B against APEX-GAN+ fair from the first step."""
        torch.manual_seed(1234)
        plus = fresh_d(PLUS_VERSION)
        torch.manual_seed(1234)
        pp = fresh_d(PP_VERSION)
        plus_layout = discriminator_layout(PLUS_VERSION)
        pp_layout = discriminator_layout(PP_VERSION)
        plus_state, pp_state = plus.state_dict(), pp.state_dict()
        for source_index, descriptor in enumerate(plus_layout):
            target_index = index_of(pp_layout, descriptor)
            for key in sub_keys(plus_state, source_index):
                target_key = f"discriminators.{target_index}." + key.split(".", 2)[2]
                self.assertTrue(torch.equal(plus_state[key], pp_state[target_key]), key)

    def test_forward_and_backward(self):
        torch.manual_seed(0)
        net = fresh_d(PP_VERSION)
        y = torch.randn(2, 1, SEGMENT) * 0.3
        y_hat = (torch.randn(2, 1, SEGMENT) * 0.3).requires_grad_(True)
        y_d_r, y_d_g, _, _ = net(y, y_hat)
        self.assertEqual(len(y_d_r), 15)
        sum(score.mean() for score in y_d_g).backward()
        self.assertTrue(torch.isfinite(y_hat.grad).all())


class WarmStartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.d = {}
        for name, version in (
            ("plus", PLUS_VERSION),
            ("pp", PP_VERSION),
            ("ringformer-plus", "codename-ringformer-cqt"),
        ):
            net = fresh_d(version)
            path = os.path.join(cls.tmp.name, f"D_{name}.pth")
            quiet(save_legacy, net, path, {"disc_version": version})
            cls.d[name] = (path, net.state_dict(), version)
        cls.g = os.path.join(cls.tmp.name, "G_plus.pth")
        quiet(save_legacy, generator(APEX_GAN_PLUS), cls.g, arch(APEX_GAN_PLUS))

    def _warm_d(self, source, target_version):
        target = fresh_d(target_version)
        before = snapshot(target)
        transfer = quiet(warm_start, target, self.d[source][0], "D")
        self.assertEqual(transfer.errors, [])
        return target.state_dict(), before, transfer

    def _check(self, source, target_version, inherited_kinds):
        loaded, before, transfer = self._warm_d(source, target_version)
        path, saved, source_version = self.d[source]
        source_layout = discriminator_layout(source_version)
        for index, descriptor in enumerate(discriminator_layout(target_version)):
            keys = sub_keys(loaded, index)
            self.assertTrue(keys)
            if descriptor[0] in inherited_kinds and descriptor in source_layout:
                source_index = source_layout.index(descriptor)
                for key in keys:
                    source_key = f"discriminators.{source_index}." + key.split(".", 2)[2]
                    self.assertTrue(torch.equal(loaded[key], saved[source_key]), key)
            else:
                for key in keys:
                    self.assertTrue(torch.equal(loaded[key], before[key]), key)
        return transfer

    def test_from_apex_gan_plus_only_the_cqts_start_fresh(self):
        transfer = self._check("plus", PP_VERSION, {"P", "R", "B"})
        self.assertEqual(transfer.dropped, {})
        self.assertEqual(len(set(transfer.reinitialised.values())), 3)

    def test_back_to_apex_gan_plus_drops_only_the_cqts(self):
        transfer = self._check("pp", PLUS_VERSION, {"P", "R", "B"})
        self.assertEqual(transfer.reinitialised, {})
        self.assertEqual(len(set(transfer.dropped.values())), 3)

    def test_from_ringformer_plus_even_the_cqts_come(self):
        """Same CQTs at the same rate, so a RingFormer+ D gives them too."""
        transfer = self._check("ringformer-plus", PP_VERSION, {"P", "R", "Q"})
        # Only the sub-band discriminators, which RingFormer+ has none of, start fresh.
        self.assertEqual(len(set(transfer.reinitialised.values())), 4)

    def test_the_generator_comes_whole_from_apex_gan_plus(self):
        target = generator(APEX_GAN_PLUS_PLUS)
        transfer = quiet(
            warm_start, target, self.g, "G", target_identity=arch(APEX_GAN_PLUS_PLUS)
        )
        self.assertEqual(transfer.errors, [])
        self.assertEqual(transfer.reinitialised, {})
        self.assertEqual(transfer.dropped, {})
        saved = normalize_weight_norm_keys(
            torch.load(self.g, map_location="cpu", weights_only=True)["model"]
        )
        for key, value in target.state_dict().items():
            self.assertTrue(torch.equal(value, saved[key].float()), key)


class ResumeGuardTest(unittest.TestCase):
    def test_apex_gan_plus_and_plus_plus_refuse_each_other(self):
        reason = describe_architecture_mismatch(arch(APEX_GAN_PLUS), arch(APEX_GAN_PLUS_PLUS))
        self.assertIn(f"discriminator version {PLUS_VERSION} -> {PP_VERSION}", reason)
        self.assertIn("generator option apex_gan_antialias False -> True", reason)
        self.assertNotIn("vocoder", reason)
        self.assertIsNone(
            describe_architecture_mismatch(arch(APEX_GAN_PLUS_PLUS), arch(APEX_GAN_PLUS_PLUS))
        )

    def test_a_checkpoint_from_before_the_option_counts_as_off(self):
        old = {"vocoder": APEX_GAN, "sample_rate": 48000, "disc_version": PLUS_VERSION}
        self.assertIsNone(describe_architecture_mismatch(old, arch(APEX_GAN_PLUS)))
        self.assertIn(
            "apex_gan_antialias False -> True",
            describe_architecture_mismatch(old, arch(APEX_GAN_PLUS_PLUS)),
        )

    def test_a_saved_apex_gan_plus_checkpoint_is_refused_and_the_reset_suggested(self):
        with tempfile.TemporaryDirectory() as folder:
            quiet(
                save_legacy,
                generator(APEX_GAN_PLUS),
                os.path.join(folder, "G_100.pth"),
                {"vocoder": APEX_GAN, "sample_rate": 48000, "disc_version": PLUS_VERSION},
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
                assert_resumable(folder, {}, arch(APEX_GAN_PLUS_PLUS))
        self.assertIn("APEX-GAN++", output.getvalue())
        self.assertIn("--reset_training", output.getvalue())


class ResetTest(unittest.TestCase):
    """A reset moves an APEX-GAN+ folder to APEX-GAN++ and back: G's weights are kept,
    its record changes, and D is rebuilt."""

    def _folder(self, recipe):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / "branch"
        root.mkdir()
        config = stock(48000, APEX_GAN)
        config["model"]["text_enc_hidden_dim"] = 768
        (root / "config.json").write_text(json.dumps(config))
        net_g = generator(recipe, speakers=config["model"]["spk_embed_dim"])
        net_d = fresh_d(discriminator_version(recipe))
        for tag, net in (("G", net_g), ("D", net_d)):
            quiet(
                save_checkpoint,
                net,
                torch.optim.AdamW(net.parameters(), lr=1e-4),
                1e-4,
                840,
                str(root / f"{tag}_2333333.pth"),
                torch.amp.GradScaler(enabled=False),
                architecture_identity=arch(recipe),
            )
        return root, snapshot(net_g)

    def _check(self, was, now):
        root, source_g = self._folder(was)
        archive = quiet(reset_training_run, root, vocoder=now)
        g = torch.load(root / "G_0.pth", map_location="cpu", weights_only=True)
        self.assertEqual(g["vocoder"], APEX_GAN)
        self.assertEqual(g["disc_version"], discriminator_version(now))
        self.assertEqual(g["apex_gan_antialias"], generator_options(now)["apex_gan_antialias"])
        weights = normalize_weight_norm_keys(g["model"])
        for key, value in source_g.items():
            self.assertTrue(torch.equal(weights[key], value), key)
        d = torch.load(root / "D_0.pth", map_location="cpu", weights_only=True)
        self.assertEqual(d["disc_version"], discriminator_version(now))
        quiet(assert_resumable, str(root), {}, arch(now))
        record = json.loads((archive / "reset.json").read_text())
        self.assertIn("apex_gan_antialias", record["generator_options_transfer"])

    def test_apex_gan_plus_to_plus_plus(self):
        self._check(APEX_GAN_PLUS, APEX_GAN_PLUS_PLUS)

    def test_apex_gan_plus_plus_back_to_plus(self):
        self._check(APEX_GAN_PLUS_PLUS, APEX_GAN_PLUS)


class ExportTest(unittest.TestCase):
    def _export(self, directory, recipe):
        config = stock(48000, APEX_GAN)
        with open(os.path.join(directory, "model_info.json"), "w") as f:
            json.dump({"embedder_model": "contentvec", "speakers_id": 1}, f)
        hps = HParams(
            **dict(config, model=dict(config["model"], spk_embed_dim=1, text_enc_hidden_dim=768))
        )
        path = os.path.join(directory, f"{recipe.replace('+', 'p')}_1e_1s.pth")
        quiet(
            extract_model,
            generator(recipe).state_dict(),
            48000,
            "m",
            path,
            1,
            1,
            hps,
            None,
            generator_vocoder(recipe),
            generator_options=generator_options(recipe),
        )
        self.assertTrue(os.path.isfile(path), "extract_model produced no file")
        return path

    def test_the_exported_model_is_rebuilt_anti_aliased(self):
        from rvc.infer.infer import VoiceConverter

        with tempfile.TemporaryDirectory() as directory:
            path = self._export(directory, APEX_GAN_PLUS_PLUS)
            cpt = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(cpt["vocoder"], APEX_GAN)
            self.assertIs(cpt["apex_gan_antialias"], True)
            converter = quiet(VoiceConverter)
            converter.load_model(path)
            quiet(converter.setup_network)
        self.assertIsInstance(converter.net_g.dec, APEXGANGenerator)
        self.assertTrue(converter.net_g.dec.antialias)

    def test_an_apex_gan_plus_model_is_not(self):
        from rvc.infer.infer import VoiceConverter

        with tempfile.TemporaryDirectory() as directory:
            path = self._export(directory, APEX_GAN_PLUS)
            self.assertIs(
                torch.load(path, map_location="cpu", weights_only=True)["apex_gan_antialias"],
                False,
            )
            converter = quiet(VoiceConverter)
            converter.load_model(path)
            quiet(converter.setup_network)
        self.assertFalse(converter.net_g.dec.antialias)

    def test_the_blender_refuses_to_mix_them(self):
        with tempfile.TemporaryDirectory() as directory:
            plus = self._export(directory, APEX_GAN_PLUS)
            pp = self._export(directory, APEX_GAN_PLUS_PLUS)
            result = quiet(model_blender, "mixed", plus, pp, 0.5)
        self.assertEqual(result, "One model is APEX-GAN++ and the other is not.")


if __name__ == "__main__":
    unittest.main()
