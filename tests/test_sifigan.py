"""SiFi-GAN: generator shape contracts, warm start ports, and the source loss.

Run with:  env\\python.exe -m unittest tests.test_sifigan -v

Follows tests/test_warm_start.py: real configs and real models, no mocks, and the same
build_generator / save_legacy / identity / snapshot fixtures so that the legacy
weight_g / weight_v key names are always in play.
"""

import io
import json
import math
import os
import shutil
import sys
import contextlib
import tempfile
import unittest

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rvc.lib.algorithm.generators.sifigan import (
    DEFAULT_SOURCE_SCALE_INIT,
    SiFiGANGenerator,
    dilated_factor,
    pd_indexing,
)
from rvc.lib.algorithm.generators.hifigan_nsf import HiFiGANNSFGenerator
from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.algorithm import commons
from rvc.lib.utils import checkpoint_text_enc_hidden_dim
from rvc.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from rvc.train.mel_processing import MultiScaleMelSpectrogramLoss
from rvc.train.process.extract_model import extract_model
from rvc.train.source_loss import ResidualLoss
from rvc.train.utils import HParams, describe_architecture_mismatch
from rvc.train.warm_start import (
    HIFIGAN,
    REFINEGAN,
    SIFIGAN,
    detect_vocoder,
    infer_sifigan_sample_rate,
    normalize_weight_norm_keys,
    warm_start,
)

from tests.test_warm_start import build_generator, identity, save_legacy, snapshot

SAMPLE_RATES = (32000, 40000, 48000)


def stock(sample_rate):
    with open(
        os.path.join("rvc", "configs", f"{sample_rate}.json"), encoding="utf-8"
    ) as f:
        return json.load(f)


def quiet(fn, *args, **kwargs):
    """These constructors and the warm start print a lot; the tests only want the value."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def build_synthesizer(vocoder, sample_rate=48000, dim=768, speakers=4, **kwargs):
    config = stock(sample_rate)
    model = dict(config["model"])
    model["text_enc_hidden_dim"] = dim
    model["spk_embed_dim"] = speakers
    return quiet(
        Synthesizer,
        config["data"]["filter_length"] // 2 + 1,
        config["train"]["segment_size"] // config["data"]["hop_length"],
        **model,
        use_f0=True,
        sr=sample_rate,
        vocoder=vocoder,
        **kwargs,
    )


def build_decoder(sample_rate=48000, **kwargs):
    model = stock(sample_rate)["model"]
    return SiFiGANGenerator(
        model["inter_channels"],
        model["resblock_kernel_sizes"],
        model["resblock_dilation_sizes"],
        model["upsample_rates"],
        model["upsample_initial_channel"],
        model["upsample_kernel_sizes"],
        gin_channels=model["gin_channels"],
        sr=sample_rate,
        **kwargs,
    )


def fake_f0(batch, frames, unvoiced_every=5):
    f0 = torch.randint(80, 400, (batch, frames)).float()
    f0[:, ::unvoiced_every] = 0.0
    return f0


class DilationFactorTest(unittest.TestCase):
    def test_unvoiced_frames_are_not_adapted(self):
        """f0 == 0 must land on a factor of exactly 1, never a division by zero."""
        f0 = torch.tensor([[0.0, 200.0, 0.0]])
        factors = dilated_factor(f0, 48000, 4.0)
        self.assertTrue(torch.isfinite(factors).all())
        self.assertAlmostEqual(factors[0, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(factors[0, 2].item(), 1.0, places=6)
        # voiced: (sample_rate / dense_factor) / f0
        self.assertAlmostEqual(factors[0, 1].item(), (48000 / 4.0) / 200.0, places=4)

    def test_factors_are_strictly_positive(self):
        f0 = fake_f0(2, 64)
        for dense in (0.5, 1.0, 4.0, 8.0):
            self.assertTrue((dilated_factor(f0, 48000, dense) > 0).all())

    def test_pd_indexing_stays_in_bounds(self):
        x = torch.arange(2 * 3 * 16, dtype=torch.float32).reshape(2, 3, 16)
        d = torch.full((2, 1, 16), 4.0)
        past, future = pd_indexing(x, d, 2)
        self.assertEqual(past.shape, x.shape)
        self.assertEqual(future.shape, x.shape)
        self.assertTrue(torch.isfinite(past).all() and torch.isfinite(future).all())


class SiFiGANGeneratorTest(unittest.TestCase):
    """Output length, and the per-stage dilation tensors, at every stock sample rate."""

    def test_output_length_is_frames_times_hop(self):
        for sample_rate in SAMPLE_RATES:
            config = stock(sample_rate)
            hop = config["data"]["hop_length"]
            frames = config["train"]["segment_size"] // hop
            for variant in ("rvc", "official"):
                with self.subTest(sample_rate=sample_rate, variant=variant):
                    decoder = build_decoder(sample_rate, filter_resblock=variant).eval()
                    x = torch.randn(2, config["model"]["inter_channels"], frames)
                    with torch.no_grad():
                        waveform, source = decoder(
                            x, fake_f0(2, frames), torch.randn(2, 256, 1)
                        )
                    self.assertEqual(waveform.shape, (2, 1, frames * hop))
                    self.assertEqual(source.shape, (2, 1, frames * hop))
                    self.assertTrue(torch.isfinite(waveform).all())
                    self.assertTrue(torch.isfinite(source).all())

    def test_dilation_tensors_match_each_stage_resolution(self):
        """d[i] must be frames * cumprod(upsample_rates)[i] long, as the official collater
        produces it. A mismatch would broadcast-error inside the adaptive blocks."""
        for sample_rate in SAMPLE_RATES:
            config = stock(sample_rate)
            rates = config["model"]["upsample_rates"]
            frames = config["train"]["segment_size"] // config["data"]["hop_length"]
            decoder = build_decoder(sample_rate)
            factors = decoder._dilated_factors(fake_f0(2, frames))
            expected = [frames * math.prod(rates[: i + 1]) for i in range(len(rates))]
            self.assertEqual([f.shape[-1] for f in factors], expected)
            self.assertEqual(expected[-1], frames * config["data"]["hop_length"])

    def test_forty_kilohertz_kernels_are_supported(self):
        """40k's stock kernels [16,16,4,4] are not 2 * rate, which the official
        implementation asserts on. This fork uses the HiFi-GAN padding instead."""
        config = stock(40000)
        self.assertNotEqual(
            config["model"]["upsample_kernel_sizes"],
            [2 * r for r in config["model"]["upsample_rates"]],
        )
        decoder = build_decoder(40000)
        self.assertEqual(
            [layer.kernel_size[0] for layer in decoder.fn["upsamples"]],
            config["model"]["upsample_kernel_sizes"],
        )

    def test_filter_network_matches_the_hifigan_decoder(self):
        """Every HiFi-GAN decoder tensor except noise_convs must have a same-shaped
        counterpart in a SiFi-GAN decoder. This is what the warm start port relies on."""
        for sample_rate in SAMPLE_RATES:
            with self.subTest(sample_rate=sample_rate):
                sifigan = build_decoder(sample_rate).state_dict()
                model = stock(sample_rate)["model"]
                hifigan = HiFiGANNSFGenerator(
                    model["inter_channels"],
                    model["resblock_kernel_sizes"],
                    model["resblock_dilation_sizes"],
                    model["upsample_rates"],
                    model["upsample_initial_channel"],
                    model["upsample_kernel_sizes"],
                    gin_channels=model["gin_channels"],
                    sr=sample_rate,
                ).state_dict()
                for key, value in hifigan.items():
                    if key.startswith("noise_convs."):
                        continue
                    if key.startswith("ups."):
                        target = "fn.upsamples." + key[len("ups.") :]
                    elif key.startswith("resblocks."):
                        target = "fn.blocks." + key[len("resblocks.") :]
                    elif key == "conv_post.weight":
                        target = "fn.output_conv.weight"
                    else:
                        target = key
                    self.assertIn(target, sifigan, f"{key} has no counterpart")
                    self.assertEqual(sifigan[target].shape, value.shape, key)

    def test_official_variant_has_no_second_convolution(self):
        """The metadata stamp in extract_model tells the variants apart by convs2."""
        rvc = build_decoder(48000, filter_resblock="rvc").state_dict()
        official = build_decoder(48000, filter_resblock="official").state_dict()
        self.assertTrue(any(".convs2." in k for k in rvc if k.startswith("fn.blocks.")))
        self.assertFalse(
            any(".convs2." in k for k in official if k.startswith("fn.blocks."))
        )

    def test_rejects_mismatched_per_stage_settings(self):
        with self.assertRaises(ValueError):
            build_decoder(48000, dense_factors=(0.5, 1.0))
        with self.assertRaises(ValueError):
            build_decoder(48000, filter_resblock="nonsense")


class SiFiGANSynthesizerTest(unittest.TestCase):
    """forward() gained a sixth element; the other vocoders must be unaffected."""

    def _run_forward(self, net, sample_rate=48000, dim=768):
        config = stock(sample_rate)
        frames = config["train"]["segment_size"] // config["data"]["hop_length"]
        length = frames * 3
        batch = 2
        return net(
            torch.randn(batch, length, dim),
            torch.full((batch,), length, dtype=torch.long),
            torch.randint(1, 255, (batch, length)),
            fake_f0(batch, length),
            torch.randn(batch, config["data"]["filter_length"] // 2 + 1, length),
            torch.full((batch,), length, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
        )

    def test_sifigan_returns_the_source_signal(self):
        for dim in (768, 1024):
            with self.subTest(dim=dim):
                net = build_synthesizer("SiFi-GAN", dim=dim)
                output = self._run_forward(net, dim=dim)
                self.assertEqual(len(output), 6)
                waveform, _, _, _, _, source = output
                self.assertIsNotNone(source)
                self.assertEqual(source.shape, waveform.shape)

    def test_other_vocoders_report_no_source(self):
        for vocoder in ("HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"):
            with self.subTest(vocoder=vocoder):
                net = build_synthesizer(vocoder)
                output = self._run_forward(net)
                self.assertEqual(len(output), 6)
                self.assertIsNone(output[5])

    def test_infer_returns_only_the_waveform(self):
        net = build_synthesizer("SiFi-GAN").eval()
        length = 48
        with torch.no_grad():
            audio, _, _ = net.infer(
                torch.randn(1, length, 768),
                torch.full((1,), length, dtype=torch.long),
                torch.randint(1, 255, (1, length)),
                fake_f0(1, length),
                torch.zeros(1, dtype=torch.long),
            )
        self.assertEqual(audio.dim(), 3)
        self.assertEqual(audio.shape[1], 1)

    def test_the_embedder_width_only_moves_emb_phone(self):
        narrow = build_synthesizer("SiFi-GAN", dim=768).state_dict()
        wide = build_synthesizer("SiFi-GAN", dim=1024).state_dict()
        self.assertEqual(set(narrow), set(wide))
        differing = [k for k in narrow if narrow[k].shape != wide[k].shape]
        self.assertEqual(differing, ["enc_p.emb_phone.weight"])


class SiFiGANWarmStartTest(unittest.TestCase):
    """The point of the whole exercise: inherit everything that means the same thing."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.paths = {}
        for name, vocoder in (
            ("hifigan", HIFIGAN),
            ("refinegan", REFINEGAN),
            ("sifigan", SIFIGAN),
        ):
            net = quiet(build_generator, vocoder, 48000, 768)
            path = os.path.join(cls.tmp, f"{name}.pth")
            quiet(save_legacy, net, path, identity(vocoder, 48000))
            cls.paths[name] = path

    def _warm(self, target, path):
        transfer = quiet(
            warm_start,
            target,
            path,
            "G",
            target_identity=identity(
                detect_vocoder(target.state_dict()) or SIFIGAN, 48000
            ),
        )
        self.assertEqual(transfer.errors, [])
        return transfer

    def test_detects_the_vocoder_and_recovers_the_sample_rate(self):
        net = quiet(build_generator, SIFIGAN, 48000, 768)
        self.assertEqual(detect_vocoder(net.state_dict()), SIFIGAN)
        saved = torch.load(self.paths["sifigan"], map_location="cpu", weights_only=True)
        weights = normalize_weight_norm_keys(saved["model"])
        self.assertEqual(detect_vocoder(weights), SIFIGAN)
        self.assertEqual(infer_sifigan_sample_rate(weights), 48000)

    def test_hifigan_gives_sifigan_everything_but_the_source_network(self):
        target = quiet(build_generator, SIFIGAN, 48000, 768)
        fresh = snapshot(target)
        transfer = self._warm(target, self.paths["hifigan"])
        state = target.state_dict()

        source = normalize_weight_norm_keys(
            torch.load(self.paths["hifigan"], map_location="cpu", weights_only=True)[
                "model"
            ]
        )
        for key in state:
            if not key.startswith("dec."):
                self.assertTrue(
                    torch.equal(state[key], source[key].float()),
                    f"{key} was not inherited intact",
                )
        for prefix in (
            "dec.conv_pre.",
            "dec.cond.",
            "dec.m_source.",
            "dec.fn.upsamples.",
            "dec.fn.blocks.",
            "dec.fn.output_conv.",
        ):
            self.assertTrue(
                any(k.startswith(prefix) for k in transfer.loaded),
                f"{prefix} should have been inherited",
            )
        # Only SiFi-GAN's own modules start from scratch, and they really do.
        expected_fresh = {
            k
            for k in state
            if k.startswith("dec.sn.")
            or k.startswith("dec.fn.downsamples.")
            or k == "dec.source_scales"
        }
        self.assertEqual(set(transfer.reinitialised), expected_fresh)
        for key in expected_fresh:
            self.assertTrue(torch.equal(state[key], fresh[key]), key)
        self.assertTrue(
            any(k.startswith("dec.noise_convs.") for k in transfer.dropped),
            "HiFi-GAN's noise_convs have no counterpart and must be reported as dropped",
        )

    def test_every_direction_is_registered(self):
        for source_name, target_vocoder in (
            ("hifigan", SIFIGAN),
            ("sifigan", HIFIGAN),
            ("refinegan", SIFIGAN),
            ("sifigan", REFINEGAN),
        ):
            with self.subTest(source=source_name, target=target_vocoder):
                target = quiet(build_generator, target_vocoder, 48000, 768)
                transfer = self._warm(target, self.paths[source_name])
                self.assertTrue(
                    any(k.startswith("dec.") for k in transfer.loaded),
                    "no decoder tensor was ported",
                )

    def test_a_wider_embedder_only_costs_emb_phone(self):
        target = quiet(build_generator, SIFIGAN, 48000, 1024)
        fresh = snapshot(target)
        transfer = self._warm(target, self.paths["hifigan"])
        state = target.state_dict()
        expected = (
            {"enc_p.emb_phone.weight", "dec.source_scales"}
            | {k for k in state if k.startswith("dec.sn.")}
            | {k for k in state if k.startswith("dec.fn.downsamples.")}
        )
        self.assertEqual(set(transfer.reinitialised), expected)
        for key in expected:
            self.assertTrue(torch.equal(state[key], fresh[key]), key)

    def test_the_official_variant_keeps_what_it_can(self):
        """A shape clash in the filter blocks must not cost the whole decoder."""
        target = build_synthesizer(
            "SiFi-GAN", speakers=1, sifigan_filter_resblock="official"
        )
        transfer = self._warm(target, self.paths["hifigan"])
        self.assertIn("dec.conv_pre.weight", transfer.loaded)
        self.assertTrue(any(k.startswith("dec.fn.upsamples.") for k in transfer.loaded))
        self.assertFalse(any(k.startswith("dec.fn.blocks.") for k in transfer.loaded))
        self.assertTrue(any(k.startswith("dec.resblocks.") for k in transfer.dropped))

    def test_a_different_sample_rate_refuses_the_decoder(self):
        net = quiet(build_generator, HIFIGAN, 40000, 768)
        path = os.path.join(self.tmp, "hifigan40.pth")
        quiet(save_legacy, net, path, identity(HIFIGAN, 40000))
        target = quiet(build_generator, SIFIGAN, 48000, 768)
        transfer = self._warm(target, path)
        self.assertFalse(any(k.startswith("dec.") for k in transfer.loaded))
        self.assertTrue(any("not ported" in note for note in transfer.notes))

    def test_an_unstamped_sifigan_is_not_assumed_rate_independent(self):
        """RefineGAN's shapes do not encode the sample rate, but SiFi-GAN's do."""
        net = quiet(build_generator, SIFIGAN, 48000, 768)
        path = os.path.join(self.tmp, "sifigan_unstamped.pth")
        quiet(save_legacy, net, path)
        target = quiet(build_generator, HIFIGAN, 48000, 768)
        transfer = self._warm(target, path)
        self.assertTrue(any(k.startswith("dec.ups.") for k in transfer.loaded))
        self.assertFalse(
            any("do not depend on it" in note for note in transfer.notes),
            "a SiFi-GAN decoder's shapes do depend on the sample rate",
        )


class SiFiGANResumeGuardTest(unittest.TestCase):
    def test_changing_the_filter_variant_is_refused(self):
        reason = describe_architecture_mismatch(
            {"vocoder": SIFIGAN, "sifigan_filter_resblock": "rvc"},
            {"vocoder": SIFIGAN, "sifigan_filter_resblock": "official"},
        )
        self.assertIsNotNone(reason)
        self.assertIn("filter block variant", reason)

    def test_the_same_variant_resumes(self):
        self.assertIsNone(
            describe_architecture_mismatch(
                {"vocoder": SIFIGAN, "sifigan_filter_resblock": "rvc"},
                {"vocoder": SIFIGAN, "sifigan_filter_resblock": "rvc"},
            )
        )

    def test_other_vocoders_are_unaffected(self):
        self.assertIsNone(
            describe_architecture_mismatch(
                {"vocoder": HIFIGAN}, {"vocoder": HIFIGAN, "sample_rate": 48000}
            )
        )


class SourceRegularizationLossTest(unittest.TestCase):
    def test_it_is_finite_and_trains_the_source_network(self):
        for sample_rate in SAMPLE_RATES:
            with self.subTest(sample_rate=sample_rate):
                config = stock(sample_rate)
                hop = config["data"]["hop_length"]
                frames = config["train"]["segment_size"] // hop
                length = frames * hop
                loss_fn = ResidualLoss(sample_rate=sample_rate, hop_size=hop)
                source = (torch.randn(2, 1, length) * 0.1).requires_grad_(True)
                target = torch.randn(2, 1, length) * 0.1
                value = loss_fn(source, target, fake_f0(2, frames))
                value.backward()
                self.assertTrue(torch.isfinite(value))
                self.assertTrue(torch.isfinite(source.grad).all())
                self.assertGreater(source.grad.abs().max().item(), 0.0)

    def test_f0_may_be_two_or_three_dimensional(self):
        config = stock(48000)
        hop = config["data"]["hop_length"]
        frames = config["train"]["segment_size"] // hop
        loss_fn = ResidualLoss(sample_rate=48000, hop_size=hop)
        source = torch.randn(2, 1, frames * hop) * 0.1
        target = torch.randn(2, 1, frames * hop) * 0.1
        f0 = fake_f0(2, frames)
        self.assertTrue(
            torch.allclose(
                loss_fn(source, target, f0), loss_fn(source, target, f0.unsqueeze(1))
            )
        )

    def test_out_of_range_f0_is_clamped(self):
        config = stock(48000)
        hop = config["data"]["hop_length"]
        frames = config["train"]["segment_size"] // hop
        loss_fn = ResidualLoss(sample_rate=48000, hop_size=hop)
        source = torch.randn(2, 1, frames * hop) * 0.1
        target = torch.randn(2, 1, frames * hop) * 0.1
        f0 = torch.full((2, frames), 4000.0)
        f0[:, ::3] = 1.0
        self.assertTrue(torch.isfinite(loss_fn(source, target, f0)))

    def test_the_fft_size_must_resolve_the_lowest_pitch(self):
        with self.assertRaises(ValueError):
            ResidualLoss(sample_rate=48000, hop_size=480, fft_size=1024, f0_floor=50)


class SourceScaleTest(unittest.TestCase):
    """The gain on the source network's contribution to the filter network.

    Both of these pin a bug that actually happened: `Synthesizer` takes `**kwargs`, so a
    constructor argument that is not threaded through reaches nothing and fails silently.
    Measuring a sweep over the gain then produced identical numbers at every value.
    """

    def test_the_gain_reaches_the_decoder_through_the_synthesizer(self):
        net = build_synthesizer("SiFi-GAN", sifigan_source_scale_init=0.25)
        self.assertTrue(
            torch.allclose(net.dec.source_scales, torch.full_like(net.dec.source_scales, 0.25)),
            f"Synthesizer swallowed the argument: {net.dec.source_scales.tolist()}",
        )

    def test_the_default_is_the_measured_one(self):
        net = build_synthesizer("SiFi-GAN")
        self.assertTrue(
            torch.allclose(
                net.dec.source_scales,
                torch.full_like(net.dec.source_scales, DEFAULT_SOURCE_SCALE_INIT),
            )
        )
        self.assertEqual(len(net.dec.source_scales), len(stock(48000)["model"]["upsample_rates"]))

    def test_the_gain_changes_the_output(self):
        """A parameter that exists but is never read would pass every other test here."""
        config = stock(48000)
        frames = config["train"]["segment_size"] // config["data"]["hop_length"]
        x = torch.randn(1, config["model"]["inter_channels"], frames)
        f0 = fake_f0(1, frames)
        cond = torch.randn(1, 256, 1)
        outputs = []
        for gain in (0.0, 1.0):
            torch.manual_seed(0)
            decoder = build_decoder(48000, source_scale_init=gain).eval()
            torch.manual_seed(1)
            with torch.no_grad():
                outputs.append(decoder(x, f0, cond)[0])
        self.assertFalse(
            torch.allclose(outputs[0], outputs[1]),
            "source_scales is not applied in forward",
        )

    def test_the_gain_is_learnable(self):
        net = build_synthesizer("SiFi-GAN")
        self.assertTrue(net.dec.source_scales.requires_grad)
        self.assertIn(
            "dec.source_scales", dict(net.named_parameters()), "not a registered parameter"
        )


class ExportRoundTripTest(unittest.TestCase):
    """export -> load -> infer, rebuilt the way rvc/infer/infer.py rebuilds it.

    The filter variant is not in opt["config"] (that list is positional), so it travels
    as its own key. Without it a SiFi-GAN model cannot be reconstructed at all.
    """

    def _round_trip(self, dim, variant):
        config = stock(48000)
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        with open(os.path.join(directory, "model_info.json"), "w") as f:
            json.dump(
                {
                    "total_dataset_duration": 12.3,
                    "embedder_model": "contentvec",
                    "embedder_feature_scale": 1.0,
                    "embedder_output_layer": 0,
                    "embedder_input_std_floor": 0.01,
                    "speakers_id": 3,
                },
                f,
            )
        hps = HParams(**json.loads(json.dumps(config)))
        hps.model.text_enc_hidden_dim = dim
        hps.model.spk_embed_dim = 3

        net_g = build_synthesizer(
            "SiFi-GAN", dim=dim, speakers=3, sifigan_filter_resblock=variant
        )
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
            "SiFi-GAN",
        )
        # extract_model swallows its own exceptions, so the file is the only evidence.
        self.assertTrue(os.path.isfile(path), "extract_model produced no file")
        return torch.load(path, map_location="cpu", weights_only=True)

    def test_the_filter_variant_is_stamped_from_the_weights(self):
        for variant in ("rvc", "official"):
            with self.subTest(variant=variant):
                cpt = self._round_trip(768, variant)
                self.assertEqual(cpt["vocoder"], "SiFi-GAN")
                self.assertEqual(cpt["sifigan_filter_resblock"], variant)

    def test_the_model_rebuilds_and_infers_at_both_widths(self):
        for dim in (768, 1024):
            for variant in ("rvc", "official"):
                with self.subTest(dim=dim, variant=variant):
                    cpt = self._round_trip(dim, variant)
                    self.assertEqual(cpt["text_enc_hidden_dim"], dim)
                    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
                    width = checkpoint_text_enc_hidden_dim(cpt)
                    self.assertEqual(width, dim)
                    net = quiet(
                        Synthesizer,
                        *cpt["config"],
                        use_f0=cpt.get("f0", 1),
                        text_enc_hidden_dim=width,
                        vocoder=cpt["vocoder"],
                        sifigan_filter_resblock=cpt["sifigan_filter_resblock"],
                    )
                    del net.enc_q
                    missing, unexpected = net.load_state_dict(
                        cpt["weight"], strict=False
                    )
                    self.assertEqual(missing, [])
                    self.assertEqual(
                        [k for k in unexpected if not k.startswith("enc_q.")], []
                    )
                    net.eval().float()
                    length = 60
                    with torch.no_grad():
                        audio, _, _ = net.infer(
                            torch.randn(1, length, dim),
                            torch.full((1,), length, dtype=torch.long),
                            torch.randint(1, 255, (1, length)),
                            fake_f0(1, length),
                            torch.zeros(1, dtype=torch.long),
                        )
                    hop = stock(48000)["data"]["hop_length"]
                    self.assertEqual(audio.shape, (1, 1, length * hop))
                    self.assertTrue(torch.isfinite(audio).all())


class TrainingStepTest(unittest.TestCase):
    """One real step with the losses train.py assembles for SiFi-GAN."""

    def test_a_full_step_runs_and_reaches_the_source_network(self):
        config = stock(48000)
        hop = config["data"]["hop_length"]
        segment = config["train"]["segment_size"]
        frames = segment // hop
        torch.manual_seed(0)

        net_g = build_synthesizer("SiFi-GAN", speakers=4)
        # SiFi-GAN takes disc_version v3 and the multi-scale mel loss, like RefineGAN.
        net_d = MultiPeriodDiscriminator(
            config["model"]["use_spectral_norm"], version="v3"
        )
        self.assertEqual(len(net_d.discriminators), 9)
        fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=48000)
        fn_reg_loss = ResidualLoss(sample_rate=48000, hop_size=hop)
        optim_g = torch.optim.AdamW(net_g.parameters(), 1e-4)
        optim_d = torch.optim.AdamW(net_d.parameters(), 1e-4)

        batch, length = 2, frames * 2
        pitchf = fake_f0(batch, length)
        wave = torch.randn(batch, 1, length * hop) * 0.1
        (
            y_hat,
            ids_slice,
            _,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            y_source,
        ) = net_g(
            torch.randn(batch, length, 768),
            torch.full((batch,), length, dtype=torch.long),
            torch.randint(1, 255, (batch, length)),
            pitchf,
            torch.randn(batch, config["data"]["filter_length"] // 2 + 1, length),
            torch.full((batch,), length, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
        )
        self.assertIsNotNone(y_source)
        wave = commons.slice_segments(wave, ids_slice * hop, segment, dim=3)
        self.assertEqual(y_hat.shape, wave.shape)
        self.assertEqual(y_hat.shape, (batch, 1, segment))

        y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
        loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad()
        loss_disc.backward()
        optim_d.step()

        _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
        loss_mel = fn_mel_loss(wave, y_hat) * config["train"]["c_mel"] / 3.0
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config["train"]["c_kl"]
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        pitchf_slice = commons.slice_segments(pitchf, ids_slice, frames, 2)
        loss_reg = (
            fn_reg_loss(y_source.float(), wave.float(), pitchf_slice.float())
            * config["train"]["c_reg"]
        )
        total = loss_gen + loss_fm + loss_mel + loss_kl + loss_reg
        optim_g.zero_grad()
        total.backward()
        optim_g.step()

        for name, value in (
            ("disc", loss_disc),
            ("gen", loss_gen),
            ("fm", loss_fm),
            ("mel", loss_mel),
            ("kl", loss_kl),
            ("reg", loss_reg),
            ("total", total),
        ):
            self.assertTrue(torch.isfinite(value), f"{name} is not finite")

        # The whole reason the regularisation loss exists: the source network must be
        # getting a gradient, otherwise it is unsupervised and this is not SiFi-GAN.
        source_grads = [
            p.grad
            for name, p in net_g.named_parameters()
            if name.startswith("dec.sn.") and p.grad is not None
        ]
        self.assertTrue(source_grads, "the source network received no gradient")
        self.assertGreater(max(g.abs().max().item() for g in source_grads), 0.0)


if __name__ == "__main__":
    unittest.main()
