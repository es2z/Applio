"""MRF HiFi-GAN: shape contracts, the HiFi-GAN warm start port, export and a real step.

Run with:  env\\python.exe -m unittest tests.test_mrf_hifigan -v

Follows tests/test_sifigan.py and tests/test_warm_start.py: real configs and real models,
no mocks, and the same build_generator / save_legacy / identity / snapshot fixtures so the
legacy weight_g / weight_v key names are always in play.
"""

import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rvc.lib.algorithm import commons
from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
from rvc.lib.algorithm.generators.hifigan_mrf import HiFiGANMRFGenerator
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.tools.pretrained_selector import pretrained_selector
from rvc.lib.utils import checkpoint_text_enc_hidden_dim
from rvc.realtime.pipeline import strip_parametrizations
from rvc.train import lr_boost
from rvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.train.mel_processing import mel_spectrogram_torch
from rvc.train.process.extract_model import extract_model
from rvc.train.utils import HParams, describe_architecture_mismatch
from rvc.train.warm_start import (
    CROSS_VOCODER_DECODER_PORTS,
    HIFIGAN,
    MRF_HIFIGAN,
    detect_vocoder,
    infer_mrf_hifigan_sample_rate,
    normalize_weight_norm_keys,
    warm_start,
)

from tests.test_warm_start import build_generator, identity, save_legacy, snapshot

SAMPLE_RATES = (32000, 40000, 48000)
HARMONIC_NUM = 8

# What the port cannot carry, in either direction. MRF mixes nine harmonics where HiFi-GAN
# mixes one, so m_source.l_linear is a different function; and HiFi-GAN's conv_post has
# bias=False where MRF's has a bias.
M_SOURCE = {"dec.m_source.l_linear.weight", "dec.m_source.l_linear.bias"}
CONV_POST_BIAS = {"dec.conv_post.bias"}


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
    return HiFiGANMRFGenerator(
        in_channel=model["inter_channels"],
        upsample_initial_channel=model["upsample_initial_channel"],
        upsample_rates=model["upsample_rates"],
        upsample_kernel_sizes=model["upsample_kernel_sizes"],
        resblock_kernel_sizes=model["resblock_kernel_sizes"],
        resblock_dilations=model["resblock_dilation_sizes"],
        gin_channels=model["gin_channels"],
        sample_rate=sample_rate,
        harmonic_num=HARMONIC_NUM,
        **kwargs,
    )


def fake_f0(batch, frames, unvoiced_every=5):
    f0 = torch.randint(80, 400, (batch, frames)).float()
    f0[:, ::unvoiced_every] = 0.0
    return f0


class GeneratorShapeTest(unittest.TestCase):
    """The decoder's own arithmetic, at every stock sample rate."""

    def test_output_length_is_frames_times_hop(self):
        for sample_rate in SAMPLE_RATES:
            config = stock(sample_rate)
            hop = config["data"]["hop_length"]
            frames = config["train"]["segment_size"] // hop
            with self.subTest(sample_rate=sample_rate):
                decoder = build_decoder(sample_rate).eval()
                x = torch.randn(2, config["model"]["inter_channels"], frames)
                with torch.no_grad():
                    waveform = decoder(x, fake_f0(2, frames), torch.randn(2, 256, 1))
                self.assertEqual(waveform.shape, (2, 1, frames * hop))
                self.assertEqual(frames * hop, config["train"]["segment_size"])
                self.assertTrue(torch.isfinite(waveform).all())

    def test_the_harmonic_source_runs_at_the_sample_rate(self):
        """f0 is replicated by prod(upsample_rates) = hop_length before the sine
        generator, so the excitation is one sample per output sample."""
        for sample_rate in SAMPLE_RATES:
            config = stock(sample_rate)
            hop = config["data"]["hop_length"]
            rates = config["model"]["upsample_rates"]
            self.assertEqual(math.prod(rates), hop)
            frames = 20
            with self.subTest(sample_rate=sample_rate):
                decoder = build_decoder(sample_rate).eval()
                f0 = fake_f0(2, frames)
                with torch.no_grad():
                    upsampled = decoder.f0_upsample(f0[:, None, :]).transpose(-1, -2)
                    source, _, _ = decoder.m_source(upsampled)
                self.assertEqual(upsampled.shape, (2, frames * hop, 1))
                self.assertEqual(source.shape, (2, frames * hop, 1))
                self.assertEqual(decoder.m_source.l_sin_gen.dim, HARMONIC_NUM + 1)
                self.assertEqual(
                    tuple(decoder.m_source.l_linear.weight.shape),
                    (1, HARMONIC_NUM + 1),
                )

    def test_every_stage_shrinks_the_source_to_its_own_rate(self):
        """x + noise_conv(har_source) only works if the two agree stage by stage."""
        for sample_rate in SAMPLE_RATES:
            config = stock(sample_rate)
            rates = config["model"]["upsample_rates"]
            hop = config["data"]["hop_length"]
            frames = 20
            with self.subTest(sample_rate=sample_rate):
                decoder = build_decoder(sample_rate).eval()
                source = torch.randn(2, 1, frames * hop)
                x = torch.randn(2, config["model"]["upsample_initial_channel"], frames)
                with torch.no_grad():
                    for i, (ups, noise_conv) in enumerate(
                        zip(decoder.upsamples, decoder.noise_convs)
                    ):
                        x = ups(x)
                        shrunk = noise_conv(source)
                        self.assertEqual(x.shape, shrunk.shape)
                        self.assertEqual(
                            x.shape[-1], frames * math.prod(rates[: i + 1])
                        )
                        x = x + shrunk

    def test_realtime_can_strip_every_parametrization(self):
        """rvc/realtime/pipeline.py folds the weight norms away before inference.

        It walks named_modules() rather than calling the decoder's own
        remove_weight_norm(), which is dead code here and in every other generator in
        this repository (they apply parametrizations.weight_norm but import the legacy
        remover, which does not recognise it).
        """
        decoder = build_decoder(48000).eval()
        with torch.no_grad():
            before = decoder(
                torch.randn(1, 192, 12), fake_f0(1, 12, unvoiced_every=1000), None
            )
        strip_parametrizations(decoder)
        self.assertFalse(
            any("parametrizations" in key for key in decoder.state_dict())
        )
        with torch.no_grad():
            after = decoder(
                torch.randn(1, 192, 12), fake_f0(1, 12, unvoiced_every=1000), None
            )
        self.assertEqual(after.shape, before.shape)
        self.assertTrue(torch.isfinite(after).all())


class SynthesizerTest(unittest.TestCase):
    def test_the_decoder_returns_only_a_waveform(self):
        net = build_synthesizer("MRF HiFi-GAN")
        self.assertIsNone(net.dec_extra)

    def test_it_forwards_and_infers_at_both_embedder_widths(self):
        config = stock(48000)
        hop = config["data"]["hop_length"]
        frames = config["train"]["segment_size"] // hop
        for dim in (768, 1024):
            with self.subTest(dim=dim):
                net = build_synthesizer("MRF HiFi-GAN", dim=dim).eval()
                length = frames * 2
                with torch.no_grad():
                    output = net(
                        torch.randn(1, length, dim),
                        torch.full((1,), length, dtype=torch.long),
                        torch.randint(1, 255, (1, length)),
                        fake_f0(1, length),
                        torch.randn(
                            1, config["data"]["filter_length"] // 2 + 1, length
                        ),
                        torch.full((1,), length, dtype=torch.long),
                        torch.zeros(1, dtype=torch.long),
                    )
                self.assertEqual(len(output), 6)
                self.assertIsNone(output[-1])
                self.assertEqual(output[0].shape, (1, 1, frames * hop))
                with torch.no_grad():
                    audio, _, _ = net.infer(
                        torch.randn(1, 40, dim),
                        torch.full((1,), 40, dtype=torch.long),
                        torch.randint(1, 255, (1, 40)),
                        fake_f0(1, 40),
                        torch.zeros(1, dtype=torch.long),
                    )
                self.assertEqual(audio.shape, (1, 1, 40 * hop))
                self.assertTrue(torch.isfinite(audio).all())

    def test_it_refuses_to_train_without_pitch_guidance(self):
        config = stock(48000)
        model = dict(config["model"], spk_embed_dim=1)
        net = quiet(
            Synthesizer,
            config["data"]["filter_length"] // 2 + 1,
            config["train"]["segment_size"] // config["data"]["hop_length"],
            **model,
            use_f0=False,
            sr=48000,
            vocoder="MRF HiFi-GAN",
        )
        self.assertIsNone(net.dec)


class WarmStartTest(unittest.TestCase):
    """HiFi-GAN <-> MRF HiFi-GAN through the common migration framework."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.directory, True)
        cls.paths = {}
        for vocoder in (HIFIGAN, MRF_HIFIGAN):
            net = quiet(build_generator, vocoder, 48000)
            path = os.path.join(cls.directory, f"{vocoder.replace(' ', '_')}.pth")
            quiet(save_legacy, net, path, identity(vocoder, 48000))
            cls.paths[vocoder] = path
            del net

    def _warm(self, target, path, target_identity=None, **kwargs):
        transfer = quiet(
            warm_start,
            target,
            path,
            "G",
            target_identity=target_identity,
            verbose=False,
            **kwargs,
        )
        self.assertEqual(transfer.errors, [])
        return transfer

    def _source(self, vocoder):
        checkpoint = torch.load(
            self.paths[vocoder], map_location="cpu", weights_only=True
        )
        return normalize_weight_norm_keys(checkpoint["model"])

    def test_both_directions_are_registered(self):
        self.assertIn((HIFIGAN, MRF_HIFIGAN), CROSS_VOCODER_DECODER_PORTS)
        self.assertIn((MRF_HIFIGAN, HIFIGAN), CROSS_VOCODER_DECODER_PORTS)

    def test_the_vocoder_and_the_sample_rate_come_off_the_weights(self):
        net = quiet(build_generator, MRF_HIFIGAN, 48000)
        self.assertEqual(detect_vocoder(net.state_dict()), MRF_HIFIGAN)
        self.assertEqual(detect_vocoder(self._source(MRF_HIFIGAN)), MRF_HIFIGAN)
        for sample_rate in SAMPLE_RATES:
            with self.subTest(sample_rate=sample_rate):
                other = quiet(build_generator, MRF_HIFIGAN, sample_rate)
                self.assertEqual(
                    infer_mrf_hifigan_sample_rate(other.state_dict()), sample_rate
                )

    def test_hifigan_gives_mrf_everything_but_the_harmonic_merge(self):
        target = quiet(build_generator, MRF_HIFIGAN, 48000)
        before = snapshot(target)
        transfer = self._warm(target, self.paths[HIFIGAN], identity(MRF_HIFIGAN, 48000))
        source = self._source(HIFIGAN)
        after = target.state_dict()

        self.assertEqual(set(transfer.reinitialised), M_SOURCE | CONV_POST_BIAS)
        self.assertEqual(set(transfer.dropped), M_SOURCE)
        for key in M_SOURCE | CONV_POST_BIAS:
            self.assertTrue(torch.equal(after[key], before[key]), key)

        # Everything outside the decoder is the pretrained model's, untouched.
        for key, value in after.items():
            if not key.startswith("dec."):
                self.assertTrue(torch.equal(value, source[key]), key)

        # Name-for-name and rename-only parts really are bit-identical.
        for target_key, source_key in (
            ("dec.cond.weight", "dec.cond.weight"),
            ("dec.noise_convs.2.weight", "dec.noise_convs.2.weight"),
            (
                "dec.upsamples.1.parametrizations.weight.original1",
                "dec.ups.1.parametrizations.weight.original1",
            ),
            ("dec.upsamples.3.bias", "dec.ups.3.bias"),
            ("dec.conv_pre.bias", "dec.conv_pre.bias"),
        ):
            self.assertTrue(
                torch.equal(after[target_key], source[source_key]), target_key
            )

    def test_the_residual_blocks_land_stage_outer_kernel_inner(self):
        target = quiet(build_generator, MRF_HIFIGAN, 48000)
        self._warm(target, self.paths[HIFIGAN], identity(MRF_HIFIGAN, 48000))
        source = self._source(HIFIGAN)
        after = target.state_dict()
        kernels = len(stock(48000)["model"]["resblock_kernel_sizes"])
        leaf = "parametrizations.weight.original1"
        for stage in range(4):
            for kernel in range(kernels):
                for dilation in range(3):
                    for mrf_conv, hifigan_conv in (
                        ("conv1", "convs1"),
                        ("conv2", "convs2"),
                    ):
                        mrf = (
                            f"dec.mrfs.{stage}.{kernel}.layers.{dilation}."
                            f"{mrf_conv}.{leaf}"
                        )
                        hifigan = (
                            f"dec.resblocks.{stage * kernels + kernel}."
                            f"{hifigan_conv}.{dilation}.{leaf}"
                        )
                        self.assertTrue(torch.equal(after[mrf], source[hifigan]), mrf)

    def test_the_weight_normed_convolutions_are_converted_not_copied(self):
        target = quiet(build_generator, MRF_HIFIGAN, 48000)
        transfer = self._warm(target, self.paths[HIFIGAN], identity(MRF_HIFIGAN, 48000))
        source = self._source(HIFIGAN)
        after = target.state_dict()
        for name in ("conv_pre", "conv_post"):
            magnitude = after[f"dec.{name}.parametrizations.weight.original0"]
            direction = after[f"dec.{name}.parametrizations.weight.original1"]
            effective = magnitude * direction / direction.norm(dim=(1, 2), keepdim=True)
            self.assertTrue(
                torch.allclose(effective, source[f"dec.{name}.weight"], atol=1e-5), name
            )
            self.assertIn(
                f"dec.{name}.parametrizations.weight.original0", transfer.transformed
            )

    def test_mrf_gives_hifigan_the_same_back(self):
        target = quiet(build_generator, HIFIGAN, 48000)
        transfer = self._warm(
            target, self.paths[MRF_HIFIGAN], identity(HIFIGAN, 48000)
        )
        source = self._source(MRF_HIFIGAN)
        after = target.state_dict()
        self.assertEqual(set(transfer.reinitialised), M_SOURCE)
        self.assertEqual(set(transfer.dropped), M_SOURCE | CONV_POST_BIAS)
        for name in ("conv_pre", "conv_post"):
            magnitude = source[f"dec.{name}.parametrizations.weight.original0"]
            direction = source[f"dec.{name}.parametrizations.weight.original1"]
            effective = magnitude * direction / direction.norm(dim=(1, 2), keepdim=True)
            self.assertTrue(
                torch.allclose(effective, after[f"dec.{name}.weight"], atol=1e-5), name
            )
        self.assertTrue(
            torch.equal(
                after["dec.ups.0.parametrizations.weight.original1"],
                source["dec.upsamples.0.parametrizations.weight.original1"],
            )
        )

    def test_a_wider_embedder_only_costs_emb_phone(self):
        target = quiet(build_generator, MRF_HIFIGAN, 48000, text_enc_hidden_dim=1024)
        transfer = self._warm(target, self.paths[HIFIGAN], identity(MRF_HIFIGAN, 48000))
        self.assertEqual(
            set(transfer.reinitialised),
            M_SOURCE | CONV_POST_BIAS | {"enc_p.emb_phone.weight"},
        )
        self.assertIn("enc_p.emb_phone.weight", transfer.dropped)

    def test_a_different_sample_rate_refuses_the_decoder(self):
        target = quiet(build_generator, MRF_HIFIGAN, 40000)
        transfer = self._warm(target, self.paths[HIFIGAN], identity(MRF_HIFIGAN, 40000))
        self.assertTrue(any("not ported" in note for note in transfer.notes))
        self.assertTrue(
            any(key.startswith("dec.mrfs.") for key in transfer.reinitialised)
        )

    def test_an_unstamped_mrf_is_not_assumed_rate_independent(self):
        """Its transposed convolutions carry the rate, so nothing is assumed about it."""
        path = os.path.join(self.directory, "unstamped_mrf.pth")
        net = quiet(build_generator, MRF_HIFIGAN, 48000)
        quiet(save_legacy, net, path, None)
        del net
        target = quiet(build_generator, MRF_HIFIGAN, 48000)
        transfer = self._warm(target, path, identity(MRF_HIFIGAN, 48000))
        self.assertFalse(
            any("do not depend on it" in note for note in transfer.notes),
            transfer.notes,
        )

    def test_the_hifigan_discriminator_carries_over_whole(self):
        """MRF trains against the same v2 layout HiFi-GAN does, so nothing is lost."""
        use_spectral_norm = stock(48000)["model"]["use_spectral_norm"]
        source_d = MultiPeriodDiscriminator(use_spectral_norm, version="v2")
        path = os.path.join(self.directory, "D.pth")
        quiet(save_legacy, source_d, path, identity(HIFIGAN, 48000))
        target_d = MultiPeriodDiscriminator(use_spectral_norm, version="v2")
        transfer = quiet(warm_start, target_d, path, "D", verbose=False)
        self.assertEqual(transfer.errors, [])
        self.assertEqual(transfer.reinitialised, {})
        self.assertEqual(transfer.dropped, {})
        self.assertEqual(len(transfer.loaded), len(target_d.state_dict()))


class ResumeGuardTest(unittest.TestCase):
    def test_moving_between_hifigan_and_mrf_is_refused(self):
        for was, now in ((HIFIGAN, MRF_HIFIGAN), (MRF_HIFIGAN, HIFIGAN)):
            with self.subTest(was=was, now=now):
                reason = describe_architecture_mismatch(
                    {"vocoder": was, "sample_rate": 48000, "disc_version": "v2"},
                    {"vocoder": now, "sample_rate": 48000, "disc_version": "v2"},
                )
                self.assertIsNotNone(reason)
                self.assertIn("vocoder", reason)

    def test_the_same_vocoder_resumes(self):
        self.assertIsNone(
            describe_architecture_mismatch(
                {"vocoder": MRF_HIFIGAN, "sample_rate": 48000, "disc_version": "v2"},
                {"vocoder": MRF_HIFIGAN, "sample_rate": 48000, "disc_version": "v2"},
            )
        )

    def test_the_vocoder_is_read_off_the_weights(self):
        net = quiet(build_generator, MRF_HIFIGAN, 48000)
        reason = describe_architecture_mismatch(
            {"model": net.state_dict()}, {"vocoder": HIFIGAN}
        )
        self.assertIsNotNone(reason)
        self.assertIn(MRF_HIFIGAN, reason)


class ExportRoundTripTest(unittest.TestCase):
    """export -> load -> infer, rebuilt the way rvc/infer/infer.py rebuilds it."""

    def _round_trip(self, dim):
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

        net_g = build_synthesizer("MRF HiFi-GAN", dim=dim, speakers=3)
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
            "MRF HiFi-GAN",
        )
        # extract_model swallows its own exceptions, so the file is the only evidence.
        self.assertTrue(os.path.isfile(path), "extract_model produced no file")
        return torch.load(path, map_location="cpu", weights_only=True)

    def test_the_vocoder_is_stamped(self):
        cpt = self._round_trip(768)
        self.assertEqual(cpt["vocoder"], "MRF HiFi-GAN")
        self.assertEqual(detect_vocoder(cpt["weight"]), MRF_HIFIGAN)

    def test_the_model_rebuilds_and_infers_at_both_widths(self):
        for dim in (768, 1024):
            with self.subTest(dim=dim):
                cpt = self._round_trip(dim)
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
                )
                del net.enc_q
                missing, unexpected = net.load_state_dict(cpt["weight"], strict=False)
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
    """One real step with exactly the losses train.py assembles for HiFi-GAN."""

    def test_a_full_step_runs_against_the_v2_discriminator(self):
        config = stock(48000)
        hop = config["data"]["hop_length"]
        segment = config["train"]["segment_size"]
        frames = segment // hop
        torch.manual_seed(0)

        net_g = build_synthesizer("MRF HiFi-GAN", speakers=4)
        # MRF HiFi-GAN takes HiFi-GAN's v2 discriminator and single-scale mel loss.
        net_d = MultiPeriodDiscriminator(
            config["model"]["use_spectral_norm"], version="v2"
        )
        self.assertEqual(len(net_d.discriminators), 9)
        optim_g = torch.optim.AdamW(net_g.parameters(), 1e-4)
        optim_d = torch.optim.AdamW(net_d.parameters(), 1e-4)

        batch, length = 2, frames * 2
        wave = torch.randn(batch, 1, length * hop) * 0.1
        (
            y_hat,
            ids_slice,
            _,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            dec_extra,
        ) = net_g(
            torch.randn(batch, length, 768),
            torch.full((batch,), length, dtype=torch.long),
            torch.randint(1, 255, (batch, length)),
            fake_f0(batch, length),
            torch.randn(batch, config["data"]["filter_length"] // 2 + 1, length),
            torch.full((batch,), length, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
        )
        self.assertIsNone(dec_extra)
        wave = commons.slice_segments(wave, ids_slice * hop, segment, dim=3)
        self.assertEqual(y_hat.shape, wave.shape)
        self.assertEqual(y_hat.shape, (batch, 1, segment))

        y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
        loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad()
        loss_disc.backward()
        optim_d.step()

        def mel(waveform):
            return mel_spectrogram_torch(
                waveform.float().squeeze(1),
                config["data"]["filter_length"],
                config["data"]["n_mel_channels"],
                config["data"]["sample_rate"],
                hop,
                config["data"]["win_length"],
                config["data"]["mel_fmin"],
                config["data"]["mel_fmax"],
            )

        _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
        loss_mel = (
            torch.nn.functional.l1_loss(mel(wave), mel(y_hat))
            * config["train"]["c_mel"]
        )
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config["train"]["c_kl"]
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        total = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        total.backward()
        optim_g.step()

        for name, value in (
            ("disc", loss_disc),
            ("gen", loss_gen),
            ("fm", loss_fm),
            ("mel", loss_mel),
            ("kl", loss_kl),
        ):
            self.assertTrue(torch.isfinite(value), name)
        # The harmonic merge is the one part a warm start leaves fresh, so it has to be
        # reachable by the gradient for the Initial Generator LR Boost to mean anything.
        self.assertIsNotNone(net_g.dec.m_source.l_linear.weight.grad)
        self.assertGreater(net_g.dec.m_source.l_linear.weight.grad.abs().sum(), 0)


class LrBoostTest(unittest.TestCase):
    def test_the_boost_scales_only_the_generator_and_restores_the_saved_state(self):
        net_g = build_synthesizer("MRF HiFi-GAN", speakers=1)
        net_d = MultiPeriodDiscriminator(False, version="v2")
        optim_g = torch.optim.AdamW(net_g.parameters(), 1e-4)
        optim_d = torch.optim.AdamW(net_d.parameters(), 1e-4)
        factor = lr_boost.generator_lr_boost_factor(1, 3.0, 10)
        saved = lr_boost.scale_learning_rates(optim_g, factor)
        self.assertAlmostEqual(optim_g.param_groups[0]["lr"], 3e-4)
        self.assertAlmostEqual(optim_d.param_groups[0]["lr"], 1e-4)
        lr_boost.restore_learning_rates(optim_g, saved)
        self.assertAlmostEqual(optim_g.param_groups[0]["lr"], 1e-4)


class PretrainedSelectorTest(unittest.TestCase):
    def test_mrf_falls_back_to_the_hifigan_pretrain(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir) / "rvc" / "models" / "pretraineds" / "hifi-gan"
            folder.mkdir(parents=True)
            for name in ("f0G48k.pth", "f0D48k.pth"):
                (folder / name).write_bytes(b"")
            os.chdir(temp_dir)
            try:
                g, d = quiet(pretrained_selector, "MRF HiFi-GAN", 48000)
                self.assertEqual(
                    Path(g), Path("rvc/models/pretraineds/hifi-gan/f0G48k.pth")
                )
                self.assertEqual(
                    Path(d), Path("rvc/models/pretraineds/hifi-gan/f0D48k.pth")
                )
                self.assertEqual(
                    quiet(pretrained_selector, "MRF HiFi-GAN", 32000), ("", "")
                )
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
