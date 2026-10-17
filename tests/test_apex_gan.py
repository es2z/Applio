"""APEX-GAN: shapes, the phase-conditioned Snake, warm start, export and a training step.

Run from the repository root:
    env\\python.exe -m unittest tests.test_apex_gan -v

Real configs, real modules, no mocks except where a module-level switch is flipped, in the
style of tests/test_codename_ringformer.py.
"""

import argparse
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
from unittest import mock

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rvc.lib.algorithm import commons
from rvc.lib.algorithm.discriminators import (
    DISCRIMINATOR_VERSIONS,
    MultiPeriodDiscriminator,
    describe_discriminator,
    discriminator_layout,
)
from rvc.lib.algorithm.generators.apex_gan import (
    APEX_GAN_ISTFT,
    APEXGANGenerator,
    ExcitationSynthesizer,
    PhaseProjector,
    default_istft_settings,
)
from rvc.lib.algorithm.generators.apex_gan_snake import Snake
from rvc.lib.algorithm.subband_discriminator import DiscriminatorSB
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.tools.pretrained_selector import pretrained_selector
from rvc.lib.utils import checkpoint_gen_istft, checkpoint_text_enc_hidden_dim
from rvc.train import warm_start as warm_start_module
from rvc.train.extract.preparing_files import (
    resolve_vocoder_model_config,
    vocoder_model_config,
)
from rvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from rvc.train.process.extract_model import extract_model
from rvc.train.reset_run import reset_training_run
from rvc.train.utils import (
    HParams,
    assert_resumable,
    describe_architecture_mismatch,
    load_pretrained,
)
from rvc.train.vocoder_recipe import (
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
    infer_apex_gan_sample_rate,
    normalize_weight_norm_keys,
    warm_start,
)
from tests.test_codename_ringformer import fake_batch
from tests.test_warm_start import identity, save_legacy, snapshot

VOCODER = "APEX-GAN"
VERSION = "apex-gan"
SEGMENT = 17280
# Sub-discriminator indices of the apex-gan layout: five periods, three resolutions, then
# the four sub-band blocks.
PERIODS = range(0, 5)
RESOLUTIONS = range(5, 8)
SUBBANDS = range(8, 12)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def stock(sample_rate=48000, vocoder=None):
    with open(REPO / "rvc" / "configs" / f"{sample_rate}.json", encoding="utf-8") as f:
        config = json.load(f)
    if vocoder == VOCODER:
        path = REPO / "rvc" / "configs" / "apex_gan" / f"{sample_rate}.json"
        with open(path, encoding="utf-8") as f:
            config["model"].update(json.load(f)["model"])
    return config


def build_synthesizer(vocoder, sample_rate=48000, dim=768, speakers=1, **kwargs):
    config = stock(sample_rate, vocoder)
    model = dict(config["model"], text_enc_hidden_dim=dim, spk_embed_dim=speakers)
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


def build_decoder(**kwargs):
    model = stock(48000, VOCODER)["model"]
    return APEXGANGenerator(
        model["inter_channels"],
        model["resblock_kernel_sizes"],
        model["resblock_dilation_sizes"],
        model["upsample_rates"],
        model["upsample_initial_channel"],
        model["upsample_kernel_sizes"],
        gin_channels=model["gin_channels"],
        sr=48000,
        gen_istft_n_fft=model["gen_istft_n_fft"],
        gen_istft_hop_size=model["gen_istft_hop_size"],
        **kwargs,
    )


def fake_f0(batch, frames):
    f0 = torch.rand(batch, frames) * 300 + 80
    f0[:, ::5] = 0.0
    return f0


def fresh_d(version=VERSION):
    return MultiPeriodDiscriminator(
        False, version=version, sample_rate=48000, segment_size=SEGMENT
    )


@contextlib.contextmanager
def switches(upsamplers, resblocks):
    """Flip the HiFi-GAN -> APEX-GAN stage port switches for the duration."""
    with mock.patch.object(
        warm_start_module, "APEX_GAN_PORTS_UPSAMPLERS", upsamplers
    ), mock.patch.object(warm_start_module, "APEX_GAN_PORTS_RESBLOCKS", resblocks):
        yield


def sub_keys(state, index):
    return [key for key in state if key.startswith(f"discriminators.{index}.")]


class GeneratorTest(unittest.TestCase):
    def test_the_output_is_exactly_the_latent_frames_times_the_hop(self):
        torch.manual_seed(0)
        decoder = build_decoder().eval()
        for frames in (36, 1, 7, 37):
            with self.subTest(frames=frames):
                out = decoder(
                    torch.randn(2, 192, frames),
                    fake_f0(2, frames),
                    torch.randn(2, 256, 1),
                )
                self.assertEqual(out.shape, (2, 1, frames * 480))
                self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(36 * 480, stock(48000)["train"]["segment_size"])

    def test_the_excitation_lines_up_with_every_stage(self):
        decoder = build_decoder().eval()
        lengths = {}

        def record(name):
            def hook(_module, _inputs, output):
                lengths[name] = output.shape[-1]

            return hook

        for i in range(decoder.num_upsamples):
            decoder.ups[i].register_forward_hook(record(f"ups{i}"))
            decoder.noise_convs[i].register_forward_hook(record(f"noise{i}"))
            decoder.phase_convs[i].proj.register_forward_hook(record(f"phase{i}"))
        frames = 36
        decoder(torch.randn(1, 192, frames), fake_f0(1, frames), torch.randn(1, 256, 1))
        self.assertEqual(lengths["ups0"], frames * 12)
        self.assertEqual(lengths["noise0"], lengths["ups0"])
        self.assertEqual(lengths["phase0"], lengths["ups0"])
        # The last stage is reflection padded by one to the STFT frame count.
        self.assertEqual(lengths["ups1"] + 1, frames * 120 + 1)
        self.assertEqual(lengths["noise1"], frames * 120 + 1)
        self.assertEqual(lengths["phase1"], frames * 120 + 1)

    def test_the_phase_reference_can_only_rotate(self):
        torch.manual_seed(0)
        projector = PhaseProjector(17, 64, 10)
        cos_beta, sin_beta = projector(torch.randn(2, 17, 401) * 3)
        radius = cos_beta**2 + sin_beta**2
        self.assertTrue(torch.all(radius <= 1.0 + 1e-6))
        self.assertGreater(radius.median().item(), 0.999)

    def test_the_excitation_is_a_voiced_sine_and_unvoiced_noise(self):
        torch.manual_seed(0)
        synthesizer = ExcitationSynthesizer(48000, hop_length=480)
        f0 = torch.full((1, 20), 200.0)
        f0[:, 10:] = 0.0
        sine, noise = synthesizer(f0)
        self.assertEqual(sine.shape, (1, 1, 20 * 480))
        voiced, unvoiced = slice(0, 10 * 480), slice(10 * 480, None)
        self.assertEqual(sine[..., unvoiced].abs().max().item(), 0.0)
        self.assertAlmostEqual(sine[..., voiced].abs().max().item(), 0.1, places=3)
        self.assertAlmostEqual(noise[..., voiced].std().item(), 0.003, places=3)
        self.assertAlmostEqual(noise[..., unvoiced].std().item(), 0.1 / 3, places=2)

    def test_checkpointing_changes_nothing(self):
        torch.manual_seed(0)
        plain = build_decoder().train()
        checkpointed = build_decoder(checkpointing=True).train()
        checkpointed.load_state_dict(plain.state_dict())
        x, f0, g = torch.randn(1, 192, 12), fake_f0(1, 12), torch.randn(1, 256, 1)
        results = []
        for decoder in (plain, checkpointed):
            torch.manual_seed(1)
            out = decoder(x, f0, g)
            out.square().mean().backward()
            results.append(
                (out.detach(), {n: p.grad for n, p in decoder.named_parameters()})
            )
        self.assertTrue(torch.allclose(results[0][0], results[1][0], atol=1e-6))
        for name, grad in results[0][1].items():
            other = results[1][1][name]
            if grad is None:
                self.assertIsNone(other, name)
            else:
                self.assertTrue(torch.allclose(grad, other, atol=1e-5), name)

    def test_removing_weight_norm_keeps_the_function(self):
        torch.manual_seed(0)
        decoder = build_decoder().eval()
        x, f0, g = torch.randn(1, 192, 10), fake_f0(1, 10), torch.randn(1, 256, 1)
        torch.manual_seed(1)
        before = decoder(x, f0, g)
        decoder.remove_weight_norm()
        torch.manual_seed(1)
        after = decoder(x, f0, g)
        self.assertTrue(torch.allclose(before, after, atol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_a_half_precision_backward_survives(self):
        """autocast does not reach backward, so only a real backward proves anything."""
        decoder = build_decoder().cuda().train()
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                decoder.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype):
                    out = decoder(
                        torch.randn(2, 192, 36, device="cuda"),
                        fake_f0(2, 36).cuda(),
                        torch.randn(2, 256, 1, device="cuda"),
                    )
                self.assertEqual(out.dtype, torch.float32)
                out.square().mean().backward()
                grads = [
                    p.grad for p in decoder.parameters() if p.grad is not None
                ]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_only_48k_has_a_decoder_config(self):
        self.assertEqual(default_istft_settings(48000), (32, 4))
        self.assertEqual(set(APEX_GAN_ISTFT), {48000})
        with self.assertRaises(ValueError):
            default_istft_settings(40000)


class SnakeTest(unittest.TestCase):
    def _inputs(self, dtype=torch.float64):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 11, dtype=dtype)
        raw = torch.randn(2, 6, 11, dtype=dtype)
        c, s = raw.chunk(2, 1)
        norm = torch.sqrt(c * c + s * s)
        return x, c / norm, s / norm

    def test_it_is_the_phase_shifted_snake(self):
        x, cos_beta, sin_beta = self._inputs()
        snake = Snake(3, init=1.0).double()
        with torch.no_grad():
            snake.alpha.copy_(torch.tensor([0.5, 1.0, 2.0]))
        alpha = snake.alpha[None, :, None]
        beta = torch.atan2(sin_beta, cos_beta)
        expected = x + torch.sin(alpha * x + beta) ** 2 / alpha
        self.assertTrue(torch.allclose(snake(x, beta=(cos_beta, sin_beta)), expected))
        plain = x + torch.sin(alpha * x) ** 2 / alpha
        self.assertTrue(torch.allclose(snake(x), plain))

    def test_without_beta_it_is_the_plain_snake(self):
        from rvc.lib.algorithm.conformer.snake import Snake as PlainSnake

        x, _, _ = self._inputs(torch.float32)
        ours = Snake(3, init=1.0)
        plain = PlainSnake(3, init=1.0, correction=None)
        self.assertTrue(torch.allclose(ours(x), plain(x), atol=1e-6))

    def test_the_backward_is_the_derivative(self):
        x, cos_beta, sin_beta = self._inputs()
        alpha = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
        x.requires_grad_(True)
        cos_beta.requires_grad_(True)
        sin_beta.requires_grad_(True)
        function = __import__(
            "rvc.lib.algorithm.generators.apex_gan_snake", fromlist=["_SnakeFunction"]
        )._SnakeFunction
        self.assertTrue(
            torch.autograd.gradcheck(
                function.apply, (x, alpha, cos_beta, sin_beta), eps=1e-6, atol=1e-5
            )
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x, a: function.apply(x, a, None, None), (x, alpha), eps=1e-6
            )
        )

    def test_the_output_keeps_the_input_dtype(self):
        snake = Snake(3)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.randn(1, 3, 5).to(dtype)
            self.assertEqual(snake(x).dtype, dtype)


class SynthesizerTest(unittest.TestCase):
    def test_forward_returns_no_extra_and_infer_the_waveform(self):
        torch.manual_seed(0)
        config = stock(48000, VOCODER)
        net = build_synthesizer(VOCODER)
        self.assertIsInstance(net.dec, APEXGANGenerator)
        batch = fake_batch(config)
        y_hat, ids_slice, *_, extra = net(
            batch["phone"],
            batch["phone_lengths"],
            batch["pitch"],
            batch["pitchf"],
            batch["spec"],
            batch["phone_lengths"],
            batch["sid"],
        )
        self.assertEqual(y_hat.shape, (2, 1, SEGMENT))
        self.assertIsNone(extra)
        net.eval()
        frames = 40
        with torch.no_grad():
            out, _, _ = net.infer(
                torch.randn(1, frames, 768),
                torch.tensor([frames]),
                torch.randint(1, 255, (1, frames)),
                fake_f0(1, frames),
                torch.zeros(1, dtype=torch.long),
            )
        self.assertEqual(out.shape, (1, 1, frames * 480))

    def test_the_decoder_keys_are_the_codename_forks(self):
        state = build_synthesizer(VOCODER).state_dict()
        for key in (
            "dec.excitation_synthesizer.l_linear.weight",
            "dec.conv_pre.parametrizations.weight.original0",
            "dec.ups.1.parametrizations.weight.original1",
            "dec.noise_convs.0.weight",
            "dec.phase_convs.1.proj.weight",
            "dec.resblocks.5.convs2.2.parametrizations.weight.original1",
            "dec.resblocks.5.snakes1.2.alpha",
            "dec.resblocks.0.snakes2.0.alpha",
            "dec.conv_post.parametrizations.weight.original1",
            "dec.cond.weight",
        ):
            self.assertIn(key, state)
        self.assertNotIn("dec.noise_convs.0.bias", state)
        self.assertEqual(len([k for k in state if k.startswith("dec.")]), 166)
        self.assertEqual(
            state["dec.conv_post.parametrizations.weight.original1"].shape[0], 34
        )

    def test_every_decoder_parameter_but_the_dead_one_learns(self):
        """DDP runs without find_unused_parameters, so nothing trainable may sit idle."""
        torch.manual_seed(0)
        config = stock(48000, VOCODER)
        net = build_synthesizer(VOCODER)
        batch = fake_batch(config)
        y_hat, *_ = net(
            batch["phone"],
            batch["phone_lengths"],
            batch["pitch"],
            batch["pitchf"],
            batch["spec"],
            batch["phone_lengths"],
            batch["sid"],
        )
        y_hat.square().mean().backward()
        idle = [
            name
            for name, p in net.named_parameters()
            if name.startswith("dec.") and p.requires_grad and p.grad is None
        ]
        self.assertEqual(idle, [])
        frozen = [n for n, p in net.named_parameters() if not p.requires_grad]
        self.assertEqual(
            sorted(frozen),
            [
                "dec.excitation_synthesizer.l_linear.bias",
                "dec.excitation_synthesizer.l_linear.weight",
            ],
        )

    def test_it_needs_pitch_guidance(self):
        config = stock(48000, VOCODER)
        net = quiet(
            Synthesizer,
            config["data"]["filter_length"] // 2 + 1,
            36,
            **dict(config["model"], spk_embed_dim=1),
            use_f0=False,
            sr=48000,
            vocoder=VOCODER,
        )
        self.assertIsNone(net.dec)


class IdentityTest(unittest.TestCase):
    def test_it_is_recognised_from_its_weights(self):
        self.assertEqual(detect_vocoder(build_synthesizer(VOCODER).state_dict()), APEX_GAN)

    def test_the_other_vocoders_are_still_recognised(self):
        self.assertEqual(detect_vocoder(build_synthesizer(HIFIGAN).state_dict()), HIFIGAN)
        from tests.test_codename_ringformer import build_synthesizer as ringformer

        net = quiet(ringformer, "CodenameRingFormer")
        self.assertEqual(detect_vocoder(net.state_dict()), "CodenameRingFormer")

    def test_the_sample_rate_survives_a_legacy_save(self):
        net = build_synthesizer(VOCODER)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "G.pth")
            quiet(save_legacy, net, path)
            weights = normalize_weight_norm_keys(
                torch.load(path, map_location="cpu", weights_only=True)["model"]
            )
        self.assertEqual(detect_vocoder(weights), APEX_GAN)
        self.assertEqual(infer_apex_gan_sample_rate(weights), 48000)


class GeneratorWarmStartTest(unittest.TestCase):
    """What a HiFi-GAN pretrain may and may not contribute."""

    def _warm(self, upsamplers=False, resblocks=False, target_dim=768, source_rate=48000):
        source = build_synthesizer(HIFIGAN, sample_rate=source_rate)
        target = build_synthesizer(VOCODER, dim=target_dim)
        before = snapshot(target)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "G.pth")
            quiet(save_legacy, source, path, identity(HIFIGAN, source_rate))
            with switches(upsamplers, resblocks):
                transfer = load_pretrained(
                    target, path, "G", verbose=False, target_identity=identity(VOCODER)
                )
        return source.state_dict(), target.state_dict(), before, transfer

    def test_the_encoders_flow_and_speaker_embedding_are_inherited(self):
        source, target, _, transfer = self._warm()
        for key, value in target.items():
            if key.startswith(("enc_p.", "enc_q.", "flow.", "emb_g.")):
                self.assertIn(key, transfer.loaded)
                self.assertTrue(torch.equal(value, source[key]), key)

    def test_conv_pre_and_the_speaker_conditioning_carry_over(self):
        source, target, _, transfer = self._warm()
        self.assertTrue(torch.equal(target["dec.cond.weight"], source["dec.cond.weight"]))
        magnitude = target["dec.conv_pre.parametrizations.weight.original0"]
        direction = target["dec.conv_pre.parametrizations.weight.original1"]
        effective = magnitude * direction / direction.norm(dim=(1, 2), keepdim=True)
        self.assertTrue(torch.allclose(effective, source["dec.conv_pre.weight"], atol=1e-5))
        self.assertIn("dec.conv_pre.parametrizations.weight.original0", transfer.transformed)

    def test_without_the_stage_port_the_stages_start_from_scratch(self):
        _, _, _, transfer = self._warm()
        for prefix in (
            "dec.ups.",
            "dec.resblocks.",
            "dec.noise_convs.",
            "dec.phase_convs.",
            "dec.conv_post.",
            "dec.excitation_synthesizer.",
        ):
            self.assertTrue(
                [k for k in transfer.reinitialised if k.startswith(prefix)], prefix
            )
            self.assertFalse([k for k in transfer.loaded if k.startswith(prefix)], prefix)
        for prefix in ("dec.noise_convs.", "dec.m_source.", "dec.conv_post.", "dec.ups.3."):
            self.assertTrue([k for k in transfer.dropped if k.startswith(prefix)], prefix)

    def test_the_switches_are_the_measured_defaults(self):
        self.assertFalse(warm_start_module.APEX_GAN_PORTS_RESBLOCKS)

    def test_the_upsampler_port_takes_the_first_two_transposed_convolutions_only(self):
        source, target, before, transfer = self._warm(upsamplers=True)
        for i in (0, 1):
            key = f"dec.ups.{i}.parametrizations.weight.original1"
            self.assertTrue(torch.equal(target[key], source[key]), key)
        self.assertFalse([k for k in transfer.loaded if k.startswith("dec.resblocks.")])
        self.assertTrue([k for k in transfer.dropped if k.startswith("dec.ups.2.")])

    def test_with_the_whole_stage_port_the_first_two_stages_carry_over(self):
        source, target, before, transfer = self._warm(upsamplers=True, resblocks=True)
        for i in (0, 1):
            for leaf in ("parametrizations.weight.original0", "parametrizations.weight.original1", "bias"):
                key = f"dec.ups.{i}.{leaf}"
                self.assertTrue(torch.equal(target[key], source[key]), key)
        for j in range(6):
            key = f"dec.resblocks.{j}.convs1.0.parametrizations.weight.original1"
            self.assertTrue(torch.equal(target[key], source[key]), key)
        # The Snakes have no counterpart in a LeakyReLU block.
        alphas = [k for k in target if k.endswith(".alpha")]
        self.assertTrue(alphas)
        for key in alphas:
            self.assertIn(key, transfer.reinitialised)
            self.assertTrue(torch.equal(target[key], before[key]), key)
        for prefix in ("dec.ups.2.", "dec.ups.3.", "dec.resblocks.6.", "dec.resblocks.11."):
            self.assertTrue([k for k in transfer.dropped if k.startswith(prefix)], prefix)
        for prefix in ("dec.noise_convs.", "dec.phase_convs.", "dec.conv_post."):
            self.assertFalse([k for k in transfer.loaded if k.startswith(prefix)], prefix)

    def test_a_wider_embedder_costs_only_the_projection(self):
        _, _, _, transfer = self._warm(target_dim=1024)
        self.assertIn("enc_p.emb_phone.weight", transfer.reinitialised)
        self.assertIn("enc_p.emb_phone.bias", transfer.loaded)

    def test_a_different_sample_rate_is_refused_for_the_decoder(self):
        _, _, _, transfer = self._warm(True, True, source_rate=40000)
        self.assertFalse([k for k in transfer.loaded if k.startswith("dec.")])
        self.assertTrue(any("40000" in note for note in transfer.notes))

    def test_it_round_trips_back_into_a_hifigan_model(self):
        for stages in (False, True):
            with self.subTest(stages=stages):
                source = build_synthesizer(VOCODER)
                target = build_synthesizer(HIFIGAN)
                with tempfile.TemporaryDirectory() as folder:
                    path = os.path.join(folder, "G.pth")
                    quiet(save_legacy, source, path, identity(VOCODER))
                    with switches(stages, stages):
                        transfer = load_pretrained(
                            target, path, "G", verbose=False,
                            target_identity=identity(HIFIGAN),
                        )
                self.assertIn("dec.conv_pre.weight", transfer.loaded)
                self.assertIn("dec.cond.weight", transfer.loaded)
                self.assertEqual(
                    "dec.ups.0.parametrizations.weight.original1" in transfer.loaded,
                    stages,
                )

    def test_the_same_vocoder_loads_whole(self):
        source, target = build_synthesizer(VOCODER), build_synthesizer(VOCODER)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "G.pth")
            quiet(save_legacy, source, path, identity(VOCODER))
            transfer = load_pretrained(
                target, path, "G", verbose=False, target_identity=identity(VOCODER)
            )
        self.assertEqual(transfer.reinitialised, {})
        self.assertEqual(transfer.dropped, {})
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value), key)


class DiscriminatorTest(unittest.TestCase):
    def test_the_layout_is_the_codename_forks(self):
        layout = discriminator_layout(VERSION)
        self.assertNotIn(("S",), layout)
        self.assertEqual(layout[:5], [("P", p) for p in (2, 3, 5, 7, 11)])
        self.assertEqual(
            layout[5:8],
            [("R", tuple(r), "ones") for r in DISCRIMINATOR_VERSIONS["v3"]["resolutions"]],
        )
        self.assertEqual(
            layout[8:],
            [
                ("B", 16, (0, 6), False),
                ("B", 16, (0, 11), False),
                ("B", 16, (0, 16), False),
                ("B", 64, (0, 64), True),
            ],
        )
        net = fresh_d()
        self.assertEqual([describe_discriminator(d) for d in net.discriminators], layout)

    def test_it_needs_the_segment_size_and_the_others_do_not(self):
        with self.assertRaises(ValueError):
            MultiPeriodDiscriminator(False, version=VERSION, sample_rate=48000)
        for version in ("v1", "v2", "v3", "v3-48k", "codename-ringformer"):
            MultiPeriodDiscriminator(False, version=version)

    def test_the_other_versions_keep_their_scale_discriminator(self):
        # apex-gan-48k is APEX-GAN+'s: apex-gan with other resolutions, so the same kinds.
        for version in DISCRIMINATOR_VERSIONS:
            if version in (VERSION, "apex-gan-48k"):
                continue
            layout = discriminator_layout(version)
            self.assertEqual(layout[0], ("S",), version)
            self.assertFalse([d for d in layout if d[0] == "B"], version)

    def test_only_trained_weights_are_saved(self):
        state = fresh_d().state_dict()
        self.assertFalse([k for k in state if "pqmf" in k or "filter" in k])

    def test_the_transposed_block_reads_the_segment_in_64_sample_frames(self):
        block = DiscriminatorSB(3, SEGMENT)
        weight = block.convs[0].d_convs[0].parametrizations.weight.original1
        self.assertEqual(weight.shape[1], SEGMENT // 64)

    def test_forward_and_backward(self):
        torch.manual_seed(0)
        net = fresh_d()
        y, y_hat = torch.randn(2, 1, SEGMENT) * 0.3, (torch.randn(2, 1, SEGMENT) * 0.3).requires_grad_()
        y_d_r, y_d_g, fmap_r, fmap_g = net(y, y_hat)
        self.assertEqual(len(y_d_r), 12)
        self.assertEqual([len(f) for f in fmap_r[8:]], [5, 5, 5, 5])
        (generator_loss(y_d_g)[0] + feature_loss(fmap_r, fmap_g)).backward()
        self.assertTrue(torch.isfinite(y_hat.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_forward_and_backward_under_half_precision_autocast(self):
        net = fresh_d().cuda()
        y = torch.randn(2, 1, SEGMENT, device="cuda") * 0.3
        y_hat = (torch.randn(2, 1, SEGMENT, device="cuda") * 0.3).requires_grad_()
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                with torch.autocast("cuda", dtype=dtype):
                    y_d_r, y_d_g, _, _ = net(y, y_hat)
                loss, _, _ = discriminator_loss(y_d_r, y_d_g)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))


class DiscriminatorWarmStartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.sources = {}
        for version in ("v2", "v3", VERSION):
            net = fresh_d(version)
            path = os.path.join(cls.tmp.name, f"D_{version}.pth")
            quiet(save_legacy, net, path, {"disc_version": version})
            cls.sources[version] = (path, net.state_dict())

    def _warm(self, source, target_version):
        target = fresh_d(target_version)
        before = snapshot(target)
        transfer = quiet(warm_start, target, self.sources[source][0], "D")
        self.assertEqual(transfer.errors, [])
        return target.state_dict(), before, self.sources[source][1], transfer

    def _inherited(self, loaded, saved, target_index, source_index):
        keys = sub_keys(loaded, target_index)
        self.assertTrue(keys)
        for key in keys:
            source_key = f"discriminators.{source_index}." + key.split(".", 2)[2]
            self.assertTrue(torch.equal(loaded[key], saved[source_key]), key)

    def _fresh(self, loaded, before, index):
        keys = sub_keys(loaded, index)
        self.assertTrue(keys)
        for key in keys:
            self.assertTrue(torch.equal(loaded[key], before[key]), key)

    def test_from_hifigan_the_five_periods_come(self):
        loaded, before, saved, transfer = self._warm("v2", VERSION)
        for index in PERIODS:  # v2: scale at 0, periods 2 .. 11 at 1 .. 5
            self._inherited(loaded, saved, index, index + 1)
        for index in (*RESOLUTIONS, *SUBBANDS):
            self._fresh(loaded, before, index)
        dropped = set(transfer.dropped.values())
        self.assertEqual(
            dropped,
            {
                "this model has no scale discriminator",
                "this model has no period 17 discriminator",
                "this model has no period 23 discriminator",
                "this model has no period 37 discriminator",
            },
        )

    def test_from_v3_the_periods_and_resolutions_come(self):
        loaded, before, saved, transfer = self._warm("v3", VERSION)
        for index in (*PERIODS, *RESOLUTIONS):
            self._inherited(loaded, saved, index, index + 1)
        for index in SUBBANDS:
            self._fresh(loaded, before, index)
        self.assertEqual(set(transfer.dropped.values()), {"this model has no scale discriminator"})
        self.assertTrue(
            all(r.startswith("the pretrained discriminator has no sub-band") for r in transfer.reinitialised.values())
        )

    def test_back_to_v3_drops_only_the_subbands(self):
        loaded, before, saved, transfer = self._warm(VERSION, "v3")
        self._fresh(loaded, before, 0)
        for index in range(1, 9):
            self._inherited(loaded, saved, index, index - 1)
        self.assertEqual(len(set(transfer.dropped.values())), 4)

    def test_the_same_layout_loads_whole(self):
        loaded, _, saved, transfer = self._warm(VERSION, VERSION)
        self.assertEqual(transfer.reinitialised, {})
        self.assertEqual(transfer.dropped, {})
        for key, value in saved.items():
            self.assertTrue(torch.equal(loaded[key], value), key)

    def test_the_layout_is_read_off_the_weights_without_a_record(self):
        path = os.path.join(self.tmp.name, "D_unrecorded.pth")
        net = fresh_d()
        quiet(save_legacy, net, path, None)
        target = fresh_d()
        transfer = quiet(warm_start, target, path, "D")
        self.assertEqual(transfer.errors, [])
        self.assertEqual(transfer.reinitialised, {})


class RecipeTest(unittest.TestCase):
    def test_it_is_its_own_generator_against_its_own_discriminator(self):
        self.assertEqual(generator_vocoder(VOCODER), VOCODER)
        self.assertEqual(discriminator_version(VOCODER), VERSION)
        self.assertFalse(uses_multiscale_mel_loss(VOCODER))

    def test_only_48k_is_accepted(self):
        self.assertIsNone(sample_rate_error(VOCODER, 48000))
        for sample_rate in (32000, 40000):
            message = sample_rate_error(VOCODER, sample_rate)
            self.assertIn("48 kHz", message)
            self.assertIn("upsampling", message)
        # SiFi-GAN+'s reason is about its discriminator, and stays that way.
        self.assertIn("STFT resolutions", sample_rate_error("SiFi-GAN+", 40000))

    def test_the_pretrained_folder_falls_back_to_hifigan(self):
        self.assertEqual(pretrained_dir_name(VOCODER), "apex-gan")
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            pretraineds = Path(temp_dir) / "rvc" / "models" / "pretraineds"
            (pretraineds / "hifi-gan").mkdir(parents=True)
            for name in ("f0G48k.pth", "f0D48k.pth"):
                (pretraineds / "hifi-gan" / name).write_bytes(b"")
            os.chdir(temp_dir)
            try:
                g, d = quiet(pretrained_selector, VOCODER, 48000)
            finally:
                os.chdir(cwd)
        self.assertEqual(Path(g), Path("rvc/models/pretraineds/hifi-gan/f0G48k.pth"))
        self.assertEqual(Path(d), Path("rvc/models/pretraineds/hifi-gan/f0D48k.pth"))

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
        self.assertIn(VOCODER, vocoder.choices)


class ConfigResolutionTest(unittest.TestCase):
    def test_the_decoder_keys_are_written_and_taken_back_out_again(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "config.json")
            shutil.copyfile(REPO / "rvc" / "configs" / "48000.json", path)
            with open(path, encoding="utf-8") as f:
                config = json.load(f)
            config["train"]["learning_rate"] = 5e-5
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f)

            quiet(resolve_vocoder_model_config, folder, 48000, VOCODER)
            with open(path, encoding="utf-8") as f:
                after = json.load(f)
            self.assertEqual(after["model"]["upsample_rates"], [12, 10])
            self.assertEqual(after["model"]["upsample_kernel_sizes"], [24, 20])
            self.assertEqual(after["model"]["gen_istft_n_fft"], 32)
            self.assertEqual(after["model"]["gen_istft_hop_size"], 4)
            self.assertEqual(after["train"]["learning_rate"], 5e-5)

            quiet(resolve_vocoder_model_config, folder, 48000, HIFIGAN)
            with open(path, encoding="utf-8") as f:
                back = json.load(f)
            self.assertEqual(back["model"]["upsample_rates"], [12, 10, 2, 2])
            self.assertNotIn("gen_istft_n_fft", back["model"])
            self.assertNotIn("gen_istft_hop_size", back["model"])

    def test_it_multiplies_out_to_the_hop_length(self):
        settings = vocoder_model_config(VOCODER, 48000)
        self.assertEqual(
            math.prod(settings["upsample_rates"]) * settings["gen_istft_hop_size"], 480
        )
        self.assertEqual(
            (settings["gen_istft_n_fft"], settings["gen_istft_hop_size"]),
            default_istft_settings(48000),
        )


class ExportTest(unittest.TestCase):
    def _export(self, folder, dim=768):
        config = stock(48000, VOCODER)
        with open(os.path.join(folder, "model_info.json"), "w", encoding="utf-8") as f:
            json.dump({"embedder_model": "contentvec", "speakers_id": 1}, f)
        hps = HParams(
            **dict(
                config,
                model=dict(config["model"], spk_embed_dim=1, text_enc_hidden_dim=dim),
            )
        )
        path = os.path.join(folder, "m_1e_1s.pth")
        quiet(
            extract_model,
            build_synthesizer(VOCODER, dim=dim).state_dict(),
            48000,
            "m",
            path,
            1,
            1,
            hps,
            None,
            VOCODER,
        )
        self.assertTrue(os.path.isfile(path), "extract_model produced no file")
        return path, torch.load(path, map_location="cpu", weights_only=True)

    def test_it_exports_loads_and_infers(self):
        for dim in (768, 1024):
            with self.subTest(dim=dim), tempfile.TemporaryDirectory() as folder:
                _, cpt = self._export(folder, dim)
                self.assertEqual(cpt["vocoder"], VOCODER)
                self.assertEqual((cpt["gen_istft_n_fft"], cpt["gen_istft_hop_size"]), (32, 4))
                self.assertTrue(any(k.endswith(".weight_v") for k in cpt["weight"]))
                self.assertEqual(checkpoint_gen_istft(cpt), (32, 4))
                # Exactly how rvc/infer/infer.py rebuilds it.
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
                n_fft, hop = checkpoint_gen_istft(cpt)
                net = quiet(
                    Synthesizer,
                    *cpt["config"],
                    use_f0=cpt.get("f0", 1),
                    text_enc_hidden_dim=checkpoint_text_enc_hidden_dim(cpt),
                    vocoder=cpt["vocoder"],
                    gen_istft_n_fft=n_fft,
                    gen_istft_hop_size=hop,
                )
                del net.enc_q
                missing, unexpected = net.load_state_dict(cpt["weight"], strict=False)
                self.assertEqual([k for k in missing if "enc_q" not in k], [])
                self.assertEqual(unexpected, [])
                frames = 30
                with torch.no_grad():
                    out, _, _ = net.eval().infer(
                        torch.randn(1, frames, dim),
                        torch.tensor([frames]),
                        torch.randint(1, 255, (1, frames)),
                        fake_f0(1, frames),
                        torch.zeros(1, dtype=torch.long),
                    )
                self.assertEqual(out.shape, (1, 1, frames * 480))
                self.assertTrue(torch.isfinite(out).all())

    def test_the_istft_settings_are_recoverable_without_the_recorded_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            _, cpt = self._export(folder)
        del cpt["gen_istft_n_fft"], cpt["gen_istft_hop_size"]
        self.assertEqual(checkpoint_gen_istft(cpt), (32, 4))

    def test_the_exported_model_loads_through_voice_converter(self):
        from rvc.infer.infer import VoiceConverter

        with tempfile.TemporaryDirectory() as folder:
            path, _ = self._export(folder)
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


class ResumeAndResetTest(unittest.TestCase):
    def test_resuming_into_a_different_vocoder_is_refused(self):
        net = build_synthesizer(HIFIGAN)
        with tempfile.TemporaryDirectory() as folder:
            quiet(save_legacy, net, os.path.join(folder, "G_100.pth"), identity(HIFIGAN))
            with self.assertRaises(SystemExit):
                quiet(assert_resumable, folder, {}, identity(VOCODER))

    def test_resuming_into_the_same_vocoder_is_allowed(self):
        net = build_synthesizer(VOCODER)
        with tempfile.TemporaryDirectory() as folder:
            quiet(
                save_legacy,
                net,
                os.path.join(folder, "G_100.pth"),
                dict(identity(VOCODER), disc_version=VERSION),
            )
            quiet(
                assert_resumable,
                folder,
                {},
                dict(identity(VOCODER), disc_version=VERSION),
            )

    def test_a_hifigan_folder_resets_into_apex_gan(self):
        from rvc.train.utils import save_checkpoint

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "branch"
            root.mkdir()
            config = stock(48000)
            config["model"]["text_enc_hidden_dim"] = 768
            (root / "config.json").write_text(json.dumps(config))
            # The reset rebuilds the generator from config.json, speakers included.
            net_g = build_synthesizer(HIFIGAN, speakers=config["model"]["spk_embed_dim"])
            net_d = MultiPeriodDiscriminator(False, version="v2")
            arch = {"vocoder": HIFIGAN, "sample_rate": 48000, "disc_version": "v2"}
            for tag, net in (("G", net_g), ("D", net_d)):
                quiet(
                    save_checkpoint,
                    net,
                    torch.optim.AdamW(net.parameters(), lr=1e-4),
                    1e-4,
                    840,
                    str(root / f"{tag}_2333333.pth"),
                    torch.amp.GradScaler(enabled=False),
                    architecture_identity=arch,
                )
            source_g = snapshot(net_g)
            archive = quiet(reset_training_run, root, vocoder=VOCODER)

            generator = torch.load(root / "G_0.pth", map_location="cpu", weights_only=True)
            self.assertEqual(generator["vocoder"], VOCODER)
            self.assertEqual(generator["disc_version"], VERSION)
            weights = normalize_weight_norm_keys(generator["model"])
            self.assertEqual(detect_vocoder(weights), APEX_GAN)
            self.assertTrue(torch.equal(weights["dec.cond.weight"], source_g["dec.cond.weight"]))
            self.assertTrue(torch.equal(weights["flow.flows.0.enc.in_layers.0.bias"], source_g["flow.flows.0.enc.in_layers.0.bias"]))

            discriminator = torch.load(root / "D_0.pth", map_location="cpu", weights_only=True)
            self.assertEqual(discriminator["disc_version"], VERSION)
            target = fresh_d()
            target.load_state_dict(discriminator["model"])
            target_identity = {"vocoder": VOCODER, "sample_rate": 48000, "disc_version": VERSION}
            self.assertIsNone(describe_architecture_mismatch(generator, target_identity))
            quiet(assert_resumable, str(root), {}, target_identity)
            self.assertIsNotNone(json.loads((archive / "reset.json").read_text())["generator_transfer"])


class TrainingStepTest(unittest.TestCase):
    """One real step with the apex-gan discriminator and every loss train.py computes."""

    def test_a_step_runs_and_the_decoder_learns(self):
        torch.manual_seed(0)
        config = stock(48000, VOCODER)
        net_g = build_synthesizer(VOCODER)
        net_d = fresh_d()
        optim_g = torch.optim.AdamW(net_g.parameters(), 1e-4)
        optim_d = torch.optim.AdamW(net_d.parameters(), 1e-4)
        batch = fake_batch(config)
        wave = torch.randn(2, 1, SEGMENT) * 0.1

        y_hat, ids_slice, _, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), _ = net_g(
            batch["phone"],
            batch["phone_lengths"],
            batch["pitch"],
            batch["pitchf"],
            batch["spec"],
            batch["phone_lengths"],
            batch["sid"],
        )
        y_d_r, y_d_g, _, _ = net_d(wave, y_hat.detach())
        loss_disc, _, _ = discriminator_loss(y_d_r, y_d_g)
        optim_d.zero_grad()
        loss_disc.backward()
        optim_d.step()

        _, y_d_g, fmap_r, fmap_g = net_d(wave, y_hat)
        data = config["data"]
        mel = spec_to_mel_torch(
            batch["spec"], data["filter_length"], data["n_mel_channels"],
            data["sample_rate"], data["mel_fmin"], data["mel_fmax"],
        )
        y_mel = commons.slice_segments(mel, ids_slice, SEGMENT // data["hop_length"], dim=3)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1), data["filter_length"], data["n_mel_channels"],
            data["sample_rate"], data["hop_length"], data["win_length"],
            data["mel_fmin"], data["mel_fmax"],
        )
        loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * config["train"]["c_mel"]
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config["train"]["c_kl"]
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_g)
        total = loss_gen + loss_fm + loss_mel + loss_kl
        for name, value in (("disc", loss_disc), ("gen", loss_gen), ("fm", loss_fm),
                            ("mel", loss_mel), ("kl", loss_kl)):
            self.assertTrue(torch.isfinite(value), name)

        before = snapshot(net_g)
        optim_g.zero_grad()
        total.backward()
        optim_g.step()
        after = net_g.state_dict()
        for prefix in ("dec.phase_convs.", "dec.noise_convs.", "dec.conv_post.", "dec.resblocks.0.snakes1."):
            changed = [k for k in after if k.startswith(prefix) and not torch.equal(after[k], before[k])]
            self.assertTrue(changed, f"{prefix} did not move")
        self.assertTrue(
            torch.equal(
                after["dec.excitation_synthesizer.l_linear.weight"],
                before["dec.excitation_synthesizer.l_linear.weight"],
            )
        )


if __name__ == "__main__":
    unittest.main()
