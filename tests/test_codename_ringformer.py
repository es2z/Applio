"""CodenameRingFormer: shapes, warm start, export and one real training step.

Run from the repository root:
    env\\python.exe -m unittest tests.test_codename_ringformer -v

Real configs, real modules, no mocks, in the style of tests/test_sifigan.py.
"""

import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# ring-attention-pytorch belongs in the environment, but a worktree that installed it
# beside the checkout should still be able to run these.
_DEPS = REPO / ".deps"
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))

from rvc.lib.algorithm import commons
from rvc.lib.algorithm.discriminators import (
    DEFAULT_RESOLUTION_WINDOW,
    DISCRIMINATOR_VERSIONS,
    MultiPeriodDiscriminator,
    describe_discriminator,
    discriminator_layout,
)
from rvc.lib.algorithm.generators.codename_ringformer import (
    CodenameRingFormerGenerator,
    default_istft_settings,
)
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.utils import checkpoint_gen_istft, checkpoint_text_enc_hidden_dim
from rvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from rvc.train.extract.preparing_files import (
    resolve_vocoder_model_config,
    vocoder_model_config,
)
from rvc.train.process.extract_model import extract_model
from rvc.train.spectral_loss import SpectralDistanceLoss
from rvc.train.utils import HParams, assert_resumable, load_pretrained
from rvc.train.warm_start import (
    CODENAME_RINGFORMER,
    detect_vocoder,
    infer_codename_ringformer_sample_rate,
)
from tests.test_warm_start import identity, save_legacy, snapshot

VOCODER = "CodenameRingFormer"


def stock(sample_rate, vocoder=None):
    """The config a run at this rate and vocoder is built from."""
    with open(REPO / "rvc" / "configs" / f"{sample_rate}.json", encoding="utf-8") as f:
        config = json.load(f)
    if vocoder == VOCODER:
        path = REPO / "rvc" / "configs" / "codename_ringformer" / f"{sample_rate}.json"
        with open(path, encoding="utf-8") as f:
            config["model"].update(json.load(f)["model"])
    return config


def build_decoder(sample_rate):
    model = stock(sample_rate, VOCODER)["model"]
    return CodenameRingFormerGenerator(
        model["inter_channels"],
        model["resblock_kernel_sizes"],
        model["resblock_dilation_sizes"],
        model["upsample_rates"],
        model["upsample_initial_channel"],
        model["upsample_kernel_sizes"],
        gin_channels=model["gin_channels"],
        sr=sample_rate,
        gen_istft_n_fft=model["gen_istft_n_fft"],
        gen_istft_hop_size=model["gen_istft_hop_size"],
    )


def build_synthesizer(vocoder, sample_rate=48000, text_enc_hidden_dim=768, speakers=1):
    config = stock(sample_rate, vocoder)
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


def fake_batch(config, frames=None, dim=768, batch=2):
    """Everything Synthesizer.forward wants, at the segment length of this config."""
    frames = frames or config["train"]["segment_size"] // config["data"]["hop_length"]
    return {
        "phone": torch.randn(batch, frames, dim),
        "phone_lengths": torch.full((batch,), frames, dtype=torch.long),
        "pitch": torch.randint(1, 255, (batch, frames)),
        "pitchf": torch.rand(batch, frames) * 200 + 100,
        "spec": torch.randn(
            batch, config["data"]["filter_length"] // 2 + 1, frames
        ).abs(),
        "sid": torch.zeros(batch, dtype=torch.long),
    }


class GeneratorShapeTest(unittest.TestCase):
    """The whole point of the iSTFT chain is that it lands on segment_size exactly."""

    def test_the_output_is_exactly_the_segment_length(self):
        for sample_rate in (32000, 40000, 48000):
            with self.subTest(sample_rate=sample_rate):
                config = stock(sample_rate, VOCODER)
                model = config["model"]
                frames = (
                    config["train"]["segment_size"] // config["data"]["hop_length"]
                )
                decoder = build_decoder(sample_rate)
                waveform, magnitude, phase = decoder(
                    torch.randn(2, model["inter_channels"], frames),
                    torch.rand(2, frames) * 200 + 100,
                    torch.randn(2, model["gin_channels"], 1),
                )
                self.assertEqual(
                    waveform.shape, (2, 1, frames * config["data"]["hop_length"])
                )
                # One iSTFT frame per upsampled step, plus the reflection pad.
                istft_frames = frames * math.prod(model["upsample_rates"]) + 1
                bins = model["gen_istft_n_fft"] // 2 + 1
                self.assertEqual(magnitude.shape, (2, bins, istft_frames))
                self.assertEqual(phase.shape, magnitude.shape)
                self.assertTrue(torch.isfinite(waveform).all())

    def test_the_harmonic_source_lines_up_with_every_stage(self):
        config = stock(48000, VOCODER)
        model = config["model"]
        frames = config["train"]["segment_size"] // config["data"]["hop_length"]
        decoder = build_decoder(48000)
        f0 = torch.rand(1, frames) * 200 + 100
        upsampled = decoder.f0_upsamp(f0[:, None]).transpose(1, 2)
        source, _, _ = decoder.m_source(upsampled)
        source = source.transpose(1, 2).squeeze(1)
        self.assertEqual(source.shape[-1], frames * config["data"]["hop_length"])

        spectrum, angle = decoder.stft.transform(source)
        harmonic = torch.cat([spectrum, angle], dim=1)
        self.assertEqual(harmonic.shape[1], model["gen_istft_n_fft"] + 2)
        # Every stage's excitation must be as long as the feature map it is added to.
        length = frames
        for i, rate in enumerate(model["upsample_rates"]):
            length *= rate
            expected = length + (1 if i == len(model["upsample_rates"]) - 1 else 0)
            self.assertEqual(decoder.noise_convs[i](harmonic).shape[-1], expected)

    def test_the_stock_istft_settings_are_recoverable_from_the_rate(self):
        for sample_rate in (32000, 40000, 48000):
            model = stock(sample_rate, VOCODER)["model"]
            self.assertEqual(
                default_istft_settings(sample_rate),
                (model["gen_istft_n_fft"], model["gen_istft_hop_size"]),
            )

    def test_the_attention_never_takes_the_triton_kernel_path(self):
        """RingAttention defaults use_cuda_kernel to torch.cuda.is_available(), and that
        path imports a Triton flash-attention kernel gated on the distribution name
        triton-nightly. Triton on Windows is distributed as triton-windows, so the gate
        fails and the module answers with print() + exit(): the training worker dies with
        no traceback and the parent still exits 0.
        """
        decoder = build_decoder(48000)
        for conformer in decoder.conformers:
            for block in conformer.layers:
                attention = block.attn.fn  # PreNorm wraps the RingAttention
                self.assertFalse(attention.use_cuda_kernel)
                self.assertFalse(attention.using_striped_ring_cuda)
                # And not the block-wise fallback either, whose hand-written backward
                # mixes dtypes under fp16 - see the half precision test below.
                self.assertTrue(attention.force_regular_attn)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() > 0,
        "needs a CUDA device",
    )
    def test_a_half_precision_backward_survives(self):
        """fp16 is the precision this fork trains at, and autocast does not reach backward.

        `ring_flash_attn`'s hand-written backward mixes fp32 and fp16 and raises "expected
        scalar type Float but found Half". A forward-only check passes anyway, so this one
        takes the gradient.
        """
        config = stock(48000, VOCODER)
        model = config["model"]
        frames = config["train"]["segment_size"] // config["data"]["hop_length"]
        decoder = build_decoder(48000).cuda()
        x = torch.randn(1, model["inter_channels"], frames, device="cuda")
        f0 = torch.rand(1, frames, device="cuda") * 200 + 100
        g = torch.randn(1, model["gin_channels"], 1, device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            waveform, magnitude, _ = decoder(x, f0, g)
        (waveform.float().pow(2).mean() + magnitude.float().mean()).backward()
        grads = [
            parameter.grad
            for name, parameter in decoder.named_parameters()
            if name.startswith("conformers.") and parameter.grad is not None
        ]
        self.assertTrue(grads, "the conformers received no gradient")
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))

    def test_the_decoder_follows_to_a_device(self):
        """The upstream TorchSTFT pins its window to cuda in the constructor."""
        decoder = build_decoder(48000)
        self.assertEqual(decoder.stft.window.device.type, "cpu")
        self.assertNotIn("stft.window", decoder.state_dict())


class SynthesizerTest(unittest.TestCase):
    def test_forward_returns_the_spectra_and_infer_does_not(self):
        config = stock(48000, VOCODER)
        for dim in (768, 1024):
            with self.subTest(dim=dim):
                net = build_synthesizer(VOCODER, text_enc_hidden_dim=dim)
                batch = fake_batch(config, dim=dim)
                output = net(
                    batch["phone"],
                    batch["phone_lengths"],
                    batch["pitch"],
                    batch["pitchf"],
                    batch["spec"],
                    batch["phone_lengths"],
                    batch["sid"],
                )
                self.assertEqual(len(output), 6)
                magnitude, phase = output[5]
                self.assertEqual(magnitude.shape, phase.shape)
                self.assertEqual(
                    output[0].shape[-1], config["train"]["segment_size"]
                )

                waveform, _, _ = net.infer(
                    batch["phone"],
                    batch["phone_lengths"],
                    batch["pitch"],
                    batch["pitchf"],
                    batch["sid"],
                )
                self.assertEqual(waveform.dim(), 3)
                self.assertEqual(waveform.shape[1], 1)

    def test_the_other_vocoders_still_return_none(self):
        config = stock(48000)
        batch = fake_batch(config)
        for vocoder in ("HiFi-GAN", "RefineGAN", "SiFi-GAN"):
            with self.subTest(vocoder=vocoder):
                net = build_synthesizer(vocoder)
                output = net(
                    batch["phone"],
                    batch["phone_lengths"],
                    batch["pitch"],
                    batch["pitchf"],
                    batch["spec"],
                    batch["phone_lengths"],
                    batch["sid"],
                )
                self.assertEqual(len(output), 6)
                if vocoder == "SiFi-GAN":
                    self.assertIsNotNone(output[5])
                else:
                    self.assertIsNone(output[5])

    def test_only_the_embedder_projection_depends_on_the_feature_width(self):
        narrow = build_synthesizer(VOCODER, text_enc_hidden_dim=768).state_dict()
        wide = build_synthesizer(VOCODER, text_enc_hidden_dim=1024).state_dict()
        self.assertEqual(set(narrow), set(wide))
        differing = {
            key for key in narrow if narrow[key].shape != wide[key].shape
        }
        self.assertEqual(differing, {"enc_p.emb_phone.weight"})


class IdentityTest(unittest.TestCase):
    def test_it_is_recognised_from_its_weights(self):
        net = build_synthesizer(VOCODER)
        self.assertEqual(detect_vocoder(net.state_dict()), CODENAME_RINGFORMER)

    def test_a_hifigan_generator_is_still_recognised_as_one(self):
        net = build_synthesizer("HiFi-GAN")
        self.assertEqual(detect_vocoder(net.state_dict()), "HiFi-GAN")

    def test_the_sample_rate_survives_a_legacy_save(self):
        for sample_rate in (32000, 40000, 48000):
            with self.subTest(sample_rate=sample_rate):
                net = build_synthesizer(VOCODER, sample_rate=sample_rate)
                with tempfile.TemporaryDirectory() as folder:
                    path = os.path.join(folder, "G.pth")
                    save_legacy(net, path)
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                from rvc.train.warm_start import normalize_weight_norm_keys

                weights = normalize_weight_norm_keys(checkpoint["model"])
                self.assertEqual(detect_vocoder(weights), CODENAME_RINGFORMER)
                self.assertEqual(
                    infer_codename_ringformer_sample_rate(weights), sample_rate
                )


class WarmStartTest(unittest.TestCase):
    """What a HiFi-GAN pretrain may and may not contribute."""

    def _warm_start_from_hifigan(self, target_dim=768):
        source = build_synthesizer("HiFi-GAN")
        target = build_synthesizer(VOCODER, text_enc_hidden_dim=target_dim)
        before = snapshot(target)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "G.pth")
            save_legacy(source, path, identity("HiFi-GAN"))
            transfer = load_pretrained(
                target, path, "G", verbose=False, target_identity=identity(VOCODER)
            )
        return source, target, before, transfer

    def test_the_encoders_flow_and_speaker_embedding_are_inherited(self):
        source, target, _, transfer = self._warm_start_from_hifigan()
        source_state = source.state_dict()
        for prefix in ("enc_p.", "enc_q.", "flow.", "emb_g."):
            inherited = [key for key in transfer.loaded if key.startswith(prefix)]
            self.assertTrue(inherited, f"nothing inherited under {prefix}")
        for key, value in target.state_dict().items():
            if key.startswith(("enc_q.", "flow.", "emb_g.")):
                self.assertTrue(torch.equal(value, source_state[key]), key)

    def test_conv_pre_and_the_speaker_conditioning_carry_over(self):
        source, target, _, transfer = self._warm_start_from_hifigan()
        source_state, target_state = source.state_dict(), target.state_dict()
        self.assertIn("dec.cond.weight", transfer.loaded)
        self.assertTrue(
            torch.equal(target_state["dec.cond.weight"], source_state["dec.cond.weight"])
        )
        # conv_pre is plain in HiFi-GAN and weight-normed here, so it is converted back to
        # the same effective weight rather than copied.
        magnitude = target_state["dec.conv_pre.parametrizations.weight.original0"]
        direction = target_state["dec.conv_pre.parametrizations.weight.original1"]
        effective = magnitude * direction / direction.norm(dim=(1, 2), keepdim=True)
        self.assertTrue(
            torch.allclose(
                effective, source_state["dec.conv_pre.weight"], atol=1e-5
            )
        )
        self.assertIn(
            "dec.conv_pre.parametrizations.weight.original0", transfer.transformed
        )

    def test_everything_structurally_new_starts_from_scratch(self):
        _, _, before, transfer = self._warm_start_from_hifigan()
        for prefix in (
            "dec.conformers.",
            "dec.alphas.",
            "dec.noise_convs.",
            "dec.noise_res.",
            "dec.ups.",
            "dec.conv_post.",
            "dec.m_source.",
        ):
            fresh = [key for key in transfer.reinitialised if key.startswith(prefix)]
            self.assertTrue(fresh, f"{prefix} should start from scratch")
            self.assertFalse(
                [key for key in transfer.loaded if key.startswith(prefix)], prefix
            )
        # And a HiFi-GAN tensor with no counterpart is reported, not silently ignored.
        self.assertTrue(
            any(key.startswith("dec.noise_convs.") for key in transfer.dropped)
        )

    def test_a_wider_embedder_costs_only_the_projection(self):
        _, _, _, transfer = self._warm_start_from_hifigan(target_dim=1024)
        self.assertIn("enc_p.emb_phone.weight", transfer.reinitialised)
        self.assertNotIn("enc_p.emb_phone.weight", transfer.loaded)
        self.assertIn("enc_p.emb_phone.bias", transfer.loaded)

    def test_a_different_sample_rate_is_refused_for_the_decoder(self):
        source = build_synthesizer("HiFi-GAN", sample_rate=40000)
        target = build_synthesizer(VOCODER, sample_rate=48000)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "G.pth")
            save_legacy(source, path, identity("HiFi-GAN", 40000))
            transfer = load_pretrained(
                target,
                path,
                "G",
                verbose=False,
                target_identity=identity(VOCODER, 48000),
            )
        self.assertNotIn("dec.cond.weight", transfer.loaded)
        self.assertTrue(any("40000" in note for note in transfer.notes))

    def test_it_round_trips_back_into_a_hifigan_model(self):
        source = build_synthesizer(VOCODER)
        target = build_synthesizer("HiFi-GAN")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "G.pth")
            save_legacy(source, path, identity(VOCODER))
            transfer = load_pretrained(
                target, path, "G", verbose=False, target_identity=identity("HiFi-GAN")
            )
        self.assertIn("dec.conv_pre.weight", transfer.loaded)
        self.assertIn("dec.cond.weight", transfer.loaded)


class DiscriminatorTest(unittest.TestCase):
    def test_the_layout_is_the_codename_one(self):
        layout = discriminator_layout("codename-ringformer")
        self.assertEqual(layout[0], ("S",))
        self.assertEqual(
            [descriptor[1] for descriptor in layout if descriptor[0] == "P"],
            [2, 3, 5, 7, 11, 17, 23, 37],
        )
        resolutions = [descriptor for descriptor in layout if descriptor[0] == "R"]
        self.assertEqual(len(resolutions), 3)
        for descriptor in resolutions:
            self.assertEqual(descriptor[2], "hann")

    def test_the_existing_versions_keep_the_rectangular_window(self):
        for version in ("v1", "v2", "v3"):
            self.assertNotIn("window", DISCRIMINATOR_VERSIONS[version])
            for descriptor in discriminator_layout(version):
                if descriptor[0] == "R":
                    self.assertEqual(descriptor[2], DEFAULT_RESOLUTION_WINDOW)
        net = MultiPeriodDiscriminator(False, version="v3")
        for sub in net.discriminators:
            descriptor = describe_discriminator(sub)
            if descriptor[0] == "R":
                self.assertEqual(sub.window, "ones")

    def test_a_v2_pretrain_gives_the_scale_and_every_period(self):
        source = MultiPeriodDiscriminator(False, version="v2")
        target = MultiPeriodDiscriminator(False, version="codename-ringformer")
        before = snapshot(target)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "D.pth")
            save_legacy(source, path)
            transfer = load_pretrained(target, path, "D", verbose=False)

        source_state, target_state = source.state_dict(), target.state_dict()
        inherited = 0
        for index, sub in enumerate(target.discriminators):
            keys = [
                key
                for key in target_state
                if key.startswith(f"discriminators.{index}.")
            ]
            if describe_discriminator(sub)[0] == "R":
                for key in keys:
                    self.assertIn(key, transfer.reinitialised)
                    self.assertTrue(torch.equal(target_state[key], before[key]), key)
            else:
                for key in keys:
                    self.assertIn(key, transfer.loaded)
                    self.assertTrue(torch.equal(target_state[key], source_state[key]))
                inherited += 1
        self.assertEqual(inherited, 9)  # the scale discriminator plus eight periods
        self.assertFalse(transfer.dropped)  # v2 has nothing this layout cannot use


class ExportRoundTripTest(unittest.TestCase):
    def _export(self, folder, dim):
        config = stock(48000, VOCODER)
        net = build_synthesizer(VOCODER, text_enc_hidden_dim=dim)
        # extract_model reads the run's model_info.json and swallows its own exceptions,
        # so without one the only symptom is a file that never appears.
        with open(os.path.join(folder, "model_info.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_dataset_duration": 12.3,
                    "embedder_model": "contentvec",
                    "embedder_feature_scale": 1.0,
                    "embedder_output_layer": 0,
                    "embedder_input_std_floor": 0.01,
                    "speakers_id": 1,
                },
                f,
            )
        hps = HParams(
            **dict(
                config,
                model=dict(
                    config["model"], spk_embed_dim=1, text_enc_hidden_dim=dim
                ),
            )
        )
        path = os.path.join(folder, "exported.pth")
        extract_model(
            ckpt=net.state_dict(),
            sr=48000,
            name="test",
            model_path=path,
            epoch=1,
            step=1,
            hps=hps,
            overtrain_info=None,
            vocoder=VOCODER,
        )
        self.assertTrue(os.path.isfile(path), "extract_model produced no file")
        return torch.load(path, map_location="cpu", weights_only=True)

    def test_it_exports_loads_and_infers(self):
        for dim in (768, 1024):
            with self.subTest(dim=dim):
                with tempfile.TemporaryDirectory() as folder:
                    cpt = self._export(folder, dim)
                self.assertEqual(cpt["vocoder"], VOCODER)
                model = stock(48000, VOCODER)["model"]
                self.assertEqual(
                    checkpoint_gen_istft(cpt),
                    (model["gen_istft_n_fft"], model["gen_istft_hop_size"]),
                )
                self.assertEqual(checkpoint_text_enc_hidden_dim(cpt), dim)

                # Exactly how rvc/infer/infer.py rebuilds it.
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
                n_fft, hop = checkpoint_gen_istft(cpt)
                net = Synthesizer(
                    *cpt["config"],
                    use_f0=cpt.get("f0", 1),
                    text_enc_hidden_dim=checkpoint_text_enc_hidden_dim(cpt),
                    vocoder=cpt["vocoder"],
                    gen_istft_n_fft=n_fft,
                    gen_istft_hop_size=hop,
                )
                del net.enc_q
                missing, unexpected = net.load_state_dict(
                    cpt["weight"], strict=False
                )
                self.assertEqual([key for key in missing if "enc_q" not in key], [])
                self.assertEqual(unexpected, [])

                frames = 40
                waveform, _, _ = net.infer(
                    torch.randn(1, frames, dim),
                    torch.tensor([frames]),
                    torch.randint(1, 255, (1, frames)),
                    torch.rand(1, frames) * 200 + 100,
                    torch.zeros(1, dtype=torch.long),
                )
                self.assertEqual(waveform.shape, (1, 1, frames * 480))
                self.assertTrue(torch.isfinite(waveform).all())

    def test_the_istft_settings_are_recoverable_without_the_recorded_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            cpt = self._export(folder, 768)
        del cpt["gen_istft_hop_size"]
        self.assertEqual(checkpoint_gen_istft(cpt), (120, 30))

    def test_another_vocoder_reports_no_istft_settings(self):
        self.assertEqual(
            checkpoint_gen_istft({"weight": {"dec.ups.0.weight_v": torch.zeros(4)}}),
            (None, None),
        )


class ConfigResolutionTest(unittest.TestCase):
    """A run's config.json has to describe the decoder its vocoder actually builds."""

    def _seed(self, folder):
        path = os.path.join(folder, "config.json")
        shutil.copyfile(REPO / "rvc" / "configs" / "48000.json", path)
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
        config["train"]["learning_rate"] = 5e-5  # a hand edit that must survive
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        return path

    def test_the_decoder_keys_are_written_and_taken_back_out_again(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._seed(folder)

            resolve_vocoder_model_config(folder, 48000, VOCODER)
            with open(path, encoding="utf-8") as f:
                after = json.load(f)
            self.assertEqual(after["model"]["upsample_rates"], [4, 4])
            self.assertEqual(after["model"]["upsample_kernel_sizes"], [8, 8])
            self.assertEqual(after["model"]["gen_istft_n_fft"], 120)
            self.assertEqual(after["model"]["gen_istft_hop_size"], 30)
            self.assertEqual(after["train"]["learning_rate"], 5e-5)

            # Pointing the same folder back at another vocoder must leave no trace: the
            # stock rates return and the two iSTFT keys are removed, not left behind.
            resolve_vocoder_model_config(folder, 48000, "HiFi-GAN")
            with open(path, encoding="utf-8") as f:
                back = json.load(f)
            stock_model = stock(48000)["model"]
            self.assertEqual(
                back["model"]["upsample_rates"], stock_model["upsample_rates"]
            )
            self.assertEqual(
                back["model"]["upsample_kernel_sizes"],
                stock_model["upsample_kernel_sizes"],
            )
            self.assertNotIn("gen_istft_n_fft", back["model"])
            self.assertNotIn("gen_istft_hop_size", back["model"])
            self.assertEqual(back["train"]["learning_rate"], 5e-5)

    def test_every_rate_multiplies_out_to_the_hop_length(self):
        for sample_rate in (32000, 40000, 48000):
            settings = vocoder_model_config(VOCODER, sample_rate)
            hop_length = sample_rate // 100
            self.assertEqual(
                math.prod(settings["upsample_rates"]) * settings["gen_istft_hop_size"],
                hop_length,
            )
            self.assertEqual(
                settings["gen_istft_n_fft"], settings["gen_istft_hop_size"] * 4
            )

    def test_the_other_vocoders_get_the_stock_decoder(self):
        for vocoder in ("HiFi-GAN", "RefineGAN", "SiFi-GAN"):
            settings = vocoder_model_config(vocoder, 48000)
            self.assertEqual(
                settings["upsample_rates"], stock(48000)["model"]["upsample_rates"]
            )
            self.assertNotIn("gen_istft_n_fft", settings)


class ResumeGuardTest(unittest.TestCase):
    def test_resuming_into_a_different_vocoder_is_refused(self):
        net = build_synthesizer("HiFi-GAN")
        with tempfile.TemporaryDirectory() as folder:
            save_legacy(net, os.path.join(folder, "G_100.pth"), identity("HiFi-GAN"))
            with self.assertRaises(SystemExit):
                assert_resumable(folder, {}, identity(VOCODER))

    def test_resuming_into_the_same_vocoder_is_allowed(self):
        net = build_synthesizer(VOCODER)
        with tempfile.TemporaryDirectory() as folder:
            save_legacy(net, os.path.join(folder, "G_100.pth"), identity(VOCODER))
            assert_resumable(folder, {}, identity(VOCODER))


class TrainingStepTest(unittest.TestCase):
    """One real step with the discriminator, every loss and the spectral loss."""

    def test_a_step_runs_and_the_conformers_learn(self):
        torch.manual_seed(0)
        config = stock(48000, VOCODER)
        model = config["model"]
        net_g = build_synthesizer(VOCODER)
        net_d = MultiPeriodDiscriminator(
            model["use_spectral_norm"], version="codename-ringformer"
        )
        fn_spectral_loss = SpectralDistanceLoss(
            n_fft=model["gen_istft_n_fft"], hop_size=model["gen_istft_hop_size"]
        )
        optim_g = torch.optim.AdamW(net_g.parameters(), 1e-4)
        optim_d = torch.optim.AdamW(net_d.parameters(), 1e-4)

        batch = fake_batch(config)
        segment = config["train"]["segment_size"]
        wave = torch.randn(2, 1, segment) * 0.1

        (
            y_hat,
            ids_slice,
            _,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            dec_extra,
        ) = net_g(
            batch["phone"],
            batch["phone_lengths"],
            batch["pitch"],
            batch["pitchf"],
            batch["spec"],
            batch["phone_lengths"],
            batch["sid"],
        )
        self.assertEqual(y_hat.shape, wave.shape)

        y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
        loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad()
        loss_disc.backward()
        optim_d.step()

        _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
        mel = spec_to_mel_torch(
            batch["spec"],
            config["data"]["filter_length"],
            config["data"]["n_mel_channels"],
            config["data"]["sample_rate"],
            config["data"]["mel_fmin"],
            config["data"]["mel_fmax"],
        )
        y_mel = commons.slice_segments(
            mel, ids_slice, segment // config["data"]["hop_length"], dim=3
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1),
            config["data"]["filter_length"],
            config["data"]["n_mel_channels"],
            config["data"]["sample_rate"],
            config["data"]["hop_length"],
            config["data"]["win_length"],
            config["data"]["mel_fmin"],
            config["data"]["mel_fmax"],
        )
        loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * config["train"][
            "c_mel"
        ]
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config["train"]["c_kl"]
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        magnitude, _ = dec_extra
        loss_sd = (
            fn_spectral_loss(wave, y_hat, magnitude) * config["train"]["c_sd"]
        )
        total = loss_gen + loss_fm + loss_mel + loss_kl + loss_sd

        optim_g.zero_grad()
        total.backward()

        for name, value in (
            ("disc", loss_disc),
            ("gen", loss_gen),
            ("fm", loss_fm),
            ("mel", loss_mel),
            ("kl", loss_kl),
            ("sd", loss_sd),
            ("total", total),
        ):
            self.assertTrue(torch.isfinite(value), f"{name} is not finite")

        # The Conformer stack is what makes this vocoder what it is, so it had better be
        # in the graph rather than sitting beside it.
        for prefix in ("dec.conformers.", "dec.alphas.", "dec.conv_post."):
            grads = [
                parameter.grad
                for name, parameter in net_g.named_parameters()
                if name.startswith(prefix) and parameter.grad is not None
            ]
            self.assertTrue(grads, f"{prefix} received no gradient")
            self.assertGreater(max(g.abs().max().item() for g in grads), 0.0)
        optim_g.step()


if __name__ == "__main__":
    unittest.main()
