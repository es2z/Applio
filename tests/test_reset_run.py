import json
import pickle
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.utils import embedder_identity_from_model_info
from rvc.train.reset_run import _pid_running, reset_training_run
from rvc.train.utils import (
    assert_resumable,
    describe_architecture_mismatch,
    save_checkpoint,
)
from rvc.train.warm_start import detect_vocoder


class ProcessCheckTests(unittest.TestCase):
    def test_japanese_tasklist_bytes(self):
        for output, expected in (
            ("情報: 指定されたタスクは実行されていません。".encode("cp932"), False),
            ('"日本語.exe","123","Console","1","1,024 K"\r\n'.encode("cp932"), True),
            (b'"python.exe","1234","Console","1","1,024 K"\r\n', False),
        ):
            with self.subTest(output=output), patch("rvc.train.reset_run.os.name", "nt"), patch(
                "rvc.train.reset_run.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, stdout=output),
            ) as run:
                self.assertEqual(_pid_running(123), expected)
                self.assertNotIn("text", run.call_args.kwargs)


class ResetRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "branch"
        self.root.mkdir()
        self.config = {"train": {"learning_rate": 0.00003, "betas": [0.8, 0.99], "eps": 1e-9, "c_mel": 60}}
        (self.root / "config.json").write_text(json.dumps(self.config))
        (self.root / "filelist.txt").write_text("../source/audio.wav|../source/feature.npy")
        (self.root / "training_data.json").write_text('{"loss_gen_history": [123]}')
        (self.root / "D_2333333.pth.old").write_bytes(b"damaged backup")
        self.model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        self.model(torch.ones(1, 2)).sum().backward()
        opt.step()
        for tag in ("G", "D"):
            torch.save({"model": self.model.state_dict(), "optimizer": opt.state_dict(),
                        "iteration": 1740, "learning_rate": 0.0001,
                        "scaler": {"scale": 65536.}, "vocoder": "test"}, self.root / f"{tag}_2333333.pth")
        with SummaryWriter(str(self.root / "eval")) as writer:
            writer.add_scalar("loss", 123, 17400)

    def test_reset_weights_lr_history_and_tensorboard(self):
        filelist = (self.root / "filelist.txt").read_bytes()
        archive = reset_training_run(self.root)
        self.assertEqual(filelist, (self.root / "filelist.txt").read_bytes())
        self.assertEqual((self.root / "D_2333333.pth.old").read_bytes(), b"damaged backup")
        self.assertFalse((self.root / "training_data.json").exists())
        self.assertTrue((archive / "training_data.json").exists())
        for tag in ("G", "D"):
            state = torch.load(self.root / f"{tag}_0.pth", weights_only=True)
            self.assertEqual(state["iteration"], 0)
            self.assertEqual(state["scaler"], {})
            self.assertEqual(state["vocoder"], "test")
            for key, tensor in self.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, state["model"][key]))
            fresh_model = torch.nn.Linear(2, 1)
            fresh_model.load_state_dict(state["model"])
            opt = torch.optim.AdamW(fresh_model.parameters(), lr=1)
            opt.load_state_dict(state["optimizer"])
            self.assertEqual(len(opt.state), 0)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99, last_epoch=-1)
            self.assertEqual(opt.param_groups[0]["lr"], 0.00003)
            fresh_model(torch.ones(1, 2)).sum().backward()
            opt.step()
            scheduler.step()
            self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.00003 * 0.99)
        with SummaryWriter(str(self.root / "eval")) as writer:
            writer.add_scalar("loss", 5, 0)
        old = EventAccumulator(str(archive / "eval")).Reload().Scalars("loss")
        new = EventAccumulator(str(self.root / "eval")).Reload().Scalars("loss")
        self.assertEqual([e.step for e in old], [17400])
        self.assertEqual([e.step for e in new], [0])

    def test_corrupt_checkpoint_leaves_run_untouched(self):
        (self.root / "D_2333333.pth").write_bytes(b"corrupt")
        before = (self.root / "G_2333333.pth").read_bytes()
        with self.assertRaises(pickle.UnpicklingError):
            reset_training_run(self.root)
        self.assertEqual(before, (self.root / "G_2333333.pth").read_bytes())
        self.assertTrue((self.root / "eval").is_dir())
        self.assertTrue((self.root / "training_data.json").exists())

    def test_mismatched_epochs_start_a_new_run(self):
        path = self.root / "D_2333333.pth"
        state = torch.load(path, weights_only=True)
        state["iteration"] = 1700
        torch.save(state, path)
        archive = reset_training_run(self.root)
        provenance = json.loads((archive / "reset.json").read_text())
        self.assertEqual(provenance["source_epochs"], {"G": 1740, "D": 1700})
        for tag in ("G", "D"):
            state = torch.load(self.root / f"{tag}_0.pth", weights_only=True)
            self.assertEqual(state["iteration"], 0)
            self.assertEqual(state["optimizer"]["state"], {})
            self.assertTrue((archive / f"{tag}_2333333.pth").exists())

    def test_rollback_on_install_failure(self):
        rename = Path.rename

        def fail(source, destination):
            if source.name == "D_0.staged":
                raise OSError("simulated install failure")
            return rename(source, destination)

        with patch.object(Path, "rename", fail), self.assertRaises(OSError):
            reset_training_run(self.root)
        self.assertTrue((self.root / "G_2333333.pth").exists())
        self.assertTrue((self.root / "D_2333333.pth").exists())
        self.assertTrue((self.root / "eval").exists())
        self.assertFalse((self.root / "G_0.pth").exists())

    def test_running_training_cannot_be_reset(self):
        self.config["process_pids"] = [123]
        (self.root / "config.json").write_text(json.dumps(self.config))
        with patch("rvc.train.reset_run._pid_running", return_value=True), self.assertRaisesRegex(ValueError, "Stop the existing"):
            reset_training_run(self.root)
        self.assertTrue((self.root / "G_2333333.pth").exists())


class ResetAfterAnEmbedderChangeTests(unittest.TestCase):
    """Re-extracting a folder with another embedder leaves its G_*.pth behind.

    Nothing in the shapes says that G's enc_p.emb_phone.weight was fitted to a feature
    space this run no longer produces, so a reset that only restarts the schedule hands
    assert_resumable a checkpoint it is right to refuse. The reset rebuilds that one
    tensor and re-stamps both files instead.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "branch"
        self.root.mkdir()
        (self.root / "config.json").write_text(json.dumps({
            "model": {"text_enc_hidden_dim": 1024, "use_spectral_norm": False},
            "train": {"learning_rate": 0.0001, "betas": [0.8, 0.99], "eps": 1e-9},
        }))
        projection = torch.nn.Linear(1024, 192)
        self.generator = {
            "enc_p.emb_phone.weight": projection.weight.detach().clone(),
            "enc_p.emb_phone.bias": projection.bias.detach().clone(),
            "emb_g.weight": torch.randn(109, 256),
        }
        self.discriminator = {"convs.0.bias": torch.randn(32)}
        optimizer = torch.optim.AdamW(projection.parameters(), lr=0.0001).state_dict()
        for tag, weights in (("G", self.generator), ("D", self.discriminator)):
            torch.save({
                "model": weights, "optimizer": optimizer, "iteration": 840,
                "learning_rate": 0.0001, "scaler": {"scale": 65536.},
                "vocoder": "CodenameRingFormer", "sample_rate": 48000,
                **self.stamp("kushinada-hubert-large"),
            }, self.root / f"{tag}_2333333.pth")

    @staticmethod
    def stamp(name, dim=1024):
        return {
            "embedder_model": name,
            "embedder_feature_scale": 1.0,
            "embedder_output_layer": None,
            "embedder_dim": dim,
            "embedder_input_std_floor": 0.01,
        }

    def write_model_info(self, name, dim=1024):
        info = {"speakers_id": 1, **self.stamp(name, dim)}
        (self.root / "model_info.json").write_text(json.dumps(info))
        return embedder_identity_from_model_info(info)

    def test_the_projection_is_rebuilt_and_both_files_restamped(self):
        target = self.write_model_info("japanese-hubert-large")
        archive = reset_training_run(self.root)

        generator = torch.load(self.root / "G_0.pth", weights_only=True)
        weight = generator["model"]["enc_p.emb_phone.weight"]
        self.assertEqual(weight.shape, self.generator["enc_p.emb_phone.weight"].shape)
        self.assertFalse(torch.equal(weight, self.generator["enc_p.emb_phone.weight"]))
        # Only the weight reads the embedder's feature space.
        for key in ("enc_p.emb_phone.bias", "emb_g.weight"):
            self.assertTrue(torch.equal(generator["model"][key], self.generator[key]), key)
        for tag in ("G", "D"):
            stamped = torch.load(self.root / f"{tag}_0.pth", weights_only=True)
            for key, value in target.items():
                self.assertEqual(stamped[key], value, f"{tag} {key}")
            self.assertEqual(stamped["vocoder"], "CodenameRingFormer")

        transfer = json.loads((archive / "reset.json").read_text())["embedder_transfer"]
        self.assertIn("kushinada-hubert-large", transfer)
        self.assertIn("japanese-hubert-large", transfer)
        # The point of all of it: the run that follows is no longer refused.
        assert_resumable(str(self.root), target, {"vocoder": "CodenameRingFormer"})

    def test_the_same_embedder_leaves_every_weight_alone(self):
        self.write_model_info("kushinada-hubert-large")
        archive = reset_training_run(self.root)
        generator = torch.load(self.root / "G_0.pth", weights_only=True)
        for key, value in self.generator.items():
            self.assertTrue(torch.equal(generator["model"][key], value), key)
        self.assertIsNone(
            json.loads((archive / "reset.json").read_text())["embedder_transfer"]
        )

    def test_a_narrower_embedder_rebuilds_the_projection_at_the_config_width(self):
        (self.root / "config.json").write_text(json.dumps({
            "model": {"text_enc_hidden_dim": 768},
            "train": {"learning_rate": 0.0001, "betas": [0.8, 0.99], "eps": 1e-9},
        }))
        self.write_model_info("contentvec", dim=768)
        reset_training_run(self.root)
        generator = torch.load(self.root / "G_0.pth", weights_only=True)
        self.assertEqual(
            tuple(generator["model"]["enc_p.emb_phone.weight"].shape), (192, 768)
        )

    def test_a_folder_that_records_no_embedder_is_left_alone(self):
        archive = reset_training_run(self.root)
        generator = torch.load(self.root / "G_0.pth", weights_only=True)
        for key, value in self.generator.items():
            self.assertTrue(torch.equal(generator["model"][key], value), key)
        self.assertEqual(generator["embedder_model"], "kushinada-hubert-large")
        self.assertIsNone(
            json.loads((archive / "reset.json").read_text())["embedder_transfer"]
        )


class ResetIntoAnotherVocoderTests(unittest.TestCase):
    """Reset is the only place a vocoder change can happen for an existing folder.

    train.py resumes from the G_0.pth / D_0.pth that reset installs rather than warm
    starting, so without this the reset hands assert_resumable a generator whose decoder
    belongs to the vocoder the folder used to train.
    """

    SOURCE = "HiFi-GAN"
    TARGET = "MRF HiFi-GAN"

    @staticmethod
    def quiet(fn, *args, **kwargs):
        import contextlib
        import io as _io

        with contextlib.redirect_stdout(_io.StringIO()):
            return fn(*args, **kwargs)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "branch"
        self.root.mkdir()
        with open("rvc/configs/48000.json", encoding="utf-8") as f:
            stock = json.load(f)
        stock["model"]["text_enc_hidden_dim"] = 1024
        (self.root / "config.json").write_text(json.dumps(stock))
        self.stock = stock
        self.stamp = {
            "embedder_model": "kushinada-hubert-large",
            "embedder_feature_scale": 1.0,
            "embedder_output_layer": None,
            "embedder_dim": 1024,
            "embedder_input_std_floor": 0.01,
        }
        (self.root / "model_info.json").write_text(
            json.dumps({"speakers_id": 1, **self.stamp})
        )

        net_g = self.quiet(
            Synthesizer,
            stock["data"]["filter_length"] // 2 + 1,
            stock["train"]["segment_size"] // stock["data"]["hop_length"],
            **stock["model"],
            use_f0=True,
            sr=48000,
            vocoder=self.SOURCE,
        )
        net_d = MultiPeriodDiscriminator(
            stock["model"]["use_spectral_norm"], version="v2"
        )
        identity = {
            "vocoder": self.SOURCE,
            "sample_rate": 48000,
            "disc_version": "v2",
            **self.stamp,
        }
        for tag, net in (("G", net_g), ("D", net_d)):
            self.quiet(
                save_checkpoint,
                net,
                torch.optim.AdamW(net.parameters(), lr=1e-4),
                1e-4,
                840,
                str(self.root / f"{tag}_2333333.pth"),
                torch.amp.GradScaler(enabled=False),
                architecture_identity=identity,
            )
        self.source_generator = {
            key: value.detach().clone() for key, value in net_g.state_dict().items()
        }
        del net_g, net_d

    def _installed(self, tag):
        return torch.load(
            self.root / f"{tag}_0.pth", map_location="cpu", weights_only=True
        )

    def test_the_generator_is_rebuilt_and_the_next_run_can_resume(self):
        archive = self.quiet(reset_training_run, self.root, vocoder=self.TARGET)
        generator = self._installed("G")
        self.assertEqual(generator["vocoder"], self.TARGET)
        self.assertEqual(generator["sample_rate"], 48000)
        self.assertEqual(generator["disc_version"], "v2")
        self.assertEqual(detect_vocoder(generator["model"]), self.TARGET)
        self.assertEqual(generator["iteration"], 0)
        self.assertEqual(generator["optimizer"]["state"], {})
        # The decoder really is the MRF one, and the parts that mean the same thing in
        # both came across rather than starting from scratch.
        self.assertIn("dec.mrfs.0.0.layers.0.conv1.weight_v", generator["model"])
        self.assertNotIn("dec.resblocks.0.convs1.0.weight_v", generator["model"])
        for key in ("enc_p.emb_phone.weight", "flow.flows.0.enc.in_layers.0.bias"):
            self.assertTrue(
                torch.equal(generator["model"][key], self.source_generator[key]), key
            )
        self.assertTrue(
            torch.equal(
                generator["model"]["dec.mrfs.0.0.layers.0.conv1.weight_v"],
                self.source_generator["dec.resblocks.0.convs1.0.parametrizations.weight.original1"],
            )
        )
        # assert_resumable is the thing that refused before.
        self.assertIsNone(
            describe_architecture_mismatch(
                generator,
                {"vocoder": self.TARGET, "sample_rate": 48000, "disc_version": "v2"},
            )
        )
        self.quiet(
            assert_resumable,
            str(self.root),
            self.stamp,
            {"vocoder": self.TARGET, "sample_rate": 48000, "disc_version": "v2"},
        )
        self.assertIsNotNone(
            json.loads((archive / "reset.json").read_text())["generator_transfer"]
        )

    def test_the_discriminator_follows_the_new_vocoder(self):
        self.quiet(reset_training_run, self.root, vocoder=self.TARGET)
        discriminator = self._installed("D")
        self.assertEqual(discriminator["disc_version"], "v2")
        self.assertEqual(discriminator["iteration"], 0)

    def test_the_same_vocoder_leaves_the_generator_alone(self):
        archive = self.quiet(reset_training_run, self.root, vocoder=self.SOURCE)
        generator = self._installed("G")
        self.assertEqual(generator["vocoder"], self.SOURCE)
        for key, value in self.source_generator.items():
            saved = generator["model"].get(
                key.replace(".parametrizations.weight.original0", ".weight_g").replace(
                    ".parametrizations.weight.original1", ".weight_v"
                )
            )
            self.assertIsNotNone(saved, key)
            self.assertTrue(torch.equal(saved, value), key)
        self.assertIsNone(
            json.loads((archive / "reset.json").read_text())["generator_transfer"]
        )

    def test_no_vocoder_given_behaves_as_before(self):
        archive = self.quiet(reset_training_run, self.root)
        self.assertEqual(self._installed("G")["vocoder"], self.SOURCE)
        self.assertIsNone(
            json.loads((archive / "reset.json").read_text())["generator_transfer"]
        )


if __name__ == "__main__":
    unittest.main()
