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

from rvc.train.reset_run import _pid_running, reset_training_run


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


if __name__ == "__main__":
    unittest.main()
