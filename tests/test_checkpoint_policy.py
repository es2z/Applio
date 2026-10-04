"""The checkpoint policy: what a run keeps beyond the last checkpoint, and its record.

Run from the repository root:
    env\\python.exe -m unittest tests.test_checkpoint_policy -v

The keeping logic is exercised with fake writers (export / save_gd that drop small
files), so every decision - last, every, best, the G/D multiple, eviction, resume - is
checked without a model. One test saves and warm starts a real weights-only checkpoint.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rvc.train.checkpoint_policy import (
    CHECKPOINTS_DIR,
    DEFAULT_POLICY,
    POLICY_KEY,
    STATE_FILE,
    CheckpointKeeper,
    apply_checkpoint_policy,
    baseline_of,
    normalize_policy,
    policy_is_active,
    read_policy,
    remove_checkpoints,
    score_of,
    update_best,
    wants_eval,
    wants_export_every,
    wants_gd_every,
)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def policy(**values):
    return normalize_policy({**values})


class Writers:
    """Stand-ins for extract_model and the weights-only save_checkpoint."""

    def __init__(self):
        self.exports, self.gds = [], []

    def export(self, path):
        Path(path).write_bytes(b"export")
        self.exports.append(path)

    def save_gd(self, g_path, d_path):
        Path(g_path).write_bytes(b"G")
        Path(d_path).write_bytes(b"D")
        self.gds.append((g_path, d_path))


class PolicyTest(unittest.TestCase):
    def test_defaults_and_activity(self):
        self.assertEqual(normalize_policy(None), DEFAULT_POLICY)
        self.assertFalse(policy_is_active(normalize_policy(None)))
        self.assertTrue(policy_is_active(policy(export_every=5)))
        self.assertTrue(policy_is_active(policy(eval_set="x")))

    def test_invalid_values_are_refused(self):
        for bad in ({"export_every": -1}, {"best_n": 1.5}, {"best_criterion": "loud"},
                    {"gd_every": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_policy(bad)
        with self.assertRaises(ValueError):
            normalize_policy({"best_n": 3})  # nothing to rank them with

    def test_unknown_keys_and_none_are_ignored(self):
        self.assertEqual(normalize_policy({"what": 1, "export_every": None}), DEFAULT_POLICY)

    def test_intervals(self):
        p = policy(export_every=5, gd_every=4, eval_set="s", eval_every=0)
        self.assertEqual([e for e in range(1, 41) if wants_export_every(p, e)], [5, 10, 15, 20, 25, 30, 35, 40])
        self.assertEqual([e for e in range(1, 41) if wants_gd_every(p, e)], [20, 40])
        # 0 = with every save.
        self.assertEqual([e for e in range(1, 21) if wants_eval(p, e, 10)], [10, 20])
        p = policy(eval_set="s", eval_every=3)
        self.assertEqual([e for e in range(1, 10) if wants_eval(p, e, 10)], [3, 6, 9])
        self.assertFalse(any(wants_eval(policy(), e, 1) for e in range(1, 5)))
        self.assertFalse(any(wants_gd_every(policy(gd_every=2), e) for e in range(1, 10)))


class ConfigTest(unittest.TestCase):
    def _folder(self, config):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        Path(temp.name, "config.json").write_text(json.dumps(config))
        return temp.name

    def test_defaults_leave_a_config_without_a_policy_untouched(self):
        folder = self._folder({"train": {"learning_rate": 1e-4}})
        before = Path(folder, "config.json").read_bytes()
        quiet(apply_checkpoint_policy, folder, export_every=0, best_n=0, eval_set="",
              best_criterion="combined", best_save_gd=True)
        self.assertEqual(Path(folder, "config.json").read_bytes(), before)
        self.assertEqual(read_policy(folder), DEFAULT_POLICY)

    def test_written_read_back_and_none_keeps_values(self):
        folder = self._folder({"train": {"learning_rate": 1e-4}})
        quiet(apply_checkpoint_policy, folder, export_every=10, gd_every=2, eval_set="tier1", best_n=3)
        self.assertEqual(read_policy(folder)["export_every"], 10)
        quiet(apply_checkpoint_policy, folder, export_every=None, best_n=5)
        written = read_policy(folder)
        self.assertEqual((written["export_every"], written["gd_every"], written["best_n"]), (10, 2, 5))
        config = json.loads(Path(folder, "config.json").read_text())
        self.assertEqual(config["train"], {"learning_rate": 1e-4})
        self.assertIn(POLICY_KEY, config)

    def test_a_missing_config_is_not_created(self):
        with tempfile.TemporaryDirectory() as folder:
            apply_checkpoint_policy(folder, export_every=5)
            self.assertFalse(os.path.exists(os.path.join(folder, "config.json")))


class ScoreTest(unittest.TestCase):
    def test_criteria_and_fallbacks(self):
        scores = {"A_mel": 0.8, "B_mel": 0.9, "B_noise": 1.2, "C_noise": 1.4}
        self.assertEqual(score_of(scores, "mel", None), 0.9)
        self.assertAlmostEqual(score_of(scores, "noise", None), 1.3)
        self.assertEqual(score_of({"A_mel": 0.8, "A_noise": 1.1}, "mel", None), 0.8)
        self.assertEqual(score_of({"A_mel": 0.8, "A_noise": 1.1}, "noise", None), 1.1)
        baseline = baseline_of(scores)
        self.assertAlmostEqual(score_of(scores, "combined", baseline), 2.0)
        better = {"B_mel": 0.45, "B_noise": 0.65, "C_noise": 0.65}
        self.assertAlmostEqual(score_of(better, "combined", baseline), 1.0)
        self.assertIsNone(score_of(scores, "combined", None))
        self.assertIsNone(score_of({}, "mel", None))

    def test_update_best(self):
        best = []
        for epoch, score in ((5, 0.9), (10, 0.7), (15, 0.8), (20, 0.95), (25, 0.6)):
            best, dropped, accepted = update_best(best, {"epoch": epoch, "score": score}, 3)
        self.assertEqual([e["epoch"] for e in best], [25, 10, 15])
        # A tie keeps the earlier epoch.
        best, dropped, accepted = update_best(best, {"epoch": 30, "score": 0.8}, 3)
        self.assertFalse(accepted)
        self.assertEqual([e["epoch"] for e in best], [25, 10, 15])
        # The same epoch again (a resume) replaces the old entry.
        best, dropped, accepted = update_best(best, {"epoch": 15, "score": 0.5}, 3)
        self.assertTrue(accepted)
        self.assertEqual([e["epoch"] for e in best], [15, 25, 10])
        self.assertEqual([e["epoch"] for e in dropped], [15])


class KeeperTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = temp.name
        self.writers = Writers()

    def keeper(self, **values):
        return CheckpointKeeper(self.dir, "m", policy(**values))

    def files(self, kind):
        path = Path(self.dir, CHECKPOINTS_DIR, kind)
        return sorted(p.name for p in path.iterdir()) if path.is_dir() else []

    def test_last_is_replaced(self):
        keeper = self.keeper(export_every=1)
        for epoch in (5, 10):
            keeper.save_last(epoch, epoch * 100, self.writers.export)
        self.assertEqual(self.files("last"), ["m_10e_1000s.pth"])

    def test_every_with_the_gd_multiple(self):
        keeper = self.keeper(export_every=5, gd_every=2)
        for epoch in range(1, 21):
            if wants_export_every(keeper.policy, epoch):
                keeper.save_every(epoch, epoch, self.writers.export, self.writers.save_gd)
        self.assertEqual(self.files("every"), sorted([
            "m_5e_5s.pth", "m_10e_10s.pth", "m_15e_15s.pth", "m_20e_20s.pth",
            "G_10e.pth", "D_10e.pth", "G_20e.pth", "D_20e.pth"]))

    def test_best_keeps_n_and_deletes_what_drops_out(self):
        keeper = self.keeper(eval_set="s", best_n=2, best_criterion="mel")
        ranks = {}
        for epoch, mel in ((5, 0.9), (10, 0.7), (15, 0.8), (20, 0.6)):
            ranks[epoch] = quiet(keeper.record_evaluation, epoch, epoch, {"A_mel": mel},
                                 self.writers.export, self.writers.save_gd)[1]
        self.assertEqual(ranks, {5: 1, 10: 1, 15: 2, 20: 1})
        self.assertEqual(self.files("best"), sorted([
            "m_20e_20s.pth", "G_20e.pth", "D_20e.pth", "m_10e_10s.pth", "G_10e.pth", "D_10e.pth"]))
        self.assertEqual(len(keeper.state["evaluations"]), 4)

    def test_best_without_gd(self):
        keeper = self.keeper(eval_set="s", best_n=1, best_criterion="mel", best_save_gd=False)
        quiet(keeper.record_evaluation, 5, 5, {"A_mel": 0.5}, self.writers.export, self.writers.save_gd)
        self.assertEqual(self.files("best"), ["m_5e_5s.pth"])
        self.assertEqual(self.writers.gds, [])

    def test_combined_is_relative_to_the_first_evaluation(self):
        keeper = self.keeper(eval_set="s", best_n=3)
        first = quiet(keeper.record_evaluation, 5, 5, {"B_mel": 1.0, "B_noise": 2.0},
                      self.writers.export, self.writers.save_gd)[0]
        second = quiet(keeper.record_evaluation, 10, 10, {"B_mel": 0.5, "B_noise": 1.0},
                       self.writers.export, self.writers.save_gd)[0]
        self.assertAlmostEqual(first, 2.0)
        self.assertAlmostEqual(second, 1.0)

    def test_state_survives_a_resume(self):
        keeper = self.keeper(eval_set="s", best_n=2, best_criterion="mel")
        quiet(keeper.record_evaluation, 5, 5, {"A_mel": 0.9}, self.writers.export, self.writers.save_gd)
        quiet(keeper.record_evaluation, 10, 10, {"A_mel": 0.7}, self.writers.export, self.writers.save_gd)
        resumed = self.keeper(eval_set="s", best_n=2, best_criterion="mel")
        self.assertEqual([e["epoch"] for e in resumed.state["best"]], [10, 5])
        quiet(resumed.record_evaluation, 15, 15, {"A_mel": 0.8}, self.writers.export, self.writers.save_gd)
        self.assertEqual([e["epoch"] for e in resumed.state["best"]], [10, 15])
        self.assertNotIn("m_5e_5s.pth", self.files("best"))

    def test_a_best_deleted_by_hand_stops_competing(self):
        keeper = self.keeper(eval_set="s", best_n=2, best_criterion="mel")
        quiet(keeper.record_evaluation, 5, 5, {"A_mel": 0.5}, self.writers.export, self.writers.save_gd)
        os.remove(keeper.state["best"][0]["export"])
        self.assertEqual(self.keeper(eval_set="s", best_n=2, best_criterion="mel").state["best"], [])

    def test_an_unreadable_state_is_set_aside(self):
        keeper = self.keeper(export_every=1)
        keeper.save_state()
        Path(keeper.state_path).write_text("{not json")
        fresh = quiet(self.keeper, export_every=1)
        self.assertEqual(fresh.state["best"], [])
        aside = [p for p in os.listdir(Path(self.dir, CHECKPOINTS_DIR)) if "unreadable" in p]
        self.assertEqual(len(aside), 1)

    def test_remove_checkpoints(self):
        keeper = self.keeper(export_every=1)
        keeper.save_last(1, 1, self.writers.export)
        remove_checkpoints(self.dir)
        self.assertFalse(os.path.exists(Path(self.dir, CHECKPOINTS_DIR)))


class WeightsOnlyCheckpointTest(unittest.TestCase):
    """A kept G/D is weights and stamps only, and a warm start loads it whole."""

    def test_save_and_warm_start(self):
        from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
        from rvc.train.utils import save_checkpoint
        from rvc.train.warm_start import warm_start

        torch.manual_seed(0)
        source = MultiPeriodDiscriminator(False, version="v2")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "D_10e.pth")
            quiet(save_checkpoint, source, None, 1e-4, 10, path, None,
                  {"embedder_model": "contentvec"}, {"vocoder": "HiFi-GAN", "disc_version": "v2"},
                  include_optimizer=False)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertNotIn("optimizer", saved)
            self.assertNotIn("scaler", saved)
            self.assertEqual((saved["iteration"], saved["disc_version"]), (10, "v2"))
            target = MultiPeriodDiscriminator(False, version="v2")
            transfer = quiet(warm_start, target, path, "D")
        self.assertEqual(transfer.reinitialised, {})
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value), key)


class ResetArchivesCheckpointsTest(unittest.TestCase):
    def test_the_reset_moves_checkpoints_into_the_archive(self):
        from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
        from rvc.train.reset_run import reset_training_run
        from rvc.train.utils import save_checkpoint

        from tests.test_codename_ringformer import build_synthesizer, stock

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            root.mkdir()
            config = stock(48000)
            config["model"]["text_enc_hidden_dim"] = 768
            (root / "config.json").write_text(json.dumps(config))
            identity = {"vocoder": "HiFi-GAN", "sample_rate": 48000, "disc_version": "v2"}
            for tag, net in (("G", quiet(build_synthesizer, "HiFi-GAN")),
                             ("D", MultiPeriodDiscriminator(False, version="v2"))):
                quiet(save_checkpoint, net, torch.optim.AdamW(net.parameters()), 1e-4, 50,
                      str(root / f"{tag}_2333333.pth"), torch.amp.GradScaler(enabled=False),
                      architecture_identity=identity)
            keeper = CheckpointKeeper(str(root), "run", policy(export_every=1))
            keeper.save_last(50, 500, Writers().export)
            archive = quiet(reset_training_run, root, vocoder="HiFi-GAN")
            self.assertFalse((root / CHECKPOINTS_DIR).exists())
            self.assertTrue((archive / CHECKPOINTS_DIR / STATE_FILE).is_file())
            self.assertTrue((root / "G_0.pth").is_file())


if __name__ == "__main__":
    unittest.main()
