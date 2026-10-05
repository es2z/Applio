"""The checkpoint policy: what a run keeps beyond the latest checkpoints, and its record.

Run from the repository root:
    env\\python.exe -m unittest tests.test_checkpoint_policy -v

The keeping logic is exercised with fake writers (export / save_gd that drop small
files), so every decision - last, the G/D every, the best with separate counts,
eviction, resume - is checked without a model. One test saves and warm starts a real
weights-only checkpoint, one runs a real training reset.
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
    best_is_active,
    normalize_policy,
    policy_from_ui,
    policy_is_active,
    rank_best,
    read_policy,
    remove_checkpoints,
    score_of,
    ui_from_policy,
    wants_eval,
    wants_gd_every,
)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def policy(**values):
    return normalize_policy(values)


def ui(**overrides):
    """policy_from_ui with everything off, as a fresh Training tab shows it."""
    values = dict(gd_keep=False, gd_n=1, best_enabled=False, best_keep_export=True,
                  best_keep_gd=True, best_n=3, best_split=False, best_export_n=3,
                  best_gd_n=3, eval_timing="save", best_criterion="combined", eval_set=None)
    values.update(overrides)
    return policy_from_ui(**values)


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
        self.assertTrue(policy_is_active(policy(gd_every=2)))
        self.assertTrue(policy_is_active(policy(best_gd_n=1, eval_set="s")))
        self.assertFalse(best_is_active(policy(gd_every=2, eval_set="s")))

    def test_invalid_values_are_refused(self):
        for bad in ({"gd_every": -1}, {"best_export_n": 1.5}, {"best_criterion": "loud"},
                    {"gd_every": True}, {"eval_every": 5}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_policy(bad)
        with self.assertRaises(ValueError):
            normalize_policy({"best_export_n": 3})  # nothing to rank them with

    def test_unknown_keys_and_none_are_ignored(self):
        self.assertEqual(
            normalize_policy({"export_every": 5, "best_n": 3, "gd_every": None}), DEFAULT_POLICY
        )

    def test_intervals_follow_the_save_interval(self):
        p = policy(gd_every=4)
        self.assertEqual([e for e in range(1, 81) if wants_gd_every(p, e, 10)], [40, 80])
        self.assertFalse(any(wants_gd_every(policy(), e, 10) for e in range(1, 50)))
        p = policy(best_export_n=1, eval_set="s")
        self.assertEqual([e for e in range(1, 21) if wants_eval(p, e, 10)], [10, 20])
        p = policy(best_export_n=1, eval_set="s", eval_every=1)
        self.assertEqual([e for e in range(1, 4) if wants_eval(p, e, 10)], [1, 2, 3])
        # No best, no evaluation, even with a set named.
        self.assertFalse(any(wants_eval(policy(eval_set="s"), e, 1) for e in range(1, 5)))


class UiConversionTest(unittest.TestCase):
    """What the Training tab's boxes mean; a hidden control's value never counts."""

    def test_everything_off(self):
        self.assertEqual(normalize_policy(ui()), DEFAULT_POLICY)

    def test_gd_keep(self):
        self.assertEqual(ui(gd_keep=True, gd_n=4)["gd_every"], 4)
        self.assertEqual(ui(gd_keep=False, gd_n=4)["gd_every"], 0)

    def test_best_shared_and_split_counts(self):
        shared = ui(best_enabled=True, best_n=5, eval_set="s")
        self.assertEqual((shared["best_export_n"], shared["best_gd_n"]), (5, 5))
        split = ui(best_enabled=True, best_split=True, best_n=9, best_export_n=4,
                   best_gd_n=2, eval_set="s")
        self.assertEqual((split["best_export_n"], split["best_gd_n"]), (4, 2))
        only_gd = ui(best_enabled=True, best_keep_export=False, best_n=3, eval_set="s")
        self.assertEqual((only_gd["best_export_n"], only_gd["best_gd_n"]), (0, 3))

    def test_best_off_or_nothing_kept_means_no_evaluation(self):
        for values in (dict(best_enabled=False), dict(best_enabled=True,
                       best_keep_export=False, best_keep_gd=False)):
            converted = ui(eval_set="s", **values)
            self.assertEqual((converted["best_export_n"], converted["best_gd_n"],
                              converted["eval_set"]), (0, 0, ""))

    def test_timing(self):
        self.assertEqual(ui(eval_timing="epoch")["eval_every"], 1)
        self.assertEqual(ui(eval_timing="save")["eval_every"], 0)

    def test_round_trip(self):
        for values in (
            dict(),
            dict(gd_keep=True, gd_n=3),
            dict(best_enabled=True, best_n=4, eval_set="s", eval_timing="epoch"),
            dict(best_enabled=True, best_split=True, best_export_n=5, best_gd_n=2, eval_set="s"),
            dict(best_enabled=True, best_keep_export=False, best_n=2, eval_set="s",
                 best_criterion="noise"),
        ):
            with self.subTest(values=values):
                first = normalize_policy(ui(**values))
                shown = ui_from_policy(first)
                self.assertEqual(normalize_policy(policy_from_ui(**shown)), first)


class ConfigTest(unittest.TestCase):
    def _folder(self, config):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        Path(temp.name, "config.json").write_text(json.dumps(config))
        return temp.name

    def test_defaults_leave_a_config_without_a_policy_untouched(self):
        folder = self._folder({"train": {"learning_rate": 1e-4}})
        before = Path(folder, "config.json").read_bytes()
        quiet(apply_checkpoint_policy, folder, **ui())
        self.assertEqual(Path(folder, "config.json").read_bytes(), before)
        self.assertEqual(read_policy(folder), DEFAULT_POLICY)

    def test_written_read_back_and_none_keeps_values(self):
        folder = self._folder({"train": {"learning_rate": 1e-4}})
        quiet(apply_checkpoint_policy, folder, gd_every=2, eval_set="tier1", best_export_n=3)
        quiet(apply_checkpoint_policy, folder, gd_every=None, best_gd_n=5)
        written = read_policy(folder)
        self.assertEqual((written["gd_every"], written["best_export_n"], written["best_gd_n"]), (2, 3, 5))
        config = json.loads(Path(folder, "config.json").read_text())
        self.assertEqual(config["train"], {"learning_rate": 1e-4})
        self.assertIn(POLICY_KEY, config)

    def test_a_missing_config_is_not_created(self):
        with tempfile.TemporaryDirectory() as folder:
            apply_checkpoint_policy(folder, gd_every=5)
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

    def test_ranking(self):
        best = []
        for epoch, score in ((5, 0.9), (10, 0.7), (15, 0.8), (20, 0.95), (25, 0.6)):
            best, left = rank_best(best, {"epoch": epoch, "score": score}, 3)
        self.assertEqual([e["epoch"] for e in best], [25, 10, 15])
        best, left = rank_best(best, {"epoch": 30, "score": 0.8}, 3)  # a tie keeps the earlier
        self.assertEqual([e["epoch"] for e in best], [25, 10, 15])
        best, left = rank_best(best, {"epoch": 15, "score": 0.5}, 3)  # a resume replaces
        self.assertEqual([e["epoch"] for e in best], [15, 25, 10])
        self.assertEqual([e["epoch"] for e in left], [15])


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

    def evaluate(self, keeper, epoch, mel):
        return quiet(keeper.record_evaluation, epoch, epoch, {"A_mel": mel},
                     self.writers.export, self.writers.save_gd)

    def test_last_is_replaced(self):
        keeper = self.keeper(gd_every=1)
        for epoch in (5, 10):
            keeper.save_last(epoch, epoch * 100, self.writers.export)
        self.assertEqual(self.files("last"), ["m_10e_1000s.pth"])

    def test_gd_every(self):
        keeper = self.keeper(gd_every=2)
        for epoch in range(5, 21, 5):
            if wants_gd_every(keeper.policy, epoch, 5):
                keeper.save_gd_every(epoch, epoch, self.writers.save_gd)
        self.assertEqual(self.files("every"), ["D_10e.pth", "D_20e.pth", "G_10e.pth", "G_20e.pth"])
        self.assertEqual(self.writers.exports, [])

    def test_best_with_separate_counts(self):
        # Three RVC models but one G/D: an entry keeps its RVC model while it is in the
        # top three and its G/D only while it is first.
        keeper = self.keeper(eval_set="s", best_export_n=3, best_gd_n=1, best_criterion="mel")
        ranks = {epoch: self.evaluate(keeper, epoch, mel)[1]
                 for epoch, mel in ((5, 0.9), (10, 0.7), (15, 0.8), (20, 0.6), (25, 0.95))}
        self.assertEqual(ranks, {5: 1, 10: 1, 15: 2, 20: 1, 25: None})
        self.assertEqual(self.files("best"), sorted([
            "m_20e_20s.pth", "m_10e_10s.pth", "m_15e_15s.pth", "G_20e.pth", "D_20e.pth"]))
        self.assertEqual([e["epoch"] for e in keeper.state["best"]], [20, 10, 15])
        self.assertNotIn("G", keeper.state["best"][1])

    def test_best_gd_only(self):
        keeper = self.keeper(eval_set="s", best_gd_n=2, best_criterion="mel")
        for epoch, mel in ((5, 0.9), (10, 0.7), (15, 0.8)):
            self.evaluate(keeper, epoch, mel)
        self.assertEqual(self.files("best"), sorted(["G_10e.pth", "D_10e.pth", "G_15e.pth", "D_15e.pth"]))
        self.assertEqual(self.writers.exports, [])

    def test_combined_is_relative_to_the_first_evaluation(self):
        keeper = self.keeper(eval_set="s", best_export_n=3)
        first = quiet(keeper.record_evaluation, 5, 5, {"B_mel": 1.0, "B_noise": 2.0},
                      self.writers.export, self.writers.save_gd)[0]
        second = quiet(keeper.record_evaluation, 10, 10, {"B_mel": 0.5, "B_noise": 1.0},
                       self.writers.export, self.writers.save_gd)[0]
        self.assertAlmostEqual(first, 2.0)
        self.assertAlmostEqual(second, 1.0)

    def test_state_survives_a_resume(self):
        keeper = self.keeper(eval_set="s", best_export_n=2, best_criterion="mel")
        self.evaluate(keeper, 5, 0.9)
        self.evaluate(keeper, 10, 0.7)
        resumed = self.keeper(eval_set="s", best_export_n=2, best_criterion="mel")
        self.assertEqual([e["epoch"] for e in resumed.state["best"]], [10, 5])
        self.evaluate(resumed, 15, 0.8)
        self.assertEqual([e["epoch"] for e in resumed.state["best"]], [10, 15])
        self.assertNotIn("m_5e_5s.pth", self.files("best"))

    def test_a_best_deleted_by_hand_stops_counting(self):
        keeper = self.keeper(eval_set="s", best_export_n=2, best_criterion="mel")
        self.evaluate(keeper, 5, 0.5)
        os.remove(keeper.state["best"][0]["export"])
        self.assertEqual(self.keeper(eval_set="s", best_export_n=2, best_criterion="mel").state["best"], [])

    def test_an_unreadable_state_is_set_aside(self):
        keeper = self.keeper(gd_every=1)
        keeper.save_state()
        Path(keeper.state_path).write_text("{not json")
        fresh = quiet(self.keeper, gd_every=1)
        self.assertEqual(fresh.state["best"], [])
        aside = [p for p in os.listdir(Path(self.dir, CHECKPOINTS_DIR)) if "unreadable" in p]
        self.assertEqual(len(aside), 1)

    def test_remove_checkpoints(self):
        keeper = self.keeper(gd_every=1)
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
            keeper = CheckpointKeeper(str(root), "run", policy(gd_every=1))
            keeper.save_last(50, 500, Writers().export)
            archive = quiet(reset_training_run, root, vocoder="HiFi-GAN")
            self.assertFalse((root / CHECKPOINTS_DIR).exists())
            self.assertTrue((archive / CHECKPOINTS_DIR / STATE_FILE).is_file())
            self.assertTrue((root / "G_0.pth").is_file())


if __name__ == "__main__":
    unittest.main()
