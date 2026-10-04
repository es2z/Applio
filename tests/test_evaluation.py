"""Evaluation sets and the in-training evaluation (rvc/train/evaluation.py).

Run from the repository root:
    env\\python.exe -m unittest tests.test_evaluation -v

The preprocessing is compared against the real PreProcess, the metrics are checked on
synthetic signals with a known answer, the cache is driven with stand-in extractors (so
no embedder download is needed), and the Evaluator runs a real generator on the CPU.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import rvc.train.evaluation as evaluation
from rvc.train.evaluation import (
    GROUPS,
    Evaluator,
    collect_eval_files,
    create_eval_set,
    extraction_spec,
    f0_tracking,
    list_eval_sets,
    pick_auto_trained,
    prepare_eval_cache,
    preprocess_like_training,
    summarize_eval_set,
    validate_set_name,
    voiced_hf_flatness,
)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def tone(seconds, sample_rate, f0=220.0, noise=0.0, seed=0):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    signal = sum(0.3 / k * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 30) if f0 * k < sample_rate / 2)
    rng = np.random.default_rng(seed)
    return (signal + noise * rng.standard_normal(t.size)).astype(np.float32)


class EvalSetFolderTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        patcher = mock.patch.object(evaluation, "EVAL_SETS_DIR", Path(temp.name) / "eval_sets")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_create_makes_the_folders_and_readme(self):
        path = create_eval_set("tier1")
        self.assertEqual(sorted(p.name for p in path.iterdir()), ["convert", "heldout", "readme.txt", "trained"])
        readme = (path / "readme.txt").read_text(encoding="utf-8")
        for group in GROUPS:
            self.assertIn(group, readme)
        self.assertIn("tier1", readme)
        self.assertEqual(list_eval_sets(), ["tier1"])

    def test_names_that_cannot_be_used(self):
        create_eval_set("a")
        for bad in ("", "  ", "a", "x/y", 'q"', "name.", "con:1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                create_eval_set(bad)
        self.assertEqual(validate_set_name("  ok  "), "ok")

    def test_the_list_reads_only_the_top_level(self):
        create_eval_set("b")
        create_eval_set("a")
        (evaluation.EVAL_SETS_DIR / "a" / "trained" / "1").mkdir()
        (evaluation.EVAL_SETS_DIR / "stray.txt").write_text("x")
        self.assertEqual(list_eval_sets(), ["a", "b"])


class SpeakerMappingTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for relative in ("trained/a.wav", "trained/1/b.wav", "trained/7/c.wav", "trained/x/d.wav",
                         "heldout/1/e.wav", "convert/f.wav", "convert/sub/g.wav", "trained/notes.txt"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        (self.root / "heldout").mkdir(exist_ok=True)

    def entries(self, n_speakers):
        entries, warnings = collect_eval_files(self.root, n_speakers)
        return {e["name"]: (e["group"], e["sid"]) for e in entries}, warnings

    def test_one_speaker_scores_everything_as_zero(self):
        entries, _ = self.entries(1)
        self.assertEqual(entries, {
            "trained/a.wav": ("trained", 0), "trained/1/b.wav": ("trained", 0),
            "trained/7/c.wav": ("trained", 0), "heldout/1/e.wav": ("heldout", 0),
            "convert/f.wav": ("convert", 0)})

    def test_several_speakers_use_the_folder_number(self):
        entries, warnings = self.entries(3)
        self.assertEqual(entries, {
            "trained/a.wav": ("trained", 0), "trained/1/b.wav": ("trained", 1),
            "heldout/1/e.wav": ("heldout", 1), "convert/f.wav": ("convert", 0)})
        text = " ".join(warnings)
        self.assertIn("no speaker 7", text)
        self.assertIn("trained/x/", text)
        self.assertIn("convert/sub/", text)

    def test_files_beyond_the_limit_are_dropped_with_a_warning(self):
        for i in range(12):
            (self.root / "convert" / f"z{i:02d}.wav").write_bytes(b"")
        entries, warnings = self.entries(1)
        self.assertEqual(sum(1 for group, _ in entries.values() if group == "convert"), 10)
        self.assertTrue(any("only the first 10" in w for w in warnings))

    def test_summary(self):
        with mock.patch.object(evaluation, "EVAL_SETS_DIR", self.root.parent):
            text = summarize_eval_set(self.root.name, 3)
        self.assertIn("trained: 2 (speaker 0: 1, speaker 1: 1)", text)
        self.assertIn("heldout: 1 (speaker 1: 1)", text)
        self.assertIn("convert: 1", text)
        self.assertIn("! ", text)


class PreprocessTest(unittest.TestCase):
    def test_it_matches_the_real_preprocessor(self):
        from rvc.train.preprocess.preprocess import PreProcess

        with tempfile.TemporaryDirectory() as temp:
            source = os.path.join(temp, "in.wav")
            sf.write(source, tone(2.5, 44100, noise=0.01), 44100)
            pre = PreProcess(48000, temp)
            quiet(pre.process_audio, source, 0, 0, "Skip", True, False, 0.7, 3.0, 0.3, "pre")
            expected, _ = sf.read(os.path.join(temp, "sliced_audios", "0_0_0.wav"), dtype="float32")
            expected_16k, _ = sf.read(os.path.join(temp, "sliced_audios_16k", "0_0_0.wav"), dtype="float32")
            audio, audio_16k = preprocess_like_training(source, 48000)
        np.testing.assert_allclose(audio, expected, atol=1e-6)
        np.testing.assert_allclose(audio_16k, expected_16k, atol=1e-6)

    def test_long_clips_are_cut_to_nine_seconds(self):
        with tempfile.TemporaryDirectory() as temp:
            source = os.path.join(temp, "long.wav")
            sf.write(source, tone(12.0, 16000), 16000)
            audio, audio_16k = preprocess_like_training(source, 32000)
        self.assertEqual(len(audio), 9 * 32000)
        self.assertEqual(len(audio_16k), 9 * 16000)


class MetricTest(unittest.TestCase):
    def test_noise_raises_the_voiced_high_band_flatness(self):
        sample_rate = 48000
        f0 = np.full(201, 220.0)
        clean = voiced_hf_flatness(tone(2.0, sample_rate), sample_rate, f0)
        noisy = voiced_hf_flatness(tone(2.0, sample_rate, noise=0.02), sample_rate, f0)
        self.assertGreater(noisy, 3 * clean)
        self.assertIsNone(voiced_hf_flatness(tone(2.0, sample_rate), sample_rate, np.zeros(201)))

    def test_f0_tracking(self):
        target = np.array([200.0] * 50 + [0.0] * 10)
        sharp = target * 2 ** (100 / 1200)
        cents, mismatch = f0_tracking(sharp, target)
        self.assertAlmostEqual(cents, 100.0, places=3)
        self.assertEqual(mismatch, 0.0)
        cents, mismatch = f0_tracking(np.where(np.arange(60) < 30, 200.0, 0.0), target)
        self.assertAlmostEqual(mismatch, 20 / 60)


def write_model_folder(root, coarse_max=1100.0, speakers=1, sample_rate=32000):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"data": {"sample_rate": sample_rate}}))
    (root / "model_info.json").write_text(json.dumps({
        "embedder_model": "contentvec",
        "embedder_output_layer": None,
        "speakers_id": speakers,
        "f0_coarse_max": coarse_max,
        "pitch_extraction_run": {"specification": {"method": "rmvpe"}},
    }))


class FakeFeatureInput:
    calls = 0

    def __init__(self, f0_method, device, coarse_max):
        self.coarse_max = coarse_max

    def compute_f0(self, audio_16k):
        FakeFeatureInput.calls += 1
        return np.full(len(audio_16k) // 160 + 1, 200.0, dtype=np.float32)

    def coarse_f0(self, f0):
        from rvc.lib.predictors.f0_quantization import quantize_f0

        return quantize_f0(f0, maximum=self.coarse_max)


class FakeEmbedder(torch.nn.Module):
    def forward(self, x):
        return x


def fake_embedder_forward(model, feats):
    frames = feats.shape[1] // 320
    return torch.zeros(1, frames, 768)


class CacheTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        sets = self.temp / "eval_sets"
        for name in ("trained/a.wav", "convert/b.wav"):
            path = sets / "s" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, tone(1.5, 32000), 32000)
        patches = [
            mock.patch.object(evaluation, "EVAL_SETS_DIR", sets),
            mock.patch.object(evaluation, "_feature_input_class", lambda: FakeFeatureInput),
            mock.patch("rvc.lib.utils.load_embedding", lambda *a, **k: FakeEmbedder()),
            mock.patch("rvc.lib.utils.embedder_forward", fake_embedder_forward),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        FakeFeatureInput.calls = 0

    def test_extracted_once_and_rebuilt_when_the_settings_change(self):
        model = self.temp / "run"
        write_model_folder(model)
        manifest = quiet(prepare_eval_cache, model, "s", "cpu")
        self.assertEqual(FakeFeatureInput.calls, 2)
        self.assertEqual(sorted(e["group"] for e in manifest["entries"]), ["convert", "trained"])
        cached = np.load(manifest["entries"][0]["cache"])
        self.assertEqual(sorted(cached.files), ["audio", "coarse", "f0", "phone"])
        self.assertTrue((model / "evaluation" / "set.json").is_file())

        quiet(prepare_eval_cache, model, "s", "cpu")
        self.assertEqual(FakeFeatureInput.calls, 2)  # all cached

        write_model_folder(model, coarse_max=1680.0)
        quiet(prepare_eval_cache, model, "s", "cpu")
        self.assertEqual(FakeFeatureInput.calls, 4)  # another coarse range, extracted again

    def test_a_run_that_cannot_be_reproduced_is_refused(self):
        model = self.temp / "custom"
        write_model_folder(model)
        info = json.loads((model / "model_info.json").read_text())
        info["embedder_model"] = "custom"
        (model / "model_info.json").write_text(json.dumps(info))
        with self.assertRaises(ValueError):
            extraction_spec(model)


class AutoTrainedTest(unittest.TestCase):
    def test_the_same_filelist_gives_the_same_clips(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "filelist.txt")
            lines = [f"w{s}_{i}.wav|p{s}_{i}.npy|c{s}_{i}.npy|f{s}_{i}.npy|{s}" for s in (0, 1) for i in range(40)]
            lines.append("logs/mute/mute.wav|m.npy|mc.npy|mf.npy|0")
            Path(path).write_text("\n".join(reversed(lines)))
            first = pick_auto_trained(path)
            second = pick_auto_trained(path)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual({e["sid"] for e in first}, {0, 1})
        self.assertFalse(any("mute" in e["wav"] for e in first))


class EvaluatorTest(unittest.TestCase):
    """A real generator on the CPU, scored on hand-made cache entries."""

    @classmethod
    def setUpClass(cls):
        from tests.test_codename_ringformer import build_synthesizer, stock

        from rvc.lib.predictors.f0_quantization import quantize_f0
        from rvc.train.utils import HParams

        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.sample_rate = 32000
        cls.config = HParams(**stock(cls.sample_rate))
        torch.manual_seed(0)
        cls.net = quiet(build_synthesizer, "HiFi-GAN", cls.sample_rate, 768, 2).eval()

        frames = 120  # 1.2 s at 100 frames/s
        f0 = np.full(frames, 220.0, dtype=np.float32)
        f0[50:60] = 0.0
        entries = []
        for group, sid in (("trained", 0), ("heldout", 1), ("convert", 0)):
            path = root / f"{group}.npz"
            np.savez(path, audio=tone(frames / 100, cls.sample_rate, noise=0.005),
                     phone=np.random.default_rng(1).standard_normal((frames // 2, 768)).astype(np.float32),
                     f0=f0, coarse=quantize_f0(f0, maximum=1100.0))
            entries.append({"group": group, "name": f"{group}/x.wav", "sid": sid, "cache": str(path)})
        filelist = root / "filelist.txt"
        rows = []
        for sid, median in ((0, 200.0), (1, 300.0)):
            f0_path = root / f"f0_{sid}.npy"
            np.save(f0_path, np.full(100, median, dtype=np.float32))
            rows.append(f"w.wav|p.npy|c.npy|{f0_path}|{sid}")
        filelist.write_text("\n".join(rows))
        cls.manifest = {"spec": {"f0_method": "rmvpe", "f0_coarse_max": 1100.0},
                        "entries": entries, "transpose": {}}
        cls.filelist = str(filelist)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def evaluator(self):
        built = Evaluator(self.manifest, [], self.config, "cpu", self.filelist, seeds=(0,))

        class EchoF0:
            # The generator's F0 is whatever it was asked for, give or take - enough to
            # exercise the tracking path without an F0 model.
            def compute_f0(self, audio_16k):
                return built._last_f0

        built._f0_extractor = lambda: EchoF0()
        return built

    def test_scores_groups_and_convert_targets(self):
        built = self.evaluator()
        # convert/x.wav is generated once per target speaker (0 and 1), pitch moved to
        # each speaker's training F0 median (200 Hz, 300 Hz) from its own 220 Hz.
        converts = [item for item in built.items if item["group"] == "convert"]
        self.assertEqual(sorted(item["sid"] for item in converts), [0, 1])
        medians = sorted(float(np.median(item["f0"][item["f0"] > 0])) for item in converts)
        np.testing.assert_allclose(medians, [200.0, 300.0], rtol=1e-5)
        built._last_f0 = converts[0]["f0"]
        scores, samples = built.evaluate(self.net, use_amp=False, dtype=torch.float32)
        for key in ("A_mel", "B_mel", "A_noise", "C_f0_cents", "C_vuv_mismatch"):
            self.assertIn(key, scores)
        self.assertEqual(sorted(samples), ["convert", "heldout", "trained"])
        self.assertEqual(samples["trained"][1].shape, (120 * self.config["data"]["hop_length"],))

    def test_scores_depend_only_on_the_weights_and_leave_the_rng_alone(self):
        built = self.evaluator()
        built._last_f0 = np.full(120, 200.0)
        torch.manual_seed(123)
        expected_next = torch.rand(3)
        torch.manual_seed(123)
        first, _ = built.evaluate(self.net, use_amp=False, dtype=torch.float32)
        after = torch.rand(3)
        second, _ = built.evaluate(self.net, use_amp=False, dtype=torch.float32)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(after, expected_next))
        self.assertFalse(self.net.training)


if __name__ == "__main__":
    unittest.main()
