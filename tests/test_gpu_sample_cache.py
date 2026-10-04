"""The GPU data cache must train on exactly what the DataLoader would.

Run from the repository root:
    env\\python.exe -m unittest tests.test_gpu_sample_cache -v

train.py used to cache the first epoch's padded batches and only shuffle their order,
which froze which clips share a batch and which clips the bucket sampler repeats. These
tests pin the replacement: GPUSampleCache holds samples, and every epoch it yields what
the DataLoader yields for that epoch, tensor for tensor. Real files on disk, no mocks.
"""

import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SAMPLE_RATE = 48000
HOP = 480
BOUNDARIES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]


def load_data_utils():
    """rvc/train/data_utils.py, with the train-script imports it expects.

    It imports `utils` and `mel_processing` as top-level modules, the way train.py runs
    with rvc/train on sys.path; point those names at the package modules while loading
    it, so an unrelated `utils` elsewhere can never stand in.
    """
    names = ("utils", "mel_processing")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["utils"] = importlib.import_module("rvc.train.utils")
    sys.modules["mel_processing"] = importlib.import_module("rvc.train.mel_processing")
    try:
        spec = importlib.util.spec_from_file_location(
            "rvc_train_data_utils", REPO / "rvc" / "train" / "data_utils.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, module_before in saved.items():
            if module_before is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module_before
    return module


data_utils = load_data_utils()
from rvc.train.utils import HParams  # noqa: E402


def original_collate(batch):
    """TextAudioCollateMultiNSFsid as it was before it learnt to pad on any device."""
    _, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
    )
    max_spec_len = max([x[0].size(1) for x in batch])
    max_wave_len = max([x[1].size(1) for x in batch])
    spec_lengths = torch.LongTensor(len(batch))
    wave_lengths = torch.LongTensor(len(batch))
    spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
    wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len)
    spec_padded.zero_()
    wave_padded.zero_()
    max_phone_len = max([x[2].size(0) for x in batch])
    phone_lengths = torch.LongTensor(len(batch))
    phone_padded = torch.FloatTensor(len(batch), max_phone_len, batch[0][2].shape[1])
    pitch_padded = torch.LongTensor(len(batch), max_phone_len)
    pitchf_padded = torch.FloatTensor(len(batch), max_phone_len)
    phone_padded.zero_()
    pitch_padded.zero_()
    pitchf_padded.zero_()
    sid = torch.LongTensor(len(batch))
    for i in range(len(ids_sorted_decreasing)):
        row = batch[ids_sorted_decreasing[i]]
        spec = row[0]
        spec_padded[i, :, : spec.size(1)] = spec
        spec_lengths[i] = spec.size(1)
        wave = row[1]
        wave_padded[i, :, : wave.size(1)] = wave
        wave_lengths[i] = wave.size(1)
        phone = row[2]
        phone_padded[i, : phone.size(0), :] = phone
        phone_lengths[i] = phone.size(0)
        pitch = row[3]
        pitch_padded[i, : pitch.size(0)] = pitch
        pitchf = row[4]
        pitchf_padded[i, : pitchf.size(0)] = pitchf
        sid[i] = row[5]
    return (
        phone_padded,
        phone_lengths,
        pitch_padded,
        pitchf_padded,
        spec_padded,
        spec_lengths,
        wave_padded,
        wave_lengths,
        sid,
    )


def assert_same_batch(test, expected, actual):
    test.assertEqual(len(expected), len(actual))
    for index, (a, b) in enumerate(zip(expected, actual)):
        test.assertEqual(a.dtype, b.dtype, f"element {index}")
        test.assertEqual(a.shape, b.shape, f"element {index}")
        test.assertTrue(torch.equal(a.to(b.device), b), f"element {index} differs")


class GPUSampleCacheTest(unittest.TestCase):
    """A small folder in the layout extraction writes, spread over several buckets."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        rng = np.random.default_rng(0)
        lines = []
        # 0.4 to 3.0 s spans buckets 50..400; one clip of 0.2 s is below the first
        # boundary and is never drawn, so it must not be cached either.
        durations = list(rng.uniform(0.4, 3.0, 45)) + [0.2]
        for index, seconds in enumerate(durations):
            samples = int(seconds * SAMPLE_RATE) // HOP * HOP
            frames = samples // HOP
            base = os.path.join(cls.tmp, f"{index % 3}_{index}_0")
            sf.write(
                base + ".wav",
                (rng.standard_normal(samples) * 0.1).astype(np.float32),
                SAMPLE_RATE,
                subtype="FLOAT",
            )
            np.save(base + ".npy", rng.standard_normal((frames // 2 + 1, 8)).astype(np.float32))
            np.save(base + ".pitch.npy", rng.integers(1, 255, frames + 2))
            np.save(base + ".pitchf.npy", rng.uniform(80, 400, frames + 2).astype(np.float32))
            lines.append(
                f"{base}.wav|{base}.npy|{base}.pitch.npy|{base}.pitchf.npy|{index % 3}"
            )
        filelist = os.path.join(cls.tmp, "filelist.txt")
        with open(filelist, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        cls.hparams = HParams(
            training_files=filelist,
            max_wav_value=32768.0,
            sample_rate=SAMPLE_RATE,
            filter_length=2048,
            hop_length=HOP,
            win_length=2048,
        )
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _loader(self, batch_size=4):
        dataset = data_utils.TextAudioLoaderMultiNSFsid(self.hparams)
        sampler = data_utils.DistributedBucketSampler(
            dataset, batch_size, list(BOUNDARIES), num_replicas=1, rank=0, shuffle=True
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=0,
            collate_fn=data_utils.TextAudioCollateMultiNSFsid(),
            batch_sampler=sampler,
        )
        return dataset, sampler, loader

    def _filled_cache(self, dataset, sampler):
        cache = data_utils.GPUSampleCache()
        cache.fill(dataset, sampler, self.device, num_workers=0)
        return cache

    def test_the_collate_is_unchanged_on_the_cpu(self):
        dataset, sampler, _ = self._loader()
        collate = data_utils.TextAudioCollateMultiNSFsid()
        sampler.set_epoch(1)
        for indices in list(sampler)[:6]:
            batch = [dataset[index] for index in indices]
            assert_same_batch(self, original_collate(batch), collate(batch))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_the_collate_pads_on_the_gpu_exactly_as_on_the_cpu(self):
        dataset, sampler, _ = self._loader()
        collate = data_utils.TextAudioCollateMultiNSFsid()
        sampler.set_epoch(1)
        for indices in list(sampler)[:6]:
            batch = [dataset[index] for index in indices]
            on_gpu = collate([tuple(t.cuda() for t in sample) for sample in batch])
            for tensor in on_gpu:
                self.assertEqual(tensor.device.type, "cuda")
            assert_same_batch(self, collate(batch), on_gpu)

    def test_every_drawable_sample_is_cached_once_and_unpadded(self):
        dataset, sampler, _ = self._loader()
        cache = self._filled_cache(dataset, sampler)
        drawable = {index for bucket in sampler.buckets for index in bucket}
        self.assertEqual(set(cache.samples), drawable)
        self.assertEqual(len(drawable), len(dataset) - 1)  # the 0.2 s clip is not
        for index in drawable:
            for cached, loaded in zip(cache.samples[index], dataset[index]):
                self.assertEqual(cached.device.type, self.device.type)
                self.assertTrue(torch.equal(cached.cpu(), loaded))

    def test_every_epoch_trains_on_what_the_dataloader_would(self):
        dataset, sampler, loader = self._loader()
        cache = self._filled_cache(dataset, sampler)
        for epoch in (1, 2, 3):
            sampler.set_epoch(epoch)
            expected = list(loader)
            sampler.set_epoch(epoch)
            actual = list(cache.batches(sampler, loader.collate_fn))
            self.assertEqual(len(expected), len(actual))
            for a, b in zip(expected, actual):
                assert_same_batch(self, a, b)

    def test_batches_are_drawn_afresh_every_epoch(self):
        """The bug: with whole batches cached, epoch 2 was epoch 1 in another order."""
        dataset, sampler, loader = self._loader()
        cache = self._filled_cache(dataset, sampler)

        def draw(epoch):
            sampler.set_epoch(epoch)
            batches = list(cache.batches(sampler, loader.collate_fn))
            # A batch is identified by the waveforms in it, independent of their order.
            return {
                frozenset(
                    round(float(wave[0, :length].sum()), 4)
                    for wave, length in zip(batch[6], batch[7])
                )
                for batch in batches
            }

        first, second = draw(1), draw(2)
        self.assertNotEqual(first, second)
        self.assertLess(len(first & second), len(first) // 2)


if __name__ == "__main__":
    unittest.main()
