"""SPIN training data: a 16 kHz cache of a speaker-folder dataset and seeded batches from it.

build_cache() reads every WAV under <dataset>/<speaker>/ the way extraction does
(rvc.lib.utils.load_audio: mono by averaging, soxr_vhq to 16 kHz) into one float32 file,
audio16k.f32, with index.json saying where each file sits, and estimates each speaker's
pitch range for Praat (speakers.json). Workers open the cache as a memmap, so nothing
large is copied into them.

A batch is SPIN's: about batch_seconds of original audio, the files cut to one common
length (at most max_seconds) so no padding is needed, and each paired with a speaker
perturbed copy. Files of similar length are batched together, as in the original's
length-sorted sampler, so the common cut loses little. Batch k is drawn from a generator
seeded with (seed, k), which makes a run reproducible and resumable at any step.
"""

import json
import os

import numpy as np
import parselmouth
import torch

from tools.spin.perturb import perturb_speaker

SAMPLE_RATE = 16000
AUDIO_FILE = "audio16k.f32"
INDEX_FILE = "index.json"
SPEAKERS_FILE = "speakers.json"
PITCH_PROBE_FILES = 20


def _pitch_range(waves):
    """(lo, hi, median) for Praat's pitch tracker, from a speaker's voiced frames."""
    voiced = []
    for wav in waves:
        pitch = parselmouth.Sound(wav.astype(np.float64), SAMPLE_RATE).to_pitch_ac(
            pitch_floor=50, pitch_ceiling=1000
        )
        f0 = pitch.selected_array["frequency"]
        voiced.append(f0[f0 > 0])
    voiced = np.concatenate(voiced) if voiced else np.array([])
    median = float(np.median(voiced)) if len(voiced) else 150.0
    lo = float(np.clip(round(median / 2), 50, 200))
    hi = float(np.clip(round(median * 2.5), 300, 1000))
    return lo, hi, median


def build_cache(dataset, cache_dir):
    from rvc.lib.utils import load_audio

    os.makedirs(cache_dir, exist_ok=True)
    speakers = sorted((d for d in os.listdir(dataset) if d.isdigit()), key=int)
    index, ranges, offset = [], {}, 0
    with open(os.path.join(cache_dir, AUDIO_FILE), "wb") as out:
        for speaker in speakers:
            folder = os.path.join(dataset, speaker)
            names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".wav"))
            probe = []
            for i, name in enumerate(names):
                wav = load_audio(os.path.join(folder, name), SAMPLE_RATE).astype(np.float32)
                out.write(wav.tobytes())
                index.append({"path": os.path.join(folder, name), "speaker": speaker,
                              "offset": offset, "length": len(wav)})
                offset += len(wav)
                if i % max(1, len(names) // PITCH_PROBE_FILES) == 0:
                    probe.append(wav)
            ranges[speaker] = _pitch_range(probe)
            print(f"speaker {speaker}: {len(names)} files, pitch lo/hi/median {ranges[speaker]}",
                  flush=True)
    with open(os.path.join(cache_dir, INDEX_FILE), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, SPEAKERS_FILE), "w", encoding="utf-8") as f:
        json.dump(ranges, f, indent=1)
    print(f"{len(index)} files, {offset / SAMPLE_RATE / 3600:.2f} h at 16 kHz")


class SpinBatches(torch.utils.data.Dataset):
    """Batch k of a run: (originals (B, L), perturbed (B, L), Praat failures)."""

    def __init__(self, cache_dir, steps, batch_seconds=256.0, min_seconds=2.5,
                 max_seconds=17.0, seed=7122, sampler="length", speaker_weights=None,
                 short_share=0.75, short_max_seconds=4.0):
        with open(os.path.join(cache_dir, INDEX_FILE), encoding="utf-8") as f:
            index = json.load(f)
        with open(os.path.join(cache_dir, SPEAKERS_FILE), encoding="utf-8") as f:
            self.ranges = {k: tuple(v[:2]) for k, v in json.load(f).items()}
        self.audio_path = os.path.join(cache_dir, AUDIO_FILE)
        self.audio = None
        self.min_len = int(min_seconds * SAMPLE_RATE)
        self.max_len = int(max_seconds * SAMPLE_RATE)
        self.batch_len = int(batch_seconds * SAMPLE_RATE)
        kept = [e for e in index if e["length"] >= self.min_len]
        self.items = sorted(kept, key=lambda e: e["length"])
        self.lengths = np.array([min(e["length"], self.max_len) for e in self.items])
        self.steps = steps
        self.seed = seed
        self.dropped = len(index) - len(kept)
        self.kept_hours = float(np.sum([e["length"] for e in kept]) / SAMPLE_RATE / 3600)
        if sampler not in ("length", "crop"):
            raise ValueError(f"unknown sampler {sampler!r}")
        self.sampler = sampler
        weights = speaker_weights or {}
        self.weights = np.array([float(weights.get(e["speaker"], 1.0)) for e in self.items])
        self.short_share = short_share
        self.short_max_len = int(short_max_seconds * SAMPLE_RATE)

    def __len__(self):
        return self.steps

    def batch_members(self, k):
        """(item indices, common length) of batch k."""
        if self.sampler == "crop":
            return self._crop_members(k)
        return self._length_members(k)

    def _crop_members(self, k):
        """Pick the crop length first, then the clips: every file at least that long is
        drawn in proportion to the crop positions it offers times its speaker's weight.
        A file's share therefore follows its duration, not the file count, and a long file
        is also cut short. short_share of the batches crop between min_seconds and
        short_max_seconds (RVC's training slices are at most ~3.3 s and realtime feeds 1-1.5 s
        windows), the rest up to max_seconds; both log-uniform."""
        rng = np.random.default_rng([self.seed, k, 0])
        lo, hi = ((self.min_len, self.short_max_len) if rng.random() < self.short_share
                  else (self.short_max_len, self.max_len))
        hi = min(hi, int(self.lengths.max()))
        lo = min(lo, hi)
        length = int(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        eligible = np.flatnonzero(self.lengths >= length)
        count = max(1, self.batch_len // length)
        # In proportion to duration, so every second of eligible audio is equally likely to
        # be covered. Weighting by the number of crop positions (length - crop + 1) would
        # cover a 12 s file's seconds ~6x as often as those of a 2 s file at a 1.7 s crop.
        w = self.lengths[eligible] * self.weights[eligible]
        chosen = rng.choice(eligible, size=count, replace=len(eligible) < count, p=w / w.sum())
        return chosen.tolist(), length

    def _length_members(self, k):
        """A run of similar lengths around a random file, cut to the shortest. What SPIN v1
        trained with: it draws files, not seconds, so long files - the user's - were used a
        third as often as the average file and never cut short."""
        rng = np.random.default_rng([self.seed, k, 0])
        start = int(rng.integers(0, len(self.items)))
        chosen = [start]
        lo, hi = start - 1, start + 1
        # Grow outwards from a random item through its neighbours in length order until
        # the cut-to-common-length batch holds batch_seconds of audio.
        while len(chosen) * int(min(self.lengths[chosen])) < self.batch_len:
            take_hi = hi < len(self.items) and (lo < 0 or rng.random() < 0.5)
            if take_hi:
                chosen.append(hi)
                hi += 1
            elif lo >= 0:
                chosen.append(lo)
                lo -= 1
            else:
                break
        length = int(min(self.lengths[chosen]))
        return chosen, length

    def __getitem__(self, k):
        if self.audio is None:
            self.audio = np.memmap(self.audio_path, dtype=np.float32, mode="r")
        chosen, length = self.batch_members(k)
        rng = np.random.default_rng([self.seed, k, 1])
        originals, perturbed, failures = [], [], 0
        for i in chosen:
            item = self.items[i]
            start = item["offset"] + int(rng.integers(0, item["length"] - length + 1))
            wav = np.array(self.audio[start:start + length], dtype=np.float32)
            lo, hi = self.ranges[item["speaker"]]
            wav_p, failed = perturb_speaker(wav, SAMPLE_RATE, lo, hi, rng)
            originals.append(wav)
            perturbed.append(wav_p)
            failures += failed
        return (torch.from_numpy(np.stack(originals)), torch.from_numpy(np.stack(perturbed)),
                failures)
