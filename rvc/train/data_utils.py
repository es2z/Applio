import os
import numpy as np
import torch
import torch.utils.data

from mel_processing import spectrogram_torch
from utils import load_filepaths_and_text, load_wav_to_torch


class TextAudioLoaderMultiNSFsid(torch.utils.data.Dataset):
    """
    Dataset that loads text and audio pairs.

    Args:
        hparams: Hyperparameters.
    """

    def __init__(self, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(hparams.training_files)
        self.max_wav_value = hparams.max_wav_value
        self.sample_rate = hparams.sample_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sample_rate = hparams.sample_rate
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 5000)
        self._filter()

    def _filter(self):
        """
        Filters audio paths and text pairs based on text length.
        """
        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text, pitch, pitchf, dv in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text, pitch, pitchf, dv])
                lengths.append(os.path.getsize(audiopath) // (3 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        """
        Converts speaker ID to a LongTensor.

        Args:
            sid (str): Speaker ID.
        """
        try:
            sid = torch.LongTensor([int(sid)])
        except ValueError as error:
            print(f"Error converting speaker ID '{sid}' to integer. Exception: {error}")
            sid = torch.LongTensor([0])
        return sid

    def get_audio_text_pair(self, audiopath_and_text):
        """
        Loads and processes audio and text data for a single pair.

        Args:
            audiopath_and_text (list): List containing audio path, text, pitch, pitchf, and speaker ID.
        """
        file = audiopath_and_text[0]
        phone = audiopath_and_text[1]
        pitch = audiopath_and_text[2]
        pitchf = audiopath_and_text[3]
        dv = audiopath_and_text[4]

        phone, pitch, pitchf = self.get_labels(phone, pitch, pitchf)
        spec, wav = self.get_audio(file)
        dv = self.get_sid(dv)

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length

            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]

            phone = phone[:len_min, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]

        return (spec, wav, phone, pitch, pitchf, dv)

    def get_labels(self, phone, pitch, pitchf):
        """
        Loads and processes phoneme, pitch, and pitchf labels.

        Args:
            phone (str): Path to phoneme label file.
            pitch (str): Path to pitch label file.
            pitchf (str): Path to pitchf label file.
        """
        phone = np.load(phone)
        phone = np.repeat(phone, 2, axis=0)
        pitch = np.load(pitch)
        pitchf = np.load(pitchf)
        n_num = min(phone.shape[0], 900)
        phone = phone[:n_num, :]
        pitch = pitch[:n_num]
        pitchf = pitchf[:n_num]
        phone = torch.FloatTensor(phone)
        pitch = torch.LongTensor(pitch)
        pitchf = torch.FloatTensor(pitchf)
        return phone, pitch, pitchf

    def get_audio(self, filename):
        """
        Loads and processes audio data.

        Args:
            filename (str): Path to audio file.
        """
        audio, sample_rate = load_wav_to_torch(filename)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{sample_rate} SR doesn't match target {self.sample_rate} SR"
            )
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename, weights_only=True)
            except Exception as error:
                print(f"An error occurred getting spec from {spec_filename}: {error}")
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.hop_length,
                self.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        return spec, audio_norm

    def __getitem__(self, index):
        """
        Returns a single audio-text pair.

        Args:
            index (int): Index of the data sample.
        """
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.audiopaths_and_text)


class TextAudioCollateMultiNSFsid:
    """
    Collates text and audio data for training.

    Args:
        return_ids (bool, optional): Whether to return sample IDs. Defaults to False.
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """
        Collates a batch of data samples.

        Args:
            batch (list): List of data samples.
        """
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )
        # Padded on the device the samples are on, so the same code collates for the
        # DataLoader on the CPU and for GPUSampleCache on the GPU. Lengths are gathered
        # as Python ints and written once, rather than element by element into a device
        # tensor.
        device = batch[0][0].device
        size = len(batch)

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wave_len = max([x[1].size(1) for x in batch])
        spec_padded = torch.zeros(
            size, batch[0][0].size(0), max_spec_len, dtype=torch.float32, device=device
        )
        wave_padded = torch.zeros(
            size, 1, max_wave_len, dtype=torch.float32, device=device
        )

        max_phone_len = max([x[2].size(0) for x in batch])
        phone_padded = torch.zeros(
            size, max_phone_len, batch[0][2].shape[1], dtype=torch.float32, device=device
        )
        pitch_padded = torch.zeros(size, max_phone_len, dtype=torch.long, device=device)
        pitchf_padded = torch.zeros(
            size, max_phone_len, dtype=torch.float32, device=device
        )
        spec_lengths, wave_lengths, phone_lengths, sids = [], [], [], []

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths.append(spec.size(1))

            wave = row[1]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths.append(wave.size(1))

            phone = row[2]
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths.append(phone.size(0))

            pitch = row[3]
            pitch_padded[i, : pitch.size(0)] = pitch
            pitchf = row[4]
            pitchf_padded[i, : pitchf.size(0)] = pitchf

            sids.append(row[5].reshape(1))

        def lengths(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        spec_lengths = lengths(spec_lengths)
        wave_lengths = lengths(wave_lengths)
        phone_lengths = lengths(phone_lengths)
        sid = torch.cat(sids).to(device=device, dtype=torch.long)

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


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Distributed sampler that groups data into buckets based on length.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        batch_size (int): Batch size.
        boundaries (list): List of length boundaries for buckets.
        num_replicas (int, optional): Number of processes participating in distributed training. Defaults to None.
        rank (int, optional): Rank of the current process. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        """
        Creates buckets of data samples based on length.
        """
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, -1, -1):  #
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        """
        Iterates over batches of data samples.
        """
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        """
        Performs binary search to find the bucket index for a given length.

        Args:
            x (int): Length to find the bucket for.
            lo (int, optional): Lower bound of the search range. Defaults to 0.
            hi (int, optional): Upper bound of the search range. Defaults to None.
        """
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        """
        Returns the length of the sampler.
        """
        return self.num_samples // self.batch_size


class _IndexedDataset(torch.utils.data.Dataset):
    """A dataset's samples paired with their index, for filling GPUSampleCache."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return index, self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def _keep_as_list(batch):
    # Module level rather than a lambda: DataLoader workers are spawned on Windows and
    # have to pickle it.
    return batch


class GPUSampleCache:
    """Every training sample on one GPU, unpadded, batched afresh every epoch.

    train.py used to cache whole padded batches the first epoch produced and only shuffle
    their order afterwards. That froze the first epoch's draw for the rest of the run:
    which clips share a batch, and which clips DistributedBucketSampler repeats to fill a
    bucket up to a multiple of the batch size - measured on a 791 clip folder at batch
    24, the same 96 clips (13%) were trained twice every epoch for the whole run. Holding
    samples rather than batches lets the sampler draw every epoch exactly as it does
    without the cache, and it drops the padding the batches carried.
    """

    def __init__(self):
        self.samples = {}

    def __len__(self):
        return len(self.samples)

    def fill(self, dataset, batch_sampler, device, num_workers=4):
        """Load every sample the sampler can draw onto `device`, once."""
        indices = sorted({index for bucket in batch_sampler.buckets for index in bucket})
        loader = torch.utils.data.DataLoader(
            _IndexedDataset(dataset),
            batch_size=16,
            sampler=indices,
            num_workers=num_workers,
            collate_fn=_keep_as_list,
        )
        for chunk in loader:
            for index, sample in chunk:
                self.samples[index] = tuple(
                    tensor.to(device, non_blocking=True) for tensor in sample
                )

    def batches(self, batch_sampler, collate_fn):
        """This epoch's batches, as the sampler draws them, padded on the GPU."""
        for indices in batch_sampler:
            yield collate_fn([self.samples[index] for index in indices])
