"""Reference preprocessing. Window reductions retain upstream float32 semantics."""

import numpy as np
import resampy


class FCNTensorPreprocessor:
    """CUDA FP32 equivalent of the integer-ratio resampy reference filter.

    Integer 16→8 kHz output positions need no table interpolation. The original
    filter loop uses taps 0..99 on each side (tap 100 is excluded).
    """

    filter_radius = 99

    def __init__(self, device, boundary="wrap"):
        import torch

        self.boundary = boundary
        window, precision, _ = resampy.filters.get_filter("kaiser_best")
        step = int(precision // 2)
        wing = (window[::step][:100] * 0.5).astype(np.float32)
        kernel = np.concatenate((wing[:0:-1], wing))
        self.kernel = torch.from_numpy(kernel.copy()).to(device)[None, None]
        self.reference_wing = torch.from_numpy((window[::step][:100] * 0.5).copy()).to(
            device
        )

    def resample(self, audio):
        import torch.nn.functional as F

        if len(audio) < 2:
            return audio[:0]
        import torch

        # resampy accumulates each double-weighted contribution into float32,
        # visiting the left wing first. Preserve those roundings: replacing this
        # with a single convolution changes near-DC normalization substantially.
        padded = F.pad(audio, (99, 99))
        output = torch.zeros(len(audio) // 2, device=audio.device)
        for offset in range(100):
            values = padded[99 - offset : 99 - offset + len(output) * 2 : 2]
            output = (
                output.double() + values.double() * self.reference_wing[offset]
            ).float()
        for offset in range(1, 100):
            values = padded[99 + offset : 99 + offset + len(output) * 2 : 2]
            output = (
                output.double() + values.double() * self.reference_wing[offset]
            ).float()
        return output

    def normalize(self, audio):
        import torch
        import torch.nn.functional as F

        if not len(audio):
            return audio
        if self.boundary == "wrap":
            indices = torch.arange(-497, len(audio) + 497, device=audio.device) % len(
                audio
            )
            padded = audio[indices]
        else:
            padded = F.pad(audio, (497, 497))
        windows = padded.unfold(0, 994, 1)
        blocks = []
        for start in range(0, len(audio), 8192):
            stop = min(start + 8192, len(audio))
            frames = windows[start:stop]
            # PyTorch scalar division otherwise multiplies by a rounded FP32
            # reciprocal; NumPy's division rounds the quotient instead.
            mean = (numpy_pairwise_sum(frames).double() / 994).float()
            centered = frames - mean[:, None]
            variance = (numpy_pairwise_sum(centered * centered).double() / 994).float()
            std = torch.sqrt(variance)
            std = torch.where(std == 0, torch.finfo(torch.float32).eps, std)
            blocks.append((audio[start:stop] - mean) / std)
        return torch.cat(blocks)

    def __call__(self, audio):
        import torch.nn.functional as F

        return self.normalize(F.pad(self.resample(audio), (496, 496)))


def numpy_pairwise_sum(values):
    """NumPy contiguous float32 reduction order (PW_BLOCKSIZE=128)."""
    size = values.shape[-1]
    if size < 8:
        result = values[..., 0] * 0
        for i in range(size):
            result = result + values[..., i]
        return result
    if size <= 128:
        accum = values[..., :8]
        stop = size - size % 8
        for i in range(8, stop, 8):
            accum = accum + values[..., i : i + 8]
        result = ((accum[..., 0] + accum[..., 1]) + (accum[..., 2] + accum[..., 3])) + (
            (accum[..., 4] + accum[..., 5]) + (accum[..., 6] + accum[..., 7])
        )
        for i in range(stop, size):
            result = result + values[..., i]
        return result
    split = (size // 2) // 8 * 8
    return numpy_pairwise_sum(values[..., :split]) + numpy_pairwise_sum(
        values[..., split:]
    )


def sliding_norm(audio, boundary="wrap", block_samples=8192):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("FCN requires finite mono audio")
    if not len(audio):
        return audio.copy()
    padded = np.pad(audio, 497, mode=boundary)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 994)
    result = np.empty_like(audio)
    for start in range(0, len(audio), block_samples):
        stop = min(start + block_samples, len(audio))
        frames = windows[start:stop]
        mean = frames.mean(axis=1)
        std = frames.std(axis=1)
        std[std == 0] = np.finfo(np.float32).eps
        result[start:stop] = (audio[start:stop] - mean) / std
    return result


class FCNPreprocessor:
    def __init__(self, boundary="wrap"):
        self.boundary = boundary

    def __call__(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("FCN requires finite mono 16 kHz audio")
        if len(audio) < 2:
            return np.empty(0, np.float32)
        audio = resampy.resample(audio, 16000, 8000, filter="kaiser_best")
        return sliding_norm(np.pad(audio, 496), self.boundary)
