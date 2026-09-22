"""FCN coarse-pitch contract shared by training and conversion."""

import numpy as np


def quantize_f0(f0, minimum=50.0, maximum=1680.0):
    low, high = 1127 * np.log1p(np.array([minimum, maximum]) / 700)
    mel = 1127 * np.log1p(np.asarray(f0) / 700)
    return np.rint(np.clip((mel - low) * 254 / (high - low) + 1, 1, 255)).astype(
        np.int64
    )
