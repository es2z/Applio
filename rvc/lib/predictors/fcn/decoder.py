import math

import torch


class FCNDecoder:
    def __call__(self, activation, centers=None):
        if activation.shape[-1] != 486:
            raise ValueError("FCN-993 requires 486 pitch bins")
        confidence, peak = activation.max(dim=-1)
        if centers is None:
            centers = peak
        offsets = torch.arange(-4, 5, device=activation.device)
        indices = centers[..., None] + offsets
        valid = (indices >= 0) & (indices < 486)
        indices = indices.clamp(0, 485)
        weights = activation.gather(-1, indices) * valid
        # Upstream uses a float64 linspace and float32 weight sum.
        mapping = torch.linspace(
            1200 * math.log2(3),
            1200 * math.log2(100),
            486,
            device=activation.device,
            dtype=torch.float64,
        )
        cents = (weights * mapping[indices]).sum(-1) / weights.sum(-1)
        hz = torch.nan_to_num(10 * torch.pow(2, cents / 1200), nan=0.0).float()
        return cents, hz, confidence
