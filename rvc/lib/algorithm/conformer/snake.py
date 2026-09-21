"""Snake activation, state_dict compatible with the Codename RVC fork's fused kernel.

Ported from rvc/lib/algorithm/conformer/snake_fused_triton.py of the Codename RVC fork,
which vendors https://github.com/falkaer/pytorch-snake (MIT).

That module holds exactly one parameter - `alpha`, shape (channels,) - and computes

    snake(x) = (x + sin(alpha * x) ** 2 / alpha) / sqrt(snake_variance(alpha))

behind a Triton kernel, with a torch.jit.script fallback when Triton is unavailable. The
kernel is an optimisation, not a different function, so what follows is the plain PyTorch
form: same parameter name, same shape, same arithmetic. A fused kernel can be dropped in
later without touching a single checkpoint. The upstream fallback path is not reused
verbatim because it is built on `torch.cuda.amp.autocast`, removed in modern torch.

`correction="std"` divides by the activation's own standard deviation, which is what keeps
a randomly initialised alpha - the periodic init draws from a gamma with a heavy right
tail - from blowing the residual stack up. It is a function of alpha, so it trains too.
"""

import torch
from torch import nn


def snake_variance(alpha: torch.Tensor) -> torch.Tensor:
    """Variance of snake(x) for x ~ N(0, 1), from pytorch-snake."""
    numerator = 1 + torch.exp(-8 * alpha**2) - 2 * torch.exp(-4 * alpha**2)
    return 1 + numerator / (8 * alpha**2)


class Snake(nn.Module):
    """x + sin(alpha x)^2 / alpha, optionally rescaled to unit variance.

    Args:
        num_channels: length of the per-channel alpha, lined up with dim 1 of a
            (batch, channels, time) input.
        init: "periodic" draws alpha from Gamma(1.5, 0.1) as upstream does ("for tasks
            with expected periodicity, larger a, usually from 5 to 50, tend to work
            well"); a number initialises every channel to it; a callable is given
            num_channels.
        correction: "std" divides by sqrt(snake_variance(alpha)); None leaves the
            activation unscaled.
    """

    def __init__(self, num_channels: int, init="periodic", correction="std"):
        super().__init__()
        if init == "periodic":
            gamma = torch.distributions.Gamma(concentration=1.5, rate=0.1)
            alpha = gamma.sample((num_channels,))
        elif callable(init):
            alpha = init(num_channels) * torch.ones(num_channels)
        else:
            alpha = float(init) * torch.ones(num_channels)
        self.alpha = nn.Parameter(alpha)
        if correction not in (None, "std"):
            raise ValueError(f"Unknown snake correction {correction!r}, expected 'std'")
        self.correction = correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Follow the input's dtype rather than promoting the whole residual block to fp32
        # the moment an fp32 parameter meets a bf16 activation under autocast.
        alpha = self.alpha.to(dtype=x.dtype)[..., None]
        out = x + torch.sin(alpha * x) ** 2 / alpha
        if self.correction == "std":
            out = out / torch.sqrt(snake_variance(alpha))
        return out

    def extra_repr(self) -> str:
        return f"num_channels={self.alpha.numel()}, correction={self.correction!r}"
