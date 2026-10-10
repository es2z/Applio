"""Phase-conditioned Snake activation for the APEX-GAN decoder.

Ported from rvc/lib/algorithm/generators/apex_gan_modules/snake_fused_triton_phase.py of
the Codename RVC fork at commit ef613f900d0b646123081b331332996b92cd52e5
(https://github.com/DanielNawa/codename-rvc-fork-4, MIT, docs/licenses/codename-rvc-fork-LICENSE),
which builds on https://github.com/falkaer/pytorch-snake (MIT).

    plain:              out = x + sin(alpha * x) ** 2 / alpha
    phase-conditioned:  out = x + sin(alpha * x + beta) ** 2 / alpha

beta arrives as a unit-norm (cos beta, sin beta) pair from the decoder's PhaseProjector and
is expanded with the angle-sum identity, sin(ax + b) = sin(ax) cos(b) + cos(ax) sin(b), so no
trigonometry is ever taken of beta itself. alpha is per channel; cos beta / sin beta are full
(batch, channels, time) tensors.

Upstream runs this behind a Triton kernel with a torch.jit.script fallback. The kernel is an
optimisation, not a different function, so what follows is the fallback's arithmetic as
plain PyTorch, in a torch.autograd.Function with the same memory behaviour as both upstream
paths: the forward output is not saved, and backward recomputes it from x, alpha and beta.
Without that, autograd would keep the sin / cos / product intermediates of every activation
alive - 18 of them per stage, the widest at the 12 kHz STFT frame rate.

Precision follows upstream's Triton path, which is the one it actually runs: the arithmetic
is done in fp32 (alpha is an fp32 parameter) and the output is returned in the input's
dtype. Float64 inputs stay float64, so torch.autograd.gradcheck can check the backward.

Only what the APEX-GAN decoder uses is kept: a constant alpha initialisation and no output
correction (upstream's Snake(channels, init=1.0, correction=None)), and the normal variant of
the Snake-aware Kaiming initialisation for the convolutions in front of it.
"""

import math

import torch
from torch import nn
from torch.nn import init


def snake_second_moment(alpha):
    """E[snake(x)^2] for x ~ N(0, 1), from pytorch-snake."""
    exp = torch.exp if torch.is_tensor(alpha) else math.exp
    numerator = 3 + exp(-8 * alpha**2) - 4 * exp(-2 * alpha**2)
    return 1 + numerator / (8 * alpha**2)


def snake_gain(kind):
    """Gain for a layer feeding a Snake: 'approx' is 1, a number is that alpha's exact one."""
    if kind == "approx":
        return 1
    return 1 / math.sqrt(snake_second_moment(kind))


def snake_kaiming_normal_(tensor, kind="approx", mode="fan_in"):
    """Kaiming normal initialisation for a convolution in front of a Snake.

    Upstream's snake_kaiming_normal_ with correction=None, which is how every APEX-GAN
    residual convolution calls it (kind = the Snake's initial alpha, 1.0).
    """
    fan = init._calculate_correct_fan(tensor, mode)
    std = snake_gain(kind) / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


def _compute_dtype(x, alpha):
    return torch.promote_types(torch.promote_types(x.dtype, alpha.dtype), torch.float32)


def _snake_terms(x, alpha, cos_beta, sin_beta, dtype):
    """(x, alpha[:, None], sin(ax), cos(ax), q, dq/d(ax)) in the compute dtype."""
    x = x.to(dtype)
    alpha = alpha.to(dtype)[..., None]
    ax = alpha * x
    s, c = torch.sin(ax), torch.cos(ax)
    if cos_beta is None:
        return x, alpha, s, c, s, c
    cos_beta, sin_beta = cos_beta.to(dtype), sin_beta.to(dtype)
    q = s * cos_beta + c * sin_beta
    dq = c * cos_beta - s * sin_beta
    return x, alpha, s, c, q, dq


class _SnakeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, cos_beta=None, sin_beta=None):
        dtype = _compute_dtype(x, alpha)
        xc = x.to(dtype)
        ax = alpha.to(dtype)[..., None] * xc
        if cos_beta is None:
            q = torch.sin(ax)
        else:
            q = torch.sin(ax) * cos_beta.to(dtype) + torch.cos(ax) * sin_beta.to(dtype)
        out = xc + q * q / alpha.to(dtype)[..., None]
        # The output is deliberately not saved: everything needed to rebuild it is.
        ctx.save_for_backward(x, alpha, cos_beta, sin_beta)
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, cos_beta, sin_beta = ctx.saved_tensors
        needs_x, needs_alpha, needs_cos, needs_sin = ctx.needs_input_grad[:4]
        dtype = _compute_dtype(x, alpha)
        xc, a, s, c, q, dq = _snake_terms(x, alpha, cos_beta, sin_beta, dtype)
        grad = grad_output.to(dtype)

        grad_x = grad_alpha = grad_cos = grad_sin = None
        if needs_x or needs_alpha:
            # d/dx (q^2 / alpha) = 2 q dq/d(ax)
            dydx = (1 + 2 * q * dq) * grad
            if needs_x:
                grad_x = dydx.to(x.dtype)
            if needs_alpha:
                # d/dalpha = (x * dydx - out * grad) / alpha, summed over batch and time
                out = xc + q * q / a
                grad_alpha = (
                    torch.sum(xc * dydx - out * grad, dim=(0, 2)) / a.squeeze(-1)
                ).to(alpha.dtype)
        if cos_beta is not None and (needs_cos or needs_sin):
            two_q_over_alpha = 2 * q / a
            if needs_cos:
                grad_cos = (two_q_over_alpha * s * grad).to(cos_beta.dtype)
            if needs_sin:
                grad_sin = (two_q_over_alpha * c * grad).to(sin_beta.dtype)
        return grad_x, grad_alpha, grad_cos, grad_sin


class Snake(nn.Module):
    """x + sin(alpha x + beta)^2 / alpha, with beta optional.

    Args:
        num_channels: length of the per-channel alpha, lined up with dim 1 of a
            (batch, channels, time) input.
        init: initial value of every channel's alpha.
    """

    def __init__(self, num_channels: int, init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(float(init) * torch.ones(num_channels))

    def forward(self, x: torch.Tensor, beta=None) -> torch.Tensor:
        """beta, if given, is a (cos beta, sin beta) pair shaped like x, unit norm.

        The caller keeps the pair on the unit circle; this only rotates the argument of
        the sine. Without beta this is the ordinary Snake and reads no extra tensors.
        """
        cos_beta, sin_beta = (None, None) if beta is None else beta
        return _SnakeFunction.apply(x, self.alpha, cos_beta, sin_beta)

    def extra_repr(self) -> str:
        return f"num_channels={self.alpha.numel()}"
