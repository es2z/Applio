"""Conformer blocks for the CodenameRingFormer decoder.

Ported from rvc/lib/algorithm/conformer/conformer.py of the Codename RVC fork, which
follows the Conformer paper (Gulati et al., 2020) with lucidrains' RingAttention in place
of the usual multi-head self attention - the "Ring" in RingFormer.

Two notes on the port:

* `ring_attention_pytorch` is imported lazily, inside ConformerBlock. Importing it at
  module scope would make a missing optional dependency break every other vocoder, since
  rvc/lib/algorithm/synthesizers.py imports all generators unconditionally.
* Upstream's Conformer accepts attn_dropout / ff_dropout / conv_dropout and then does not
  pass them to its blocks, so its models train with dropout 0 whatever is asked for. The
  plumbing is fixed here and the defaults are 0.0, which keeps the behaviour that its
  checkpoints were actually trained with while making it settable.
"""

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn


def calc_same_padding(kernel_size: int):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


def _ring_attention_class():
    try:
        from ring_attention_pytorch import RingAttention
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The CodenameRingFormer vocoder needs ring-attention-pytorch. Install the "
            "requirements again, or `pip install ring-attention-pytorch`."
        ) from error
    return RingAttention


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in: int, chan_out: int, kernel_size: int, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        return self.conv(F.pad(x, self.padding))


class Scale(nn.Module):
    def __init__(self, scale: float, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim: int,
        causal: bool = False,
        expansion_factor: int = 2,
        kernel_size: int = 31,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        conv_dropout: float = 0.0,
        conv_causal: bool = False,
        block_size: int = 512,
    ):
        super().__init__()
        ring_attention = _ring_attention_class()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        # ring_attn only splits the sequence across ranks when torch.distributed is
        # initialised; in a single process this is ordinary causal attention.
        #
        # use_cuda_kernel is not left at its default. RingAttention defaults it to
        # torch.cuda.is_available(), and that path imports a Triton flash-attention kernel
        # behind `importlib.metadata.version('triton-nightly')`. Triton on Windows is
        # distributed as triton-windows (import name `triton`, 3.7.1 here), so that lookup
        # raises and the module answers with print() + exit() - which kills the training
        # worker outright, with no traceback and no non-zero exit code to notice it by.
        # The kernel would buy nothing here in any case: the Conformer sees 36 and 144
        # frames per segment while training. Same maths, same parameters, same state_dict.
        #
        # force_regular_attn is set for a second, independent reason. Without it, turning
        # the CUDA kernel off lands on `ring_flash_attn`, a hand-written autograd.Function
        # whose backward mixes fp32 and fp16 and dies with "expected scalar type Float but
        # found Half" the moment training runs in fp16. autocast does not extend to
        # backward, so a forward-only check does not see it. `default_attention` is plain
        # einsum and softmax, which autograd handles at whatever dtype autocast chose.
        #
        # Nothing is lost by it: `ring_attn` is already `self.ring_attn & is_distributed()`
        # inside forward, so in a single process no ring reduction was happening either
        # way, and the block-wise form is only a memory optimisation - identical maths -
        # that has nothing to save at 36 and 144 frames.
        self.attn = ring_attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            causal=True,
            auto_shard_seq=True,
            bucket_size=block_size,
            ring_attn=True,
            ring_seq_size=512,
            use_cuda_kernel=False,
            force_regular_attn=True,
        )
        self.self_attn_dropout = nn.Dropout(attn_dropout)
        self.conv = ConformerConvModule(
            dim=dim,
            causal=conv_causal,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x_ff1 = self.ff1(x) + x
        x = self.attn(x, mask=mask)
        x = self.self_attn_dropout(x)
        x = x + x_ff1
        x = self.conv(x) + x
        x = self.ff2(x) + x
        return self.post_norm(x)


class Conformer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        conv_dropout: float = 0.0,
        conv_causal: bool = False,
        block_size: int = 512,
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    ff_mult=ff_mult,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    conv_dropout=conv_dropout,
                    conv_causal=conv_causal,
                    block_size=block_size,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x
