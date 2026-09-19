"""Warm start a generator or discriminator from a checkpoint, inheriting only what means
the same thing in both.

This replaces a loader that picked keys with `key in target` before doing anything else.
Every checkpoint on disk - the stock f0G48k.pth as much as a G_2333333.pth - stores its
weight-normed layers under the legacy `weight_g` / `weight_v` names, while the model's
state_dict calls them `parametrizations.weight.original0/1`. Those keys were all skipped
and `load_state_dict(strict=False)` hid it: 20% of the generator and none of the
discriminator was actually loaded, so a run "from a pretrained model" started with a
random flow, decoder and discriminator. Resuming never had the problem because
load_checkpoint renames the keys first.

So everything here normalises names first, decides the fate of every single tensor
before loading any of them, refuses to finish while a source tensor is unaccounted for,
and always says what it did.

What is allowed to differ between the pretrained model and this one:
- enc_p.emb_phone.weight, when the pretrained model was fitted to a different embedder -
  a different width (768 <-> 1024), or the same width and a different embedder, which is
  just as stale and does not announce itself in any shape. Nothing else in the generator
  depends on the embedder. Its bias is an offset in the encoder's own hidden space, which
  is inherited intact, so the bias is inherited with it.
- The decoder, when the vocoder differs. enc_p / enc_q / flow / emb_g are identical
  across vocoders and are inherited whole; decoder parts with the same structure and
  role are ported (see CROSS_VOCODER_DECODER_PORTS) and the rest starts from scratch.
- Individual sub-discriminators, matched by what they look at (scale, period, STFT
  resolution), never by position.
Anything else that does not line up stops the run.

Optimizer and scaler state are never read here: a warm start is always weights only.
"""

import json
import os
import sys

import torch

from rvc.lib.algorithm.discriminators import (
    DISCRIMINATOR_VERSIONS,
    describe_discriminator,
    discriminator_layout,
)
from rvc.lib.algorithm.generators.refinegan import (
    RESBLOCK_DILATION as REFINEGAN_RESBLOCK_DILATION,
)
from rvc.lib.utils import describe_embedder_mismatch

HIFIGAN = "HiFi-GAN"
MRF_HIFIGAN = "MRF HiFi-GAN"
REFINEGAN = "RefineGAN"
SIFIGAN = "SiFi-GAN"
CODENAME_RINGFORMER = "CodenameRingFormer"

# Warm starting a run whose embedder differs from the pretrain's only works because
# exactly one tensor reads the embedder's feature space.
EMBEDDER_PROJECTION_PREFIX = "enc_p.emb_phone."
DECODER_PREFIX = "dec."

STOCK_CONFIG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
)
STOCK_SAMPLE_RATES = (32000, 40000, 48000)

LEGACY_WEIGHT_NORM_SUFFIXES = (
    (".weight_g", ".parametrizations.weight.original0"),
    (".weight_v", ".parametrizations.weight.original1"),
)
WEIGHT_NORM_LEAVES = (
    ".parametrizations.weight.original0",
    ".parametrizations.weight.original1",
)


def normalize_weight_norm_keys(state_dict):
    """Rename legacy weight_g / weight_v keys to the parametrization names."""
    normalized = {}
    for key, value in state_dict.items():
        for legacy, current in LEGACY_WEIGHT_NORM_SUFFIXES:
            if key.endswith(legacy):
                key = key[: -len(legacy)] + current
                break
        normalized[key] = value
    return normalized


def detect_vocoder(state_dict):
    """Which vocoder a generator state_dict belongs to, from its decoder's keys.

    Works on legacy and current key names alike, and needs no metadata, so it covers
    every checkpoint ever saved. None when the decoder is not one of ours.
    """

    def has(prefix):
        return any(key.startswith(prefix) for key in state_dict)

    if has("dec.sn.") and has("dec.fn."):
        return SIFIGAN
    if has("dec.upsample_conv_blocks."):
        return REFINEGAN
    if has("dec.mrfs."):
        return MRF_HIFIGAN
    # Before HiFi-GAN: a CodenameRingFormer decoder has dec.resblocks and dec.ups too, and
    # only the Conformer stack tells the two apart.
    if has("dec.conformers."):
        return CODENAME_RINGFORMER
    if has("dec.resblocks.") and has("dec.ups."):
        return HIFIGAN
    return None


def checkpoint_vocoder(checkpoint):
    """Vocoder of a G_*.pth-style checkpoint: from its weights, else what it recorded."""
    return detect_vocoder(checkpoint.get("model", {})) or checkpoint.get("vocoder")


def _stock_model_config(sample_rate):
    with open(
        os.path.join(STOCK_CONFIG_DIR, f"{sample_rate}.json"), "r", encoding="utf-8"
    ) as f:
        return json.load(f)["model"]


def _sample_rate_from_upsample_kernels(state_dict, prefix):
    """Sample rate from a stack of transposed convolutions, by their kernel widths.

    The stock configs all differ there ([20,16,4,4] / [16,16,4,4] / [24,20,4,4]), so the
    widths identify the rate on their own. None when they match no stock config.
    """
    kernels = []
    while f"{prefix}{len(kernels)}.parametrizations.weight.original1" in state_dict:
        weight = state_dict[f"{prefix}{len(kernels)}.parametrizations.weight.original1"]
        kernels.append(weight.shape[-1])
    if not kernels:
        return None
    matches = [
        sample_rate
        for sample_rate in STOCK_SAMPLE_RATES
        if _stock_model_config(sample_rate)["upsample_kernel_sizes"] == kernels
    ]
    return matches[0] if len(matches) == 1 else None


def infer_hifigan_sample_rate(state_dict):
    """Sample rate of a HiFi-GAN generator that did not record one."""
    return _sample_rate_from_upsample_kernels(state_dict, "dec.ups.")


def infer_sifigan_sample_rate(state_dict):
    """Sample rate of a SiFi-GAN generator that did not record one.

    Unlike RefineGAN, a SiFi-GAN decoder's shapes *do* depend on the sample rate - its
    filter network uses the same transposed convolutions as HiFi-GAN - so this must be
    established before any decoder tensor is inherited.
    """
    return _sample_rate_from_upsample_kernels(state_dict, "dec.fn.upsamples.")


def infer_codename_ringformer_sample_rate(state_dict):
    """Sample rate of a CodenameRingFormer generator that did not record one.

    Its transposed convolutions are [8, 8] at every rate, so the kernel widths say
    nothing. conv_post does: it emits gen_istft_n_fft + 2 channels, and every stock config
    sets gen_istft_n_fft = 4 * gen_istft_hop_size while prod(upsample_rates) *
    gen_istft_hop_size = hop_length = sample_rate / 100, so the rate is n_fft * 400
    (120 -> 48000, 100 -> 40000, 80 -> 32000). Its shapes therefore do depend on the rate.
    """
    weight = state_dict.get("dec.conv_post.parametrizations.weight.original1")
    if weight is None:
        return None
    sample_rate = (weight.shape[0] - 2) * 400
    return sample_rate if sample_rate in STOCK_SAMPLE_RATES else None


def _indices(state_dict, prefix):
    """Sorted distinct integers that directly follow `prefix` in the keys."""
    found = set()
    for key in state_dict:
        if key.startswith(prefix):
            head = key[len(prefix) :].split(".", 1)[0]
            if head.isdigit():
                found.add(int(head))
    return sorted(found)


def _module_name(key):
    for leaf in WEIGHT_NORM_LEAVES:
        if key.endswith(leaf):
            return key[: -len(leaf)]
    return key.rsplit(".", 1)[0]


def _describe_modules(keys, depth=3, limit=8):
    names = []
    for key in keys:
        name = ".".join(_module_name(key).split(".")[:depth])
        if name not in names:
            names.append(name)
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"


def _shape(tensor):
    return tuple(tensor.shape)


class Transfer:
    """What a warm start does with every tensor, decided before anything is loaded."""

    def __init__(self, tag, source, target):
        self.tag = tag
        self.source = source
        self.target = target
        self.loaded = {}
        self.transformed = []
        self.consumed = set()
        self.reinitialised = {}
        self.dropped = {}
        self.errors = []
        self.notes = []

    def inherit(self, target_key, source_key):
        value = self.source.get(source_key)
        if value is None:
            self.errors.append(f"{target_key}: missing from the pretrained model")
        elif _shape(value) != _shape(self.target[target_key]):
            self.errors.append(
                f"{target_key}: {_shape(value)} -> {_shape(self.target[target_key])}"
            )
        else:
            self.loaded[target_key] = value
            self.consumed.add(source_key)

    def reinitialise(self, target_key, reason):
        self.reinitialised[target_key] = reason

    def drop(self, source_key, reason):
        self.dropped[source_key] = reason

    def check_complete(self):
        uncovered = [
            key
            for key in self.target
            if key not in self.loaded and key not in self.reinitialised
        ]
        unaccounted = [
            key
            for key in self.source
            if key not in self.consumed and key not in self.dropped
        ]
        if not self.errors:
            self.errors += [f"{key}: not decided (a bug in the warm start rules)" for key in uncovered]
        self.errors += [
            f"{key}: in the pretrained model but has no place in this one"
            for key in unaccounted
        ]

    def summary(self, checkpoint_path):
        total = sum(value.numel() for value in self.target.values())
        inherited = sum(self.target[key].numel() for key in self.loaded)
        lines = [
            f"Warm start ({self.tag}) from '{checkpoint_path}':",
            f"  inherited {len(self.loaded)} of {len(self.target)} tensors, "
            f"{inherited / max(total, 1):.1%} of this model's parameters",
        ]
        if self.transformed:
            lines.append(
                f"  converted between plain and weight-normed convolutions: "
                f"{_describe_modules(self.transformed)}"
            )
        for heading, table, tensors in (
            ("starts from scratch", self.reinitialised, self.target),
            ("not used from the pretrained model", self.dropped, self.source),
        ):
            by_reason = {}
            for key, reason in table.items():
                by_reason.setdefault(reason, []).append(key)
            for reason, keys in by_reason.items():
                count = sum(tensors[key].numel() for key in keys)
                lines.append(
                    f"  {heading}: {len(keys)} tensors / {count:,} params - {reason}"
                )
                lines.append(f"    {_describe_modules(keys)}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def _same(tensor):
    return tensor


def _weight_norm_magnitude(weight):
    """g of weight_norm(dim=0): the norm over every dimension but the first."""
    weight = weight.float()
    return torch.linalg.vector_norm(
        weight, dim=tuple(range(1, weight.dim())), keepdim=True
    )


def _weight_norm_compose(magnitude, direction):
    """The effective weight g * v / ||v|| of a weight-normed convolution."""
    direction = direction.float()
    return magnitude.float() * direction / _weight_norm_magnitude(direction)


def _hifigan_resblock_prefixes(state_dict):
    return [f"dec.resblocks.{j}." for j in _indices(state_dict, "dec.resblocks.")]


def _refinegan_resblock_prefixes(state_dict):
    return [
        f"dec.upsample_conv_blocks.{i}.blocks.{k}.1."
        for i in _indices(state_dict, "dec.upsample_conv_blocks.")
        for k in _indices(state_dict, f"dec.upsample_conv_blocks.{i}.blocks.")
    ]


def _sifigan_resblock_prefixes(state_dict):
    return [f"dec.fn.blocks.{j}." for j in _indices(state_dict, "dec.fn.blocks.")]


def _sifigan_filter_is_rvc(decoder):
    """Whether a SiFi-GAN decoder's filter blocks are this repository's ResBlock.

    The "official" variant drops the second, undilated convolution of every dilation and
    uses different kernel sizes, so its blocks have no `convs2` and cannot be paired with
    a HiFi-GAN decoder's residual blocks.
    """
    blocks = list(decoder.fn["blocks"])
    return bool(blocks) and hasattr(blocks[0], "convs2")


def _module_resblock_dilations(decoder):
    if hasattr(decoder, "upsample_conv_blocks"):
        blocks = [
            block[1] for stage in decoder.upsample_conv_blocks for block in stage.blocks
        ]
    elif hasattr(decoder, "fn"):
        blocks = list(decoder.fn["blocks"])
    else:
        blocks = list(decoder.resblocks)
    return [tuple(conv.dilation[0] for conv in block.convs1) for block in blocks]


def _hifigan_config_dilations(sample_rate, count):
    sizes = _stock_model_config(sample_rate)["resblock_dilation_sizes"]
    return [tuple(sizes[j % len(sizes)]) for j in range(count)]


def _port_resblocks(source, target, source_prefixes, target_prefixes, stages):
    """Pair every residual block's tensors, or say why the decoders do not line up.

    Both decoders hold num_stages x num_kernels residual blocks in the same order
    (stage outer, kernel size inner), and the shape check on every tensor then proves
    channels and kernel sizes agree.
    """
    source_stages, target_stages = stages
    if source_stages != target_stages or len(source_prefixes) != len(target_prefixes):
        return None, (
            f"the decoders have {source_stages} vs {target_stages} upsampling stages "
            f"and {len(source_prefixes)} vs {len(target_prefixes)} residual blocks"
        )
    entries = []
    for source_prefix, target_prefix in zip(source_prefixes, target_prefixes):
        for key in target:
            if key.startswith(target_prefix):
                entries.append(
                    (key, (source_prefix + key[len(target_prefix) :],), _same)
                )
    return entries, None


def _port_leading_resblocks(source, target, source_prefixes, target_prefixes):
    """Pair each of the target's residual blocks with the source's block of the same index.

    Unlike _port_resblocks this does not require the two decoders to have the same number
    of upsampling stages: a CodenameRingFormer decoder has two where a HiFi-GAN one has
    four, and it is the leading stages whose channel counts agree (512 -> 256 -> 128 in
    both). Whatever the shorter decoder runs out of is simply left for the caller to
    reinitialise. Only the convolutions are paired - a Snake activation's alpha has no
    counterpart in a LeakyReLU block - and every shape is still checked before anything is
    loaded.
    """
    entries = []
    for source_prefix, target_prefix in zip(source_prefixes, target_prefixes):
        for key in target:
            if key.startswith(target_prefix) and (
                ".convs1." in key or ".convs2." in key
            ):
                entries.append(
                    (key, (source_prefix + key[len(target_prefix) :],), _same)
                )
    return entries


def _port_hifigan_to_refinegan(source, target, source_sample_rate, target_decoder):
    source_prefixes = _hifigan_resblock_prefixes(source)
    target_prefixes = _refinegan_resblock_prefixes(target)
    if _hifigan_config_dilations(
        source_sample_rate, len(source_prefixes)
    ) != _module_resblock_dilations(target_decoder):
        return None, "the residual block dilations differ"
    entries, reason = _port_resblocks(
        source,
        target,
        source_prefixes,
        target_prefixes,
        (
            len(_indices(source, "dec.ups.")),
            len(_indices(target, "dec.upsample_conv_blocks.")),
        ),
    )
    if entries is None:
        return None, reason
    entries += [
        (
            "dec.conv_post.parametrizations.weight.original0",
            ("dec.conv_post.weight",),
            _weight_norm_magnitude,
        ),
        (
            "dec.conv_post.parametrizations.weight.original1",
            ("dec.conv_post.weight",),
            _same,
        ),
    ]
    return entries, None


def _port_refinegan_to_hifigan(source, target, source_sample_rate, target_decoder):
    source_prefixes = _refinegan_resblock_prefixes(source)
    target_prefixes = _hifigan_resblock_prefixes(target)
    if [REFINEGAN_RESBLOCK_DILATION] * len(
        source_prefixes
    ) != _module_resblock_dilations(target_decoder):
        return None, "the residual block dilations differ"
    entries, reason = _port_resblocks(
        source,
        target,
        source_prefixes,
        target_prefixes,
        (
            len(_indices(source, "dec.upsample_conv_blocks.")),
            len(_indices(target, "dec.ups.")),
        ),
    )
    if entries is None:
        return None, reason
    entries.append(
        (
            "dec.conv_post.weight",
            (
                "dec.conv_post.parametrizations.weight.original0",
                "dec.conv_post.parametrizations.weight.original1",
            ),
            _weight_norm_compose,
        )
    )
    return entries, None


# A SiFi-GAN decoder's filter network *is* this repository's HiFi-GAN decoder: the same
# conv_pre and speaker conditioning, the same sine source module, transposed convolutions
# of identical shape (the stock upsample_kernel_sizes are kept rather than the official
# 2 * rate assertion, precisely so they line up), the same residual block stack in stage
# outer / kernel inner order, and the same final 32 -> 1 projection before tanh. So these
# pair by name alone.
_SIFIGAN_SHARED_WITH_HIFIGAN = ("dec.conv_pre.", "dec.cond.", "dec.m_source.")
_SIFIGAN_HIFIGAN_RENAMES = (
    # (HiFi-GAN prefix, SiFi-GAN prefix)
    ("dec.ups.", "dec.fn.upsamples."),
    ("dec.conv_post.", "dec.fn.output_conv."),
)


def _hifigan_key_for_sifigan(sifigan_key):
    """The HiFi-GAN name of a SiFi-GAN decoder tensor, or None if it has no counterpart."""
    for shared in _SIFIGAN_SHARED_WITH_HIFIGAN:
        if sifigan_key.startswith(shared):
            return sifigan_key
    for hifigan_prefix, sifigan_prefix in _SIFIGAN_HIFIGAN_RENAMES:
        if sifigan_key.startswith(sifigan_prefix):
            return hifigan_prefix + sifigan_key[len(sifigan_prefix) :]
    return None


def _sifigan_hifigan_entries(target_keys, to_sifigan):
    """Name-for-name entries between a HiFi-GAN and a SiFi-GAN decoder."""
    entries = []
    for key in target_keys:
        if not key.startswith(DECODER_PREFIX):
            continue
        if to_sifigan:
            counterpart = _hifigan_key_for_sifigan(key)
        else:
            # target is HiFi-GAN: find the SiFi-GAN name that maps back to this key
            counterpart = None
            for shared in _SIFIGAN_SHARED_WITH_HIFIGAN:
                if key.startswith(shared):
                    counterpart = key
                    break
            else:
                for hifigan_prefix, sifigan_prefix in _SIFIGAN_HIFIGAN_RENAMES:
                    if key.startswith(hifigan_prefix):
                        counterpart = sifigan_prefix + key[len(hifigan_prefix) :]
                        break
        if counterpart is not None:
            entries.append((key, (counterpart,), _same))
    return entries


def _port_hifigan_to_sifigan(source, target, source_sample_rate, target_decoder):
    entries = _sifigan_hifigan_entries(target, to_sifigan=True)
    if _sifigan_filter_is_rvc(target_decoder):
        source_prefixes = _hifigan_resblock_prefixes(source)
        target_prefixes = _sifigan_resblock_prefixes(target)
        if _hifigan_config_dilations(
            source_sample_rate, len(source_prefixes)
        ) != _module_resblock_dilations(target_decoder):
            return None, "the residual block dilations differ"
        block_entries, reason = _port_resblocks(
            source,
            target,
            source_prefixes,
            target_prefixes,
            (
                len(_indices(source, "dec.ups.")),
                len(_indices(target, "dec.fn.upsamples.")),
            ),
        )
        if block_entries is None:
            return None, reason
        entries += block_entries
    return entries, None


def _port_sifigan_to_hifigan(source, target, source_sample_rate, target_decoder):
    entries = _sifigan_hifigan_entries(target, to_sifigan=False)
    source_prefixes = _sifigan_resblock_prefixes(source)
    target_prefixes = _hifigan_resblock_prefixes(target)
    if len(source_prefixes) == len(target_prefixes):
        block_entries, reason = _port_resblocks(
            source,
            target,
            source_prefixes,
            target_prefixes,
            (
                len(_indices(source, "dec.fn.upsamples.")),
                len(_indices(target, "dec.ups.")),
            ),
        )
        if block_entries is None:
            return None, reason
        entries += block_entries
    return entries, None


def _port_refinegan_to_sifigan(source, target, source_sample_rate, target_decoder):
    if not _sifigan_filter_is_rvc(target_decoder):
        return None, (
            "this model's SiFi-GAN filter network follows the official paper and has no "
            "counterpart in a RefineGAN decoder"
        )
    source_prefixes = _refinegan_resblock_prefixes(source)
    target_prefixes = _sifigan_resblock_prefixes(target)
    if [REFINEGAN_RESBLOCK_DILATION] * len(
        source_prefixes
    ) != _module_resblock_dilations(target_decoder):
        return None, "the residual block dilations differ"
    entries, reason = _port_resblocks(
        source,
        target,
        source_prefixes,
        target_prefixes,
        (
            len(_indices(source, "dec.upsample_conv_blocks.")),
            len(_indices(target, "dec.fn.upsamples.")),
        ),
    )
    if entries is None:
        return None, reason
    entries.append(
        (
            "dec.fn.output_conv.weight",
            (
                "dec.conv_post.parametrizations.weight.original0",
                "dec.conv_post.parametrizations.weight.original1",
            ),
            _weight_norm_compose,
        )
    )
    return entries, None


def _port_sifigan_to_refinegan(source, target, source_sample_rate, target_decoder):
    source_prefixes = _sifigan_resblock_prefixes(source)
    target_prefixes = _refinegan_resblock_prefixes(target)
    if [REFINEGAN_RESBLOCK_DILATION] * len(
        target_prefixes
    ) != _module_resblock_dilations(target_decoder):
        return None, "the residual block dilations differ"
    entries, reason = _port_resblocks(
        source,
        target,
        source_prefixes,
        target_prefixes,
        (
            len(_indices(source, "dec.fn.upsamples.")),
            len(_indices(target, "dec.upsample_conv_blocks.")),
        ),
    )
    if entries is None:
        return None, reason
    entries += [
        (
            "dec.conv_post.parametrizations.weight.original0",
            ("dec.fn.output_conv.weight",),
            _weight_norm_magnitude,
        ),
        (
            "dec.conv_post.parametrizations.weight.original1",
            ("dec.fn.output_conv.weight",),
            _same,
        ),
    ]
    return entries, None


# Whether a HiFi-GAN decoder's residual blocks are offered to a CodenameRingFormer one.
#
# They are the same shape - stages of 256 and 128 channels, kernel sizes 3/7/11, dilations
# 1/3/5, two convolutions per dilation - but they do not do the same job: HiFi-GAN's run on
# a signal that is already most of the way to a waveform, while CodenameRingFormer's run on
# iSTFT frames that only become samples after conv_post and the inverse transform.
#
# Shape agreement alone is not a reason to inherit, so this was measured rather than
# assumed. Mel L1 on 16 real 48 kHz clips before any optimizer step, every configuration
# built from the same RNG state, warm started from a 1024-dim HiFi-GAN G_2333333.pth:
#
#   from scratch          7.518
#   conv_pre + cond       7.505   (40.5% of parameters inherited)
#   + residual blocks    15.760   (58.8% inherited, and twice as bad as scratch)
#
# So the blocks are worse than useless here: fed iSTFT frames they are far out of the
# distribution they were trained on, exactly as the reasoning above predicts. Turning this
# on trades 18 points of parameter count for double the error. Controls on the same clips,
# where inheriting a decoder does work: HiFi-GAN 4.468 from scratch, 3.970 warm started.
CODENAME_RINGFORMER_PORTS_RESBLOCKS = False

# conv_pre and cond mean exactly the same thing in both decoders: project the latent to
# the decoder's width, and add the speaker embedding to it. conv_pre is weight-normed in
# CodenameRingFormer and plain in HiFi-GAN, so it is converted.
_CODENAME_RINGFORMER_TO_HIFIGAN_PLAIN = ("dec.cond.weight", "dec.cond.bias")


def _fitting_entries(entries, source, target):
    """Drop entries whose keys are absent on either side, e.g. cond with gin_channels 0."""
    return [
        entry
        for entry in entries
        if entry[0] in target and all(key in source for key in entry[1])
    ]


def _port_hifigan_to_codename_ringformer(
    source, target, source_sample_rate, target_decoder
):
    entries = [
        (
            "dec.conv_pre.parametrizations.weight.original0",
            ("dec.conv_pre.weight",),
            _weight_norm_magnitude,
        ),
        (
            "dec.conv_pre.parametrizations.weight.original1",
            ("dec.conv_pre.weight",),
            _same,
        ),
        ("dec.conv_pre.bias", ("dec.conv_pre.bias",), _same),
    ] + [(key, (key,), _same) for key in _CODENAME_RINGFORMER_TO_HIFIGAN_PLAIN]
    if CODENAME_RINGFORMER_PORTS_RESBLOCKS:
        target_prefixes = _hifigan_resblock_prefixes(target)
        if _hifigan_config_dilations(
            source_sample_rate, len(target_prefixes)
        ) != _module_resblock_dilations(target_decoder):
            return None, "the residual block dilations differ"
        entries += _port_leading_resblocks(
            source, target, _hifigan_resblock_prefixes(source), target_prefixes
        )
    return _fitting_entries(entries, source, target), None


def _port_codename_ringformer_to_hifigan(
    source, target, source_sample_rate, target_decoder
):
    entries = [
        (
            "dec.conv_pre.weight",
            (
                "dec.conv_pre.parametrizations.weight.original0",
                "dec.conv_pre.parametrizations.weight.original1",
            ),
            _weight_norm_compose,
        ),
        ("dec.conv_pre.bias", ("dec.conv_pre.bias",), _same),
    ] + [(key, (key,), _same) for key in _CODENAME_RINGFORMER_TO_HIFIGAN_PLAIN]
    if CODENAME_RINGFORMER_PORTS_RESBLOCKS:
        target_prefixes = _hifigan_resblock_prefixes(target)
        if _hifigan_config_dilations(
            source_sample_rate, len(target_prefixes)
        ) != _module_resblock_dilations(target_decoder):
            return None, "the residual block dilations differ"
        entries += _port_leading_resblocks(
            source, target, _hifigan_resblock_prefixes(source), target_prefixes
        )
    return _fitting_entries(entries, source, target), None


# Decoder parts that mean the same thing in two vocoders, for a warm start between them.
#
# HiFi-GAN <-> RefineGAN: the residual blocks of each upsampling stage (same channels
# 256/128/64/32, kernel sizes 3/7/11 and dilations 1,3,5, averaged the same way) and
# conv_post, the final 32 -> 1 projection before tanh (plain in HiFi-GAN, weight-normed
# in RefineGAN, so it is converted). Nothing else lines up: conv_pre projects to 512
# channels where mel_conv projects to 256, cond therefore differs too, the transposed
# convolutions have no counterpart in a parameter-free linear upsample, noise_convs are
# not downsample_blocks, and only HiFi-GAN's harmonic merge has a bias.
#
# HiFi-GAN <-> SiFi-GAN: everything above plus conv_pre, cond, m_source and the
# transposed convolutions, because SiFi-GAN's filter network is the HiFi-GAN decoder.
# Only HiFi-GAN's noise_convs (SiFi-GAN injects the sine through its source network) and
# SiFi-GAN's whole source network have no counterpart.
#
# RefineGAN <-> SiFi-GAN: the same range as HiFi-GAN <-> RefineGAN, for the same reasons.
#
# These are a good starting point rather than an identical function: HiFi-GAN's blocks
# use a LeakyReLU slope of 0.1 where RefineGAN's use 0.2, and in RefineGAN they sit
# behind a freshly initialised input_conv.
CROSS_VOCODER_DECODER_PORTS = {
    (HIFIGAN, REFINEGAN): _port_hifigan_to_refinegan,
    (REFINEGAN, HIFIGAN): _port_refinegan_to_hifigan,
    (HIFIGAN, SIFIGAN): _port_hifigan_to_sifigan,
    (SIFIGAN, HIFIGAN): _port_sifigan_to_hifigan,
    (REFINEGAN, SIFIGAN): _port_refinegan_to_sifigan,
    (SIFIGAN, REFINEGAN): _port_sifigan_to_refinegan,
    (HIFIGAN, CODENAME_RINGFORMER): _port_hifigan_to_codename_ringformer,
    (CODENAME_RINGFORMER, HIFIGAN): _port_codename_ringformer_to_hifigan,
}


def _plan_decoder(transfer, source_identity, target_identity, target_module):
    source, target = transfer.source, transfer.target
    source_vocoder, target_vocoder = (
        source_identity["vocoder"],
        target_identity["vocoder"],
    )
    source_rate, target_rate = (
        source_identity["sample_rate"],
        target_identity["sample_rate"],
    )
    source_keys = [key for key in source if key.startswith(DECODER_PREFIX)]
    target_keys = [key for key in target if key.startswith(DECODER_PREFIX)]

    if source_vocoder == target_vocoder:
        if source_rate and target_rate and source_rate != target_rate:
            transfer.errors.append(
                f"the pretrained {source_vocoder} decoder was trained at {source_rate} Hz "
                f"and this model is {target_rate} Hz"
            )
            return
        if source_rate is None and target_vocoder not in (
            HIFIGAN,
            SIFIGAN,
            CODENAME_RINGFORMER,
            None,
        ):
            transfer.notes.append(
                f"the pretrained model does not record its sample rate and a "
                f"{target_vocoder} decoder's shapes do not depend on it, so it is assumed "
                f"to have been trained at {target_rate} Hz like this model"
            )
        for key in target_keys:
            transfer.inherit(key, key)
        return

    port = CROSS_VOCODER_DECODER_PORTS.get((source_vocoder, target_vocoder))
    reason = None
    if port is None:
        reason = (
            f"there is no known correspondence between a "
            f"{source_vocoder or 'unrecognised'} and a {target_vocoder or 'unrecognised'} "
            "decoder"
        )
    elif source_rate is None or target_rate is None:
        reason = (
            "the pretrained model's sample rate cannot be established (it does not "
            "record one)"
        )
    elif source_rate != target_rate:
        reason = (
            f"the pretrained model was trained at {source_rate} Hz and this model is "
            f"{target_rate} Hz"
        )
    else:
        entries, reason = port(source, target, source_rate, target_module.dec)
        if entries is not None:
            ported = {}
            for target_key, entry_source_keys, convert in entries:
                if target_key not in target or any(
                    key not in source for key in entry_source_keys
                ):
                    reason = f"{target_key} has no counterpart in the pretrained decoder"
                    break
                value = convert(*(source[key] for key in entry_source_keys))
                if _shape(value) != _shape(target[target_key]):
                    reason = (
                        f"{target_key} would be {_shape(value)} but this model has "
                        f"{_shape(target[target_key])}"
                    )
                    break
                ported[target_key] = (value, entry_source_keys, convert is not _same)
            else:
                for target_key, (value, entry_source_keys, converted) in ported.items():
                    transfer.loaded[target_key] = value
                    transfer.consumed.update(entry_source_keys)
                    if converted:
                        transfer.transformed.append(target_key)
                transfer.notes.append(
                    f"decoder ported {source_vocoder} -> {target_vocoder}: "
                    f"{_describe_modules(sorted(ported))}. A starting point, not an "
                    "identical function (the blocks can sit behind a different LeakyReLU "
                    "slope and freshly initialised input convolutions)."
                )

    if reason is not None:
        transfer.notes.append(
            f"decoder not ported ({reason}); the whole {target_vocoder} decoder starts "
            "from scratch"
        )
    for key in target_keys:
        if key not in transfer.loaded:
            transfer.reinitialise(
                key, f"no counterpart in the pretrained {source_vocoder} decoder"
            )
    for key in source_keys:
        if key not in transfer.consumed:
            transfer.drop(key, f"no counterpart in this model's {target_vocoder} decoder")


def _embedder_projection_reason(key, value, source, embedder_reason):
    """Why this enc_p.emb_phone tensor cannot be inherited, or None.

    A different width says so in the shapes. A different embedder at the same width does
    not, and is exactly as stale: kushinada-hubert-large and japanese-hubert-large are
    both 1024 wide and their feature spaces have nothing to do with one another, so a
    weight fitted to one is a scrambled map of the other. Only the weight matrix reads
    that space; the bias is an offset in the encoder's own hidden space, which is
    inherited intact, so it is kept.
    """
    if key not in source:
        return None
    if _shape(source[key]) != _shape(value):
        return (
            "the pretrained model was trained on a different sized embedder "
            f"({_shape(source[key])} -> {_shape(value)})"
        )
    if embedder_reason and value.dim() > 1:
        return embedder_reason
    return None


def _plan_generator(
    transfer, checkpoint, target_identity, target_module, target_embedder=None
):
    source, target = transfer.source, transfer.target
    embedder_reason = (
        describe_embedder_mismatch(checkpoint, target_embedder, "the pretrained model")
        if target_embedder
        else None
    )
    for key, value in target.items():
        if key.startswith(DECODER_PREFIX):
            continue
        reason = (
            _embedder_projection_reason(key, value, source, embedder_reason)
            if key.startswith(EMBEDDER_PROJECTION_PREFIX)
            else None
        )
        if reason:
            transfer.reinitialise(key, reason)
            transfer.drop(key, reason)
        else:
            transfer.inherit(key, key)

    source_vocoder = detect_vocoder(source) or checkpoint.get("vocoder")
    source_rate = checkpoint.get("sample_rate")
    if source_rate is None:
        # Both of these decoders encode the sample rate in their transposed convolution
        # kernel widths, so it can be recovered from a checkpoint written before the
        # metadata stamp existed. A RefineGAN decoder's shapes carry no such trace.
        if source_vocoder == HIFIGAN:
            source_rate = infer_hifigan_sample_rate(source)
        elif source_vocoder == SIFIGAN:
            source_rate = infer_sifigan_sample_rate(source)
        elif source_vocoder == CODENAME_RINGFORMER:
            source_rate = infer_codename_ringformer_sample_rate(source)
    target_identity = target_identity or {}
    _plan_decoder(
        transfer,
        {"vocoder": source_vocoder, "sample_rate": source_rate},
        {
            "vocoder": detect_vocoder(target) or target_identity.get("vocoder"),
            "sample_rate": target_identity.get("sample_rate"),
        },
        target_module,
    )


def _discriminator_kind(weight):
    if weight is None:
        return None
    if weight.dim() == 3:
        return "S"
    if tuple(weight.shape[2:]) == (5, 1):
        return "P"
    if tuple(weight.shape[2:]) == (3, 9):
        return "R"
    return None


def _describe_discriminator_descriptor(descriptor):
    if descriptor[0] == "P":
        return f"period {descriptor[1]} discriminator"
    if descriptor[0] == "R":
        return f"STFT resolution {list(descriptor[1])} discriminator"
    return "scale discriminator"


def _source_discriminator_layout(transfer, checkpoint):
    """The pretrained discriminator's sub-discriminators, in order.

    Periods cannot be read off the weights (every period's layers are the same shape),
    so the layout is recognised as one of the known versions from the kind and number of
    sub-discriminators, and cross-checked against what the checkpoint recorded.
    """
    source = transfer.source
    kinds = []
    while any(key.startswith(f"discriminators.{len(kinds)}.") for key in source):
        kinds.append(
            _discriminator_kind(
                source.get(
                    f"discriminators.{len(kinds)}.convs.0.parametrizations.weight.original1"
                )
            )
        )
    matches = [
        version
        for version in DISCRIMINATOR_VERSIONS
        if [descriptor[0] for descriptor in discriminator_layout(version)] == kinds
    ]
    recorded = checkpoint.get("disc_version")
    if len(matches) != 1:
        transfer.errors.append(
            f"the pretrained discriminator's layout ({len(kinds)} sub-discriminators: "
            f"{kinds}) is not one of the known versions {list(DISCRIMINATOR_VERSIONS)}"
        )
        return None
    if recorded is not None and recorded != matches[0]:
        transfer.errors.append(
            f"the pretrained discriminator records version '{recorded}' but its weights "
            f"are laid out as '{matches[0]}'"
        )
        return None
    return discriminator_layout(matches[0])


def _plan_discriminator(transfer, checkpoint, target_module):
    source_layout = _source_discriminator_layout(transfer, checkpoint)
    if source_layout is None:
        return
    source_index = {descriptor: i for i, descriptor in enumerate(source_layout)}
    used = set()
    for target_index, discriminator in enumerate(target_module.discriminators):
        descriptor = describe_discriminator(discriminator)
        target_prefix = f"discriminators.{target_index}."
        keys = [key for key in transfer.target if key.startswith(target_prefix)]
        source_i = source_index.get(descriptor)
        if source_i is None:
            reason = (
                f"the pretrained discriminator has no "
                f"{_describe_discriminator_descriptor(descriptor)}"
            )
            for key in keys:
                transfer.reinitialise(key, reason)
            continue
        used.add(source_i)
        for key in keys:
            transfer.inherit(
                key, f"discriminators.{source_i}." + key[len(target_prefix) :]
            )
    for source_i, descriptor in enumerate(source_layout):
        if source_i in used:
            continue
        reason = f"this model has no {_describe_discriminator_descriptor(descriptor)}"
        for key in transfer.source:
            if key.startswith(f"discriminators.{source_i}."):
                transfer.drop(key, reason)


def warm_start(
    net,
    checkpoint_path,
    tag,
    target_identity=None,
    target_embedder=None,
    verbose=True,
):
    """Load a pretrained G or D into `net`, inheriting only what is compatible.

    Args:
        net: The generator (Synthesizer) or MultiPeriodDiscriminator, optionally DDP wrapped.
        checkpoint_path: A G_*.pth / D_*.pth style checkpoint with a "model" state_dict.
        tag: "G" or "D", for messages.
        target_identity: {"vocoder", "sample_rate"} of the model being trained. Without a
            sample rate a decoder is only ever inherited from the same vocoder.
        target_embedder: the embedder identity of the features this run trains on. Without
            it enc_p.emb_phone.weight is only rebuilt when the width differs.
        verbose: Print what was inherited, converted, started from scratch and left out.

    Returns the Transfer describing what happened. Exits the process instead when the
    checkpoint does not fit.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" not in checkpoint:
        exported = "weight" in checkpoint
        print(
            f"Cannot warm start ({tag}) from '{checkpoint_path}': "
            + (
                "this is an exported inference model (logs/<model>/<model>_*e_*s.pth). "
                "It has no posterior encoder and no training state; use the G_*.pth / "
                "D_*.pth from the model folder instead."
                if exported
                else "it has no 'model' state_dict."
            )
        )
        sys.exit(1)

    module = net.module if hasattr(net, "module") else net
    transfer = Transfer(
        tag, normalize_weight_norm_keys(checkpoint["model"]), module.state_dict()
    )
    if hasattr(module, "discriminators"):
        _plan_discriminator(transfer, checkpoint, module)
    else:
        _plan_generator(
            transfer, checkpoint, target_identity, module, target_embedder
        )
    transfer.check_complete()

    if transfer.errors:
        print(
            f"Cannot warm start ({tag}) from '{checkpoint_path}': it does not match this "
            "model, most likely a different sample rate, vocoder or model type:"
        )
        for line in transfer.errors[:40]:
            print(f"  {line}")
        if len(transfer.errors) > 40:
            print(f"  ... and {len(transfer.errors) - 40} more")
        for note in transfer.notes:
            print(f"  note: {note}")
        sys.exit(1)

    module.load_state_dict(transfer.loaded, strict=False)
    if verbose:
        print(transfer.summary(checkpoint_path))
    return transfer
