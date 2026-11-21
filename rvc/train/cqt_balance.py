"""Hold back the CQT discriminators of a recipe that has them (APEX-GAN++, RingFormer+).

Why: the APEX-GAN++ pretrain pretrain_20261102_48k_APEXPP_1100_rmvpe112a_kushinadaSPv2
collapsed at step 87,750 (epoch 99.9) - feature matching ran away, the generator broke
and every loss after it was NaN. Of its fifteen sub-discriminators only the three CQT
ones kept growing: their effective weight norm went 8.5 -> 11.8 by epoch 80 (+39%) and
13.0 by epoch 100, while the periods, resolutions and sub-bands moved under 10%, and fm
and the discriminator's gradient norm rose with them. The APEX-GAN+ pretrain it was warm
started from has no CQT discriminators and stayed flat for 333,750 steps. The one stable
APEX-GAN++ fine-tune (naru_20261027, LR 6e-5, batch 16) grew them only 8.0 -> 10.5-11.9
over 1,600 epochs. So the CQT discriminators are the part that escalates; that they are
the cause is the best reading of the data, not a proof.

Two knobs, both in logs/<model>/config.json under "train", both default 0.5:
- cqt_lr_scale: the CQT discriminators learn at the learning rate times this. Done by
  shrinking their update after each optimizer step (p = before + s * (after - before)),
  which for AdamW is exactly a parameter group at lr * s - the decoupled weight decay is
  proportional to lr too - while leaving the optimizer's state and its single parameter
  group as they are, so checkpoints, resumes and warm starts are untouched and the value
  can change between runs of the same folder. A step GradScaler skips leaves the
  parameters as they were, and so does this.
- cqt_loss_scale: the CQT discriminators' adversarial and feature matching terms count
  this much in the generator's loss. Their loss on the discriminator's side is not
  scaled: Adam would undo a constant factor on terms that only reach the CQT
  discriminators' own parameters, which is why that side is a learning rate instead.
At 1.0 both are exactly the previous training loop, bit for bit. 0 for the loss scale
leaves the CQT discriminators training but with no say over the generator.
"""

import math

import torch

from rvc.train.losses import feature_loss, generator_loss

CQT_LR_SCALE_KEY = "cqt_lr_scale"
CQT_LOSS_SCALE_KEY = "cqt_loss_scale"
DEFAULT_CQT_LR_SCALE = 0.5
DEFAULT_CQT_LOSS_SCALE = 0.5


def validate_cqt_scale(value, name):
    """A scale as a float in [0, 1]; ValueError otherwise."""
    try:
        scale = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number between 0 and 1, got {value!r}.")
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}.")
    return scale


def read_cqt_balance(train_config):
    """(lr_scale, loss_scale) from a run's train settings (a dict or HParams)."""
    get = (
        train_config.get
        if isinstance(train_config, dict)
        else lambda key, default: getattr(train_config, key, default)
    )
    return (
        validate_cqt_scale(get(CQT_LR_SCALE_KEY, DEFAULT_CQT_LR_SCALE), CQT_LR_SCALE_KEY),
        validate_cqt_scale(
            get(CQT_LOSS_SCALE_KEY, DEFAULT_CQT_LOSS_SCALE), CQT_LOSS_SCALE_KEY
        ),
    )


def cqt_indices(net_d):
    """Positions of the CQT sub-discriminators in net_d.discriminators (DDP or not)."""
    module = net_d.module if hasattr(net_d, "module") else net_d
    return [
        i
        for i, d in enumerate(module.discriminators)
        if (getattr(d, "descriptor", None) or ("",))[0] == "Q"
    ]


def cqt_parameters(net_d, indices):
    module = net_d.module if hasattr(net_d, "module") else net_d
    return [p for i in indices for p in module.discriminators[i].parameters()]


class CQTUpdateScale:
    """Shrink the CQT discriminators' optimizer update to `scale` of itself.

    Wrap the discriminator's optimizer step: before() keeps the parameters, after() moves
    them back to before + scale * (after - before). Inactive (no copies, no work) at
    scale 1.0 or without CQT discriminators.
    """

    def __init__(self, parameters, scale):
        self.parameters = list(parameters)
        self.scale = float(scale)
        self.active = bool(self.parameters) and self.scale != 1.0
        self._before = None

    def before(self):
        if self.active:
            self._before = [p.detach().clone() for p in self.parameters]

    @torch.no_grad()
    def after(self):
        if not self.active or self._before is None:
            return
        for p, b in zip(self.parameters, self._before):
            # p = b + scale * (p - b)
            p.sub_(b).mul_(self.scale).add_(b)
        self._before = None


def generator_adversarial_terms(y_d_hat_g, fmap_r, fmap_g, indices, loss_scale):
    """(loss_gen, loss_fm, adv_cqt, fm_cqt) for the generator.

    loss_gen / loss_fm are what the generator trains on, with the CQT sub-discriminators'
    share weighted by loss_scale. adv_cqt / fm_cqt are that share unweighted, detached,
    for logging (None without CQT discriminators). With no CQT discriminators or a scale
    of 1.0 the totals are computed exactly as before, by the same functions over the same
    lists, so the result is bit-identical to the loop without this.
    """
    if not indices or loss_scale == 1.0:
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        if not indices:
            return loss_gen, loss_fm, None, None
        with torch.no_grad():
            adv_cqt = generator_loss([y_d_hat_g[i] for i in indices])[0].detach()
            fm_cqt = feature_loss(
                [fmap_r[i] for i in indices], [fmap_g[i] for i in indices]
            ).detach()
        return loss_gen, loss_fm, adv_cqt, fm_cqt
    chosen = set(indices)
    others = [i for i in range(len(y_d_hat_g)) if i not in chosen]
    adv_cqt = generator_loss([y_d_hat_g[i] for i in indices])[0]
    fm_cqt = feature_loss([fmap_r[i] for i in indices], [fmap_g[i] for i in indices])
    adv_rest = generator_loss([y_d_hat_g[i] for i in others])[0]
    fm_rest = feature_loss([fmap_r[i] for i in others], [fmap_g[i] for i in others])
    return (
        adv_rest + loss_scale * adv_cqt,
        fm_rest + loss_scale * fm_cqt,
        adv_cqt.detach(),
        fm_cqt.detach(),
    )


@torch.no_grad()
def effective_weight_norms(net_d, indices):
    """Each CQT sub-discriminator's effective weight norm: the Frobenius norm of every
    convolution's / linear layer's weight as used in the forward pass (weight norm
    applied), biases left out. This is the number that ran away before the collapse."""
    module = net_d.module if hasattr(net_d, "module") else net_d
    norms = []
    for i in indices:
        total = 0.0
        for m in module.discriminators[i].modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                total += float(m.weight.detach().double().pow(2).sum())
        norms.append(math.sqrt(total))
    return norms


# The NaN stop. A run whose losses stay non-finite has broken and will not come back:
# GradScaler skips every step, so the weights never change again, and every save from
# then on overwrites the last good checkpoint with the broken one. The collapsed pretrain
# above ran 70 more epochs (14 hours) like that. Stopping before the next save keeps the
# last good G/D to resume from.
NAN_STOP_STEPS = 20
# Steps after a (re)start that are not counted: a freshly initialised APEX-GAN decoder's
# first fp16 steps can overflow (3 of the first 10 measured), which GradScaler skips.
NAN_STOP_GRACE_STEPS = 100
NAN_STOP_EXIT = 3


class NonFiniteStop:
    """Count consecutive steps whose losses are not finite."""

    def __init__(self, limit=NAN_STOP_STEPS, grace=NAN_STOP_GRACE_STEPS):
        self.limit = limit
        self.grace = grace
        self.seen = 0
        self.consecutive = 0
        self.in_epoch = 0

    def update(self, *losses):
        """Record one step; True once `limit` consecutive steps were non-finite."""
        self.seen += 1
        finite = all(bool(torch.isfinite(loss).all()) for loss in losses)
        if finite:
            self.consecutive = 0
            return False
        if self.seen <= self.grace:
            return False
        self.consecutive += 1
        self.in_epoch += 1
        return self.consecutive >= self.limit

    def take_epoch_count(self):
        count, self.in_epoch = self.in_epoch, 0
        return count
