"""rvc/train/cqt_balance.py: holding back the CQT discriminators, and the NaN stop."""

import json

import pytest
import torch

from rvc.train.cqt_balance import (
    CQTUpdateScale,
    NonFiniteStop,
    cqt_indices,
    cqt_parameters,
    effective_weight_norms,
    generator_adversarial_terms,
    read_cqt_balance,
    validate_cqt_scale,
)
from rvc.train.losses import feature_loss, generator_loss


def _two_part_model(seed):
    torch.manual_seed(seed)
    return torch.nn.ModuleDict(
        {"rest": torch.nn.Linear(6, 5), "cqt": torch.nn.Linear(5, 3)}
    ).double()


@pytest.mark.parametrize("scale", [0.5, 0.25, 0.0])
def test_update_scale_is_adamw_at_a_scaled_learning_rate(scale):
    """Shrinking the update after the step is exactly a parameter group at lr * scale,
    weight decay included, while the optimizer keeps its single group."""
    lr = 1e-3
    a, b = _two_part_model(0), _two_part_model(0)
    opt_a = torch.optim.AdamW(a.parameters(), lr, betas=(0.8, 0.99), eps=1e-9)
    shrink = CQTUpdateScale(a["cqt"].parameters(), scale)
    opt_b = torch.optim.AdamW(
        [
            {"params": b["rest"].parameters(), "lr": lr},
            {"params": b["cqt"].parameters(), "lr": lr * scale},
        ],
        betas=(0.8, 0.99),
        eps=1e-9,
    )
    gen = torch.Generator().manual_seed(1)
    for _ in range(25):
        x = torch.randn(4, 6, generator=gen, dtype=torch.float64)
        for model, opt in ((a, opt_a), (b, opt_b)):
            opt.zero_grad()
            model["cqt"](model["rest"](x)).pow(2).mean().backward()
        shrink.before()
        opt_a.step()
        shrink.after()
        opt_b.step()
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-12, atol=1e-12, msg=name)
    assert len(opt_a.param_groups) == 1


def test_update_scale_is_inert_at_one_and_without_cqt():
    model = _two_part_model(0)
    assert not CQTUpdateScale(model["cqt"].parameters(), 1.0).active
    assert not CQTUpdateScale([], 0.5).active
    shrink = CQTUpdateScale(model["cqt"].parameters(), 1.0)
    shrink.before()
    assert shrink._before is None


def test_a_skipped_step_leaves_the_parameters_alone():
    model = _two_part_model(0)
    shrink = CQTUpdateScale(model["cqt"].parameters(), 0.5)
    kept = [p.detach().clone() for p in model["cqt"].parameters()]
    shrink.before()
    shrink.after()  # no optimizer step in between, as when GradScaler skips one
    for p, k in zip(model["cqt"].parameters(), kept):
        assert torch.equal(p, k)


def _fake_outputs(n, seed=0):
    gen = torch.Generator().manual_seed(seed)
    y_d_hat_g = [torch.rand(2, 7, generator=gen, requires_grad=True) for _ in range(n)]
    fmap_r = [[torch.rand(2, 3, 5, generator=gen) for _ in range(3)] for _ in range(n)]
    fmap_g = [
        [torch.rand(2, 3, 5, generator=gen, requires_grad=True) for _ in range(3)]
        for _ in range(n)
    ]
    return y_d_hat_g, fmap_r, fmap_g


def test_generator_terms_are_bit_identical_at_one_and_without_cqt():
    y, fr, fg = _fake_outputs(6)
    plain_gen, _ = generator_loss(y)
    plain_fm = feature_loss(fr, fg)
    for indices in ([], [2, 3, 4]):
        loss_gen, loss_fm, adv_cqt, fm_cqt = generator_adversarial_terms(
            y, fr, fg, indices, 1.0
        )
        assert torch.equal(loss_gen, plain_gen) and torch.equal(loss_fm, plain_fm)
        assert (adv_cqt is None) == (not indices)
    _, _, adv_cqt, fm_cqt = generator_adversarial_terms(y, fr, fg, [2, 3, 4], 1.0)
    torch.testing.assert_close(adv_cqt, generator_loss(y[2:5])[0].detach())
    torch.testing.assert_close(fm_cqt, feature_loss(fr[2:5], fg[2:5]).detach())


def test_generator_terms_weight_only_the_cqt_share():
    y, fr, fg = _fake_outputs(6, seed=3)
    idx = [2, 3, 4]
    rest = [0, 1, 5]
    loss_gen, loss_fm, adv_cqt, fm_cqt = generator_adversarial_terms(y, fr, fg, idx, 0.5)
    want_gen = generator_loss([y[i] for i in rest])[0] + 0.5 * generator_loss(y[2:5])[0]
    want_fm = feature_loss([fr[i] for i in rest], [fg[i] for i in rest]) + 0.5 * feature_loss(
        fr[2:5], fg[2:5]
    )
    torch.testing.assert_close(loss_gen, want_gen)
    torch.testing.assert_close(loss_fm, want_fm)
    assert not adv_cqt.requires_grad and not fm_cqt.requires_grad
    # The CQT outputs get half the gradient they would at 1.0, the rest the same.
    (loss_gen + loss_fm).backward()
    half = [y[i].grad.clone() for i in range(6)]
    for t in y:
        t.grad = None
    for maps in fg:
        for t in maps:
            t.grad = None
    full_gen, full_fm, _, _ = generator_adversarial_terms(y, fr, fg, idx, 1.0)
    (full_gen + full_fm).backward()
    for i in range(6):
        factor = 0.5 if i in idx else 1.0
        torch.testing.assert_close(half[i], factor * y[i].grad)
    # 0 takes the CQT discriminators out of the generator's loss entirely.
    zero_gen, zero_fm, _, _ = generator_adversarial_terms(y, fr, fg, idx, 0.0)
    torch.testing.assert_close(zero_gen, generator_loss([y[i] for i in rest])[0])


@pytest.mark.parametrize(
    "version, expected",
    [
        ("apex-gan-48k-cqt", [8, 9, 10]),
        ("codename-ringformer-cqt", [12, 13, 14]),
        ("apex-gan-48k", []),
        ("v2", []),
    ],
)
def test_cqt_indices_find_the_cqt_sub_discriminators(version, expected):
    from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator

    net_d = MultiPeriodDiscriminator(version=version, sample_rate=48000, segment_size=17280)
    assert cqt_indices(net_d) == expected
    for i in expected:
        assert net_d.discriminators[i].descriptor[0] == "Q"
    params = cqt_parameters(net_d, expected)
    assert sum(p.numel() for p in params) == sum(
        p.numel() for i in expected for p in net_d.discriminators[i].parameters()
    )
    norms = effective_weight_norms(net_d, expected)
    assert len(norms) == len(expected) and all(n > 0 for n in norms)


def test_effective_norm_matches_the_saved_weight_g():
    """The logged norm is the number measured on the collapsed run's D_*.pth files: the
    weight-norm magnitudes plus the plain convolutions' weights, biases left out."""
    from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator

    net_d = MultiPeriodDiscriminator(
        version="apex-gan-48k-cqt", sample_rate=48000, segment_size=17280
    )
    state = net_d.state_dict()
    for k, i in enumerate([8, 9, 10]):
        total = 0.0
        for key, value in state.items():
            if not key.startswith(f"discriminators.{i}."):
                continue
            if key.endswith("original1") or key.endswith("bias"):
                continue
            total += float(value.double().pow(2).sum())
        assert effective_weight_norms(net_d, [i])[0] == pytest.approx(total**0.5, rel=1e-6)


def test_non_finite_stop_counts_consecutive_steps_after_the_grace():
    nan = torch.tensor(float("nan"))
    ok = torch.tensor(1.0)
    stop = NonFiniteStop(limit=3, grace=2)
    assert not stop.update(nan, ok) and not stop.update(nan, ok)  # in the grace
    assert not stop.update(nan, ok) and not stop.update(ok, nan)
    assert not stop.update(ok, ok)  # a finite step resets the run of them
    assert not stop.update(ok, torch.tensor(float("inf")))
    assert not stop.update(nan, nan)
    assert stop.update(nan, ok)
    assert stop.take_epoch_count() == 5 and stop.take_epoch_count() == 0


def test_settings_validate_and_default_to_half(tmp_path):
    assert read_cqt_balance({}) == (0.5, 0.5)
    assert read_cqt_balance({"cqt_lr_scale": 1, "cqt_loss_scale": 0}) == (1.0, 0.0)
    for bad in (-0.1, 1.5, "x", float("nan")):
        with pytest.raises(ValueError):
            validate_cqt_scale(bad, "cqt_lr_scale")

    from rvc.train.extract.preparing_files import (
        apply_cqt_balance_settings,
        read_cqt_balance_settings,
    )

    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"train": {"learning_rate": 6e-5}}))
    assert read_cqt_balance_settings(str(folder)) == {"lr_scale": 0.5, "loss_scale": 0.5}
    apply_cqt_balance_settings(str(folder), lr_scale=0.25, loss_scale=None)
    train = json.loads((folder / "config.json").read_text())["train"]
    assert train == {"learning_rate": 6e-5, "cqt_lr_scale": 0.25}
    assert read_cqt_balance_settings(str(folder)) == {"lr_scale": 0.25, "loss_scale": 0.5}
    with pytest.raises(ValueError):
        apply_cqt_balance_settings(str(folder), loss_scale=2)
