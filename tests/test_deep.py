"""Losses, the shared output head and both forecasters."""

import math

import numpy as np
import pytest
import torch
from scipy import stats

from aegis.deep import ForecastHead, PatchTST, TFT, joint_loss, qlike_loss, student_t_nll

BATCH, T, F = 6, 30, 5
TFT_ARGS = dict(n_features=F, n_tickers=4, n_sectors=2, hidden=16, heads=2, dropout=0.0)
PATCH_ARGS = dict(n_features=F, n_tickers=4, n_sectors=2, lookback=T, hidden=16, heads=2, dropout=0.0,
                  patch=5, stride=5, layers=1)


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(BATCH, T, F, generator=g)
    static = torch.stack([torch.arange(BATCH) % 4, torch.arange(BATCH) % 2], dim=1)
    base = torch.full((BATCH,), math.log(1e-3))
    return x, static, base


def test_student_t_nll_matches_scipy():
    y, mu, sigma, nu = 0.03, 0.01, 0.05, 5.0
    got = student_t_nll(torch.tensor([y]), torch.tensor([mu]), torch.log(torch.tensor([sigma])), torch.tensor(nu))
    assert got.item() == pytest.approx(-stats.t.logpdf(y, df=nu, loc=mu, scale=sigma), rel=1e-5)


def test_qlike_loss_is_smallest_at_the_true_variance():
    y = torch.tensor([0.002])
    at_truth = qlike_loss(y, torch.log(y))
    assert at_truth < qlike_loss(y, torch.log(y * 1.5))
    assert at_truth < qlike_loss(y, torch.log(y / 1.5))


def test_joint_loss_weights_the_variance_term():
    y, rv = torch.tensor([0.01]), torch.tensor([0.002])
    mu, log_sigma, log_rv = torch.tensor([0.0]), torch.tensor([-3.0]), torch.tensor([-6.0])
    nu = torch.tensor(5.0)
    base = student_t_nll(y, mu, log_sigma, nu)
    gap = qlike_loss(rv, log_rv)
    assert joint_loss(y, rv, mu, log_sigma, nu, log_rv, 2.0).item() == pytest.approx((base + 2.0 * gap).item())


def test_head_starts_from_the_baseline_forecast():
    head = ForecastHead(8, dropout=0.0)
    h = torch.randn(3, 8)
    base = torch.full((3,), math.log(1e-3))
    mu, log_sigma, nu, log_rv = head(h, base)
    assert torch.allclose(mu, torch.zeros(3))
    assert torch.allclose(log_rv, base)
    assert torch.allclose(log_sigma, base / 2)
    assert nu.item() > 2


@pytest.mark.parametrize("cls,args", [(TFT, TFT_ARGS), (PatchTST, PATCH_ARGS)])
def test_models_return_one_forecast_per_row(cls, args):
    x, static, base = inputs()
    mu, log_sigma, nu, log_rv = cls(**args)(x, static, base)
    assert mu.shape == log_sigma.shape == log_rv.shape == (BATCH,)


@pytest.mark.parametrize("cls,args", [(TFT, TFT_ARGS), (PatchTST, PATCH_ARGS)])
def test_stock_identity_changes_the_forecast(cls, args):
    torch.manual_seed(0)
    model = cls(**args)
    for p in model.head.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x, static, base = inputs()
    other = static.clone()
    other[:, 0] = (other[:, 0] + 1) % 4
    assert not torch.allclose(model(x, static, base)[3], model(x, other, base)[3])


@pytest.mark.parametrize("cls,args", [(TFT, TFT_ARGS), (PatchTST, PATCH_ARGS)])
def test_same_seed_gives_the_same_forecast(cls, args):
    x, static, base = inputs()
    torch.manual_seed(5)
    a = cls(**args)(x, static, base)[1]
    torch.manual_seed(5)
    b = cls(**args)(x, static, base)[1]
    assert torch.equal(a, b)


@pytest.mark.parametrize("cls,args", [(TFT, TFT_ARGS), (PatchTST, PATCH_ARGS)])
def test_each_model_can_learn_a_simple_variance_pattern(cls, args):
    torch.manual_seed(0)
    n = 256
    x = torch.randn(n, T, F)
    x[:, :, 0] = x[:, -1:, 0]  # first channel holds one value per sequence
    static = torch.zeros(n, 2, dtype=torch.long)
    base = torch.full((n,), math.log(1e-3))
    rv = torch.exp(base + 1.5 * torch.tanh(x[:, -1, 0]))
    y = rv.sqrt() * torch.randn(n)
    model = cls(**args)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def loss():
        mu, ls, nu, lr = model(x, static, base)
        return joint_loss(y, rv, mu, ls, nu, lr, 1.0)

    start = qlike_loss(rv, base).item()
    for _ in range(150):
        opt.zero_grad()
        loss().backward()
        opt.step()
    model.eval()
    assert qlike_loss(rv, model(x, static, base)[3]).item() < start - 0.3


def test_patchtst_splits_the_lookback_into_expected_patches():
    assert PatchTST(**(PATCH_ARGS | {"lookback": 60, "patch": 10, "stride": 5})).n_patches == 11


def test_tft_variable_weights_sum_to_one():
    model = TFT(**TFT_ARGS)
    x, static, base = inputs()
    weights = model.variable_weights(x, static)
    assert weights.shape == (BATCH, T, F)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(BATCH, T))


def test_tft_attention_cannot_see_later_steps():
    torch.manual_seed(0)
    model = TFT(**TFT_ARGS).eval()
    x, static, _ = inputs()
    later = x.clone()
    later[:, 20:] += 5.0
    a = model.encode(x, static)[:, :20]
    b = model.encode(later, static)[:, :20]
    assert torch.allclose(a, b, atol=1e-6)


def test_patchtst_forecast_depends_on_the_latest_patch():
    torch.manual_seed(0)
    model = PatchTST(**PATCH_ARGS).eval()
    for p in model.head.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x, static, base = inputs()
    later = x.clone()
    later[:, -1] += 5.0
    assert not torch.allclose(model(x, static, base)[3], model(later, static, base)[3])
