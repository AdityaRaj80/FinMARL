"""Forecast scores must behave correctly on cases with known answers."""

import numpy as np
import pandas as pd
import pytest

from aegis.metrics import block_bootstrap_ci, daily_ic, gaussian_nll, ic_tstat, oos_r2, qlike


def test_qlike_is_zero_for_a_perfect_variance_forecast():
    y = np.array([0.001, 0.002, 0.004])
    assert qlike(y, y) == pytest.approx(0.0)


def test_qlike_punishes_under_prediction_more_than_over_prediction():
    y = np.array([0.002])
    assert qlike(y, y / 2) > qlike(y, y * 2)


def test_r2_against_a_zero_forecast_is_zero_for_zero_predictions():
    y = np.array([0.01, -0.02, 0.03])
    assert oos_r2(y, np.zeros(3)) == pytest.approx(0.0)


def test_r2_is_one_for_a_perfect_forecast():
    y = np.array([0.01, -0.02, 0.03])
    assert oos_r2(y, y) == pytest.approx(1.0)


def test_r2_is_negative_when_the_forecast_is_worse_than_the_benchmark():
    y = np.array([0.01, -0.02, 0.03])
    assert oos_r2(y, -y) < 0


def test_daily_ic_is_one_when_ranks_match_on_every_date():
    df = pd.DataFrame({"date": [1, 1, 1, 2, 2, 2], "y": [1, 2, 3, 3, 2, 1], "pred": [10, 20, 30, 5, 4, 3]})
    assert daily_ic(df, "pred", "y").tolist() == pytest.approx([1.0, 1.0])


def test_ic_tstat_uses_non_overlapping_blocks():
    # 5 identical values per block: 20 observations contain only 4 independent ones.
    block_means = np.array([0.01, 0.03, 0.02, 0.04])
    ic = pd.Series(np.repeat(block_means, 5))
    expected = block_means.mean() / (block_means.std(ddof=1) / np.sqrt(4))
    assert ic_tstat(ic, block=5) == pytest.approx(expected)


def test_gaussian_nll_of_a_standard_normal_at_zero():
    assert gaussian_nll(np.array([0.0]), np.array([0.0]), np.array([1.0])) == pytest.approx(0.5 * np.log(2 * np.pi))


def test_bootstrap_interval_contains_the_estimate_and_is_repeatable():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 750))
    a = block_bootstrap_ci(r, lambda s: s.mean(), block=21, n=300, seed=7)
    b = block_bootstrap_ci(r, lambda s: s.mean(), block=21, n=300, seed=7)
    assert a == b
    assert a[0] < r.mean() < a[1]
