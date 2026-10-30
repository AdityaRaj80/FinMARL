"""Baseline models and portfolio rules on inputs with hand-checkable answers."""

import numpy as np
import pandas as pd
import pytest

from aegis.models import HAR, ClippedRidge, equal_weight_targets, inverse_vol_targets, top_k_targets


def har_panel(n=4000, seed=0, tickers=("AAA", "BBB")):
    rng = np.random.default_rng(seed)
    frames = []
    for i, t in enumerate(tickers):
        d, w, m = (np.exp(rng.normal(-7 - i, 1.0, n)) for _ in range(3))
        y = np.exp(0.5 * i + 0.2 + 0.5 * np.log(d) + 0.3 * np.log(w) + 0.1 * np.log(m) + rng.normal(0, 0.05, n))
        frames.append(pd.DataFrame({"ticker": t, "har_d": d, "har_w": w, "har_m": m, "y": y}))
    return pd.concat(frames, ignore_index=True)


def test_har_recovers_a_known_log_relationship():
    df = har_panel(tickers=("AAA",))
    model = HAR().fit(df, "y")
    new = pd.DataFrame({"ticker": ["AAA"], "har_d": [1e-3], "har_w": [1e-3], "har_m": [1e-3]})
    truth = np.exp(0.2 + 0.9 * np.log(1e-3))
    assert model.predict(new)[0] == pytest.approx(truth, rel=0.02)


def test_har_mean_forecast_matches_the_training_mean():
    # Exponentiating a log forecast is biased low; QLIKE punishes that, so the level is rescaled on train.
    df = har_panel(seed=1, tickers=("AAA",))
    df["y"] *= np.exp(np.random.default_rng(2).normal(0, 1.0, len(df)))
    model = HAR().fit(df, "y")
    assert model.predict(df).mean() == pytest.approx(df.y.mean())


def test_per_stock_har_keeps_each_stocks_own_level():
    df = har_panel(seed=3)
    model = HAR(per_ticker=True).fit(df, "y")
    new = pd.DataFrame({"ticker": ["AAA", "BBB"], "har_d": 1e-3, "har_w": 1e-3, "har_m": 1e-3})
    p = model.predict(new)
    assert p[1] / p[0] == pytest.approx(np.exp(0.5), rel=0.05)


def test_clipped_ridge_caps_inputs_at_training_bounds():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(0, 1, 2000)})
    y = pd.Series(0.5 * x.a.to_numpy())
    model = ClippedRidge(alpha=1.0, clip=0.01).fit(x, y)
    upper = x.a.quantile(0.99)
    at_bound = model.predict(pd.DataFrame({"a": [upper]}))[0]
    far_out = model.predict(pd.DataFrame({"a": [1000.0]}))[0]
    assert far_out == pytest.approx(at_bound)


def scores_frame():
    dates = pd.bdate_range("2021-01-04", periods=6)
    rows = [(d, t, s) for d in dates for t, s in zip("ABCD", (4.0, 3.0, 2.0, 1.0))]
    return pd.DataFrame(rows, columns=["date", "ticker", "score"])


def test_top_k_holds_the_best_scores_equally():
    t = top_k_targets(scores_frame(), "score", k=2, every=1, sector={})
    assert t.iloc[0].to_dict() == {"A": 0.5, "B": 0.5, "C": 0.0, "D": 0.0}


def test_top_k_only_rebalances_on_schedule():
    t = top_k_targets(scores_frame(), "score", k=2, every=5, sector={})
    assert t.notna().all(axis=1).tolist() == [True, False, False, False, False, True]


def test_top_k_applies_the_name_cap():
    t = top_k_targets(scores_frame(), "score", k=2, every=1, sector={}, max_name=0.15)
    assert t.iloc[0]["A"] == pytest.approx(0.15)


def test_equal_weight_splits_nav_across_names():
    dates = pd.bdate_range("2021-01-04", periods=3)
    t = equal_weight_targets(dates, list("ABCD"), every=21)
    assert t.iloc[0].tolist() == [0.25] * 4


def test_inverse_vol_gives_the_calmer_stock_more_weight():
    df = pd.DataFrame({"date": pd.Timestamp("2021-01-04"), "ticker": ["A", "B"], "vol_63": [0.01, 0.03]})
    t = inverse_vol_targets(df, every=21, sector={}, max_name=1.0, max_sector=1.0)
    assert t.iloc[0]["A"] == pytest.approx(0.75)
    assert t.iloc[0]["B"] == pytest.approx(0.25)
