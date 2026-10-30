"""Labels follow the 17:00 decision convention; features at t never depend on anything after t."""

import numpy as np
import pandas as pd
import pytest

from aegis.features import make_labels, price_features


def synthetic_prices(tickers=("AAA", "BBB", "SPY", "QQQ", "IWM", "TLT", "GLD"), n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    frames = []
    for t in tickers:
        r = rng.normal(0.0005, 0.01, n)
        close = 100 * np.exp(np.cumsum(r))
        openp = close * np.exp(rng.normal(0, 0.002, n))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": t, "adj_open": openp, "adj_close": close,
            "adj_high": np.maximum(openp, close) * 1.005, "adj_low": np.minimum(openp, close) * 0.995,
            "dollar_volume": rng.lognormal(18, 0.3, n)}))
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby("ticker")
    df["ret_cc"] = np.log(df.adj_close / g.adj_close.shift())
    df["ret_oo"] = np.log(df.adj_open / g.adj_open.shift())
    return df


def one_ticker(opens):
    dates = pd.bdate_range("2019-01-01", periods=len(opens))
    return pd.DataFrame({"date": dates, "ticker": "AAA", "adj_open": np.asarray(opens, float)})


def test_five_day_label_runs_from_next_open_to_the_open_five_sessions_later():
    opens = [100, 101, 103, 102, 105, 107, 110, 111, 112]
    y = make_labels(one_ticker(opens), horizons=(5,))
    assert y.y_ret_5.iloc[0] == pytest.approx(np.log(110 / 101))  # enter t+1, exit t+6


def test_label_end_date_is_the_exit_session():
    df = one_ticker([100, 101, 103, 102, 105, 107, 110, 111, 112])
    y = make_labels(df, horizons=(5,))
    assert y.label_end_5.iloc[0] == df.date.iloc[6]


def test_realised_variance_label_sums_squared_open_to_open_returns():
    opens = [100, 101, 103, 102, 105, 107, 110, 111, 112]
    y = make_labels(one_ticker(opens), horizons=(5,))
    r = np.diff(np.log(opens))[1:6]
    assert y.y_rv_5.iloc[0] == pytest.approx((r ** 2).sum())


def test_decisions_without_a_finished_label_stay_missing():
    y = make_labels(one_ticker([100, 101, 103, 102, 105, 107, 110, 111, 112]), horizons=(5,))
    assert y.y_ret_5.iloc[2] == pytest.approx(np.log(112 / 102))  # last complete decision
    assert y.y_ret_5.iloc[3:].isna().all()


def test_features_at_t_ignore_all_later_prices():
    df = synthetic_prices()
    cut = df.date.drop_duplicates().iloc[90]
    base = price_features(df, market=("SPY", "QQQ", "IWM", "TLT", "GLD"))
    shocked = df.copy()
    later = shocked.date > cut
    shocked.loc[later, ["adj_open", "adj_close", "adj_high", "adj_low"]] *= 3.0
    shocked.loc[later, "ret_cc"] = 0.5
    shocked.loc[later, "ret_oo"] = 0.5
    shocked.loc[later, "dollar_volume"] *= 50
    after = price_features(shocked, market=("SPY", "QQQ", "IWM", "TLT", "GLD"))
    key = ["date", "ticker"]
    a = base[base.date <= cut].sort_values(key).reset_index(drop=True)
    b = after[after.date <= cut].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_scaled_prices_give_identical_features():
    # Adjusted levels embed later dividends, so features may only use returns and ratios.
    df = synthetic_prices()
    scaled = df.copy()
    scaled[["adj_open", "adj_close", "adj_high", "adj_low"]] *= 0.73
    key = ["date", "ticker"]
    a = price_features(df, market=("SPY", "QQQ", "IWM", "TLT", "GLD")).sort_values(key).reset_index(drop=True)
    b = price_features(scaled, market=("SPY", "QQQ", "IWM", "TLT", "GLD")).sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_ewma_variance_follows_the_riskmetrics_recursion():
    df = synthetic_prices()
    f = price_features(df, market=("SPY", "QQQ", "IWM", "TLT", "GLD"))
    a = df[df.ticker == "AAA"].reset_index(drop=True)
    v = a.ret_cc.iloc[1] ** 2
    for r in a.ret_cc.iloc[2:6]:
        v = 0.94 * v + 0.06 * r ** 2
    got = f[(f.ticker == "AAA") & (f.date == a.date.iloc[5])].ewma_var.iloc[0]
    assert got == pytest.approx(v)


def test_five_day_return_feature_sums_the_last_five_close_to_close_returns():
    df = synthetic_prices()
    f = price_features(df, market=("SPY", "QQQ", "IWM", "TLT", "GLD"))
    a = df[df.ticker == "AAA"].reset_index(drop=True)
    expected = a.ret_cc.iloc[56:61].sum()
    got = f[(f.ticker == "AAA") & (f.date == a.date.iloc[60])].ret_5.iloc[0]
    assert got == pytest.approx(expected)
