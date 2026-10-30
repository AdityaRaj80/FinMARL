"""Real-data checks. They need the local tables and are skipped on a machine without them."""

import numpy as np
import pandas as pd
import pytest

from aegis.backtest import simulate
from aegis.data import PROCESSED, DevelopmentOnlyError, load_prices, load_riskfree, universe
from aegis.features import make_labels
from aegis.models import top_k_targets

pytestmark = pytest.mark.skipif(not (PROCESSED / "prices.parquet").exists(), reason="local data not present")


def universe_prices():
    tickers = [t for names in universe()["universe"].values() for t in names]
    px = load_prices()
    return px[px.ticker.isin(tickers)], tickers


def test_development_loader_returns_nothing_after_validation():
    assert load_prices().date.max() <= pd.Timestamp("2022-12-31")


def test_development_loader_refuses_the_frozen_test_year():
    with pytest.raises(DevelopmentOnlyError):
        load_prices("2023-06-30")


def test_backtest_weekly_pnl_equals_the_labels_of_the_held_stocks():
    # Independent reference: holding k names equally from t+1 to t+6 must earn the mean of exp(y_ret_5).
    px, tickers = universe_prices()
    labels = make_labels(px, horizons=(5,)).dropna(subset=["y_ret_5"])
    labels = labels[labels.date >= "2021-01-01"]
    dates = pd.DatetimeIndex(sorted(px.loc[px.date >= "2021-01-01", "date"].unique()))
    opens = px.pivot(index="date", columns="ticker", values="adj_open").loc[dates, tickers]
    targets = top_k_targets(labels, "y_ret_5", k=8, every=5, sector={})
    res = simulate(targets, opens, load_riskfree().rf_period, cost_bps=0)

    wide = labels.pivot(index="date", columns="ticker", values="y_ret_5")
    decisions = targets.dropna().index[:-1]
    for d in decisions[:40]:
        entry = dates[dates.get_loc(d) + 1]
        exit_ = dates[dates.get_loc(d) + 6]
        held = targets.loc[d] > 0
        expected = np.exp(wide.loc[d, held[held].index]).mean()
        assert res.nav[exit_] / res.nav[entry] == pytest.approx(expected, rel=1e-9)
