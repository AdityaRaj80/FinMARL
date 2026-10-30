"""Backtest non-negotiables from the plan: cash, reference P&L, cost on changes, next-open execution, limits."""

import numpy as np
import pandas as pd
import pytest

from aegis.backtest import apply_limits, annual_sharpe, max_drawdown, simulate


def opens(values, tickers=("AAA", "BBB")):
    dates = pd.bdate_range("2021-01-04", periods=len(values))
    return pd.DataFrame(values, index=dates, columns=list(tickers), dtype=float)


def flat_rf(index, per_session=0.0):
    return pd.Series(per_session, index=index)


def test_no_positions_earn_exactly_cash():
    px = opens([[100, 50]] * 5)
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    res = simulate(w, px, flat_rf(px.index, 0.0001), cost_bps=5)
    assert res.nav.iloc[-1] == pytest.approx(1.0001 ** 4)


def test_zero_fee_full_position_matches_the_price_ratio():
    px = opens([[100, 50], [110, 50], [121, 50], [99, 50], [105, 50]])
    w = pd.DataFrame([[1.0, 0.0]] * 5, index=px.index, columns=px.columns)
    res = simulate(w, px, flat_rf(px.index), cost_bps=0)
    # Decided at the close of day 0, entered at the open of day 1, marked at the open of day 4.
    assert res.nav.iloc[-1] == pytest.approx(105 / 110)


def test_signal_known_at_close_cannot_capture_that_days_open_to_open_move():
    px = opens([[100, 100], [200, 100], [200, 100]])
    w = pd.DataFrame([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], index=px.index, columns=px.columns)
    res = simulate(w, px, flat_rf(px.index), cost_bps=0)
    # Deciding on day 1 cannot profit from the jump into day 1's open.
    assert res.nav.iloc[-1] == pytest.approx(1.0)


def test_cost_is_charged_on_position_changes_only():
    px = opens([[100, 100]] * 6)
    w = pd.DataFrame([[0.5, 0.5]] * 6, index=px.index, columns=px.columns)
    res = simulate(w, px, flat_rf(px.index), cost_bps=10)
    assert res.nav.iloc[-1] == pytest.approx(1 - 0.001)  # one purchase of 100% of NAV, nothing afterwards


def test_drift_back_to_target_is_a_real_trade():
    px = opens([[100, 100], [100, 100], [200, 100], [200, 100]])
    w = pd.DataFrame([[0.5, 0.5]] * 4, index=px.index, columns=px.columns)
    res = simulate(w, px, flat_rf(px.index), cost_bps=0)
    # After AAA doubles, weights are 2/3 and 1/3; returning to 50/50 turns over 1/3 of NAV.
    assert res.turnover.iloc[2] == pytest.approx(1 / 3)


def test_turnover_cap_limits_each_days_trading():
    px = opens([[100, 100]] * 5)
    w = pd.DataFrame([[1.0, 0.0]] * 5, index=px.index, columns=px.columns)
    res = simulate(w, px, flat_rf(px.index), cost_bps=0, max_turnover=0.20)
    assert res.turnover.max() == pytest.approx(0.20)
    assert res.weights["AAA"].iloc[1] == pytest.approx(0.20)


def test_capped_trade_keeps_working_towards_the_last_target():
    px = opens([[100, 100]] * 7)
    w = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    w.iloc[0] = [1.0, 0.0]  # a single rebalance decision; later rows are "no new decision"
    res = simulate(w, px, flat_rf(px.index), cost_bps=0, max_turnover=0.20)
    assert res.weights["AAA"].iloc[-1] == pytest.approx(1.0)


def test_finished_rebalance_lets_positions_drift():
    px = opens([[100, 100], [100, 100], [200, 100], [200, 100]])
    w = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    w.iloc[0] = [0.5, 0.5]
    res = simulate(w, px, flat_rf(px.index), cost_bps=0)
    assert res.turnover.iloc[2:].sum() == 0
    assert res.weights["AAA"].iloc[-1] == pytest.approx(2 / 3)


def test_first_funding_trade_can_ignore_the_turnover_cap():
    px = opens([[100, 100]] * 4)
    w = pd.DataFrame([[1.0, 0.0]] * 4, index=px.index, columns=px.columns)
    res = simulate(w, px, flat_rf(px.index), cost_bps=0, max_turnover=0.20, fund_first_trade=True)
    assert res.weights["AAA"].iloc[1] == pytest.approx(1.0)


def test_single_name_cap_sends_the_excess_to_cash():
    w = pd.Series({"AAA": 0.5, "BBB": 0.5})
    out = apply_limits(w, sector={"AAA": "x", "BBB": "y"}, max_name=0.15, max_sector=0.40)
    assert out["AAA"] == pytest.approx(0.15)
    assert out.sum() == pytest.approx(0.30)


def test_sector_cap_scales_names_in_that_sector():
    w = pd.Series({"AAA": 0.15, "BBB": 0.15, "CCC": 0.15, "DDD": 0.15})
    sector = {"AAA": "x", "BBB": "x", "CCC": "x", "DDD": "y"}
    out = apply_limits(w, sector=sector, max_name=0.15, max_sector=0.40)
    assert out[["AAA", "BBB", "CCC"]].sum() == pytest.approx(0.40)
    assert out["DDD"] == pytest.approx(0.15)


def test_sharpe_of_a_known_series():
    r = pd.Series([0.01, -0.01] * 500) + 0.001
    rf = pd.Series(0.0, index=r.index)
    expected = 0.001 / r.std() * np.sqrt(252)
    assert annual_sharpe(r, rf) == pytest.approx(expected)


def test_max_drawdown_of_a_known_path():
    nav = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(nav) == pytest.approx(0.9 / 1.2 - 1)
