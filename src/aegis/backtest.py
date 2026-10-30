"""Daily long/cash paper backtest: decide after the close of t, trade at the open of t+1.

P&L is marked open to open from adjusted opens, so dividends and splits create no artificial wealth.
Costs are charged on position changes, as bps per side of the traded NAV.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Result:
    nav: pd.Series
    turnover: pd.Series
    costs: pd.Series
    weights: pd.DataFrame

    @property
    def returns(self):
        return self.nav.pct_change().dropna()


def apply_limits(weights, sector, max_name=0.15, max_sector=0.40):
    """Clip single names, then scale down any sector over its cap. The excess stays in cash."""
    w = weights.clip(upper=max_name)
    groups = w.groupby(pd.Series(sector).reindex(w.index))
    scale = (max_sector / groups.transform("sum")).clip(upper=1.0)
    return w * scale.fillna(1.0)


def simulate(targets, opens, rf_period, cost_bps=5.0, max_turnover=None, fund_first_trade=False):
    """targets: weights decided at each close; a NaN row means no new decision. opens: adjusted opens, same index.

    A trade cut short by max_turnover continues on later sessions until it reaches its target or a new
    decision replaces it. fund_first_trade exempts the initial funding trade from the cap, so strategies
    start invested on the same day.
    """
    tickers = list(opens.columns)
    px = opens.to_numpy(float)
    t = targets.reindex(index=opens.index, columns=tickers).to_numpy(float)
    rf = rf_period.reindex(opens.index).to_numpy(float)
    n, k = px.shape
    held = np.zeros((n, k))
    nav, turnover, costs = np.ones(n), np.zeros(n), np.zeros(n)
    w = np.zeros(k)
    goal, funded = None, not fund_first_trade
    for i in range(n - 1):
        growth = px[i + 1] / px[i]
        value = (w * growth).sum() + (1 - w.sum()) * (1 + rf[i])
        w = w * growth / value
        nav[i + 1] = nav[i] * value
        if not np.isnan(t[i]).any():
            goal = t[i]
        if goal is not None:
            delta = goal - w
            traded = np.abs(delta).sum()
            capped = funded and max_turnover is not None and traded > max_turnover
            if capped:
                delta, traded = delta * max_turnover / traded, max_turnover
            else:
                goal = None
            funded = True
            w = w + delta
            costs[i + 1] = traded * cost_bps / 1e4
            turnover[i + 1] = traded
            nav[i + 1] *= 1 - costs[i + 1]
        held[i + 1] = w
    idx = opens.index
    return Result(pd.Series(nav, idx), pd.Series(turnover, idx), pd.Series(costs, idx),
                  pd.DataFrame(held, idx, tickers))


def annual_sharpe(returns, rf_period):
    excess = returns - rf_period.reindex(returns.index)
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(nav):
    return (nav / nav.cummax() - 1).min()
