"""Price features known at the 17:00 decision, and forward labels from next-open execution."""

import numpy as np
import pandas as pd

RETURN_WINDOWS = (1, 5, 21, 63)
VOL_WINDOWS = (5, 21, 63)
LIQUIDITY_WINDOW = 63
BETA_WINDOW = 63
EWMA_LAMBDA = 0.94  # RiskMetrics daily decay


def make_labels(df, horizons=(5, 1)):
    """y_ret_H = log(adj_open[t+1+H] / adj_open[t+1]); y_rv_H sums squared open-to-open returns over the same span."""
    df = df.sort_values(["ticker", "date"])
    out = df[["date", "ticker"]].copy()
    log_open = np.log(df.adj_open)
    squared = log_open.groupby(df.ticker).diff() ** 2
    for h in horizons:
        out[f"y_ret_{h}"] = (log_open.groupby(df.ticker).shift(-(1 + h))
                             - log_open.groupby(df.ticker).shift(-1))
        out[f"y_rv_{h}"] = squared.groupby(df.ticker).transform(
            lambda s: s.rolling(h).sum().shift(-(1 + h)))
        out[f"label_end_{h}"] = df.groupby("ticker").date.shift(-(1 + h))
    return out.reset_index(drop=True)


def _one_ticker(g, spy):
    g = g.sort_values("date").set_index("date")
    r = g.ret_cc
    f = pd.DataFrame(index=g.index)
    for w in RETURN_WINDOWS:
        f[f"ret_{w}"] = r.rolling(w).sum()
    f["overnight"] = np.log(g.adj_open / g.adj_close.shift())
    f["intraday"] = np.log(g.adj_close / g.adj_open)
    for w in VOL_WINDOWS:
        f[f"vol_{w}"] = np.sqrt((r ** 2).rolling(w).mean())
    f["range_21"] = np.log(g.adj_high / g.adj_low).rolling(21).mean()
    ldv = np.log(g.dollar_volume)
    f["dv_z"] = (ldv - ldv.rolling(LIQUIDITY_WINDOW).mean()) / ldv.rolling(LIQUIDITY_WINDOW).std()
    m = spy.reindex(g.index)
    f["beta"] = r.rolling(BETA_WINDOW).cov(m) / m.rolling(BETA_WINDOW).var()
    for w in (5, 21):
        f[f"rel_mkt_{w}"] = f[f"ret_{w}"] - m.rolling(w).sum()
    f["har_d"] = r ** 2
    f["har_w"] = (r ** 2).rolling(5).mean()
    f["har_m"] = (r ** 2).rolling(22).mean()
    f["ewma_var"] = (r ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean()
    return f


def price_features(df, market=("SPY", "QQQ", "IWM", "TLT", "GLD")):
    """One row per (date, non-market ticker). Uses returns and ratios only, never adjusted price levels."""
    is_market = df.ticker.isin(market)
    spy = df[df.ticker == market[0]].set_index("date").ret_cc
    frames = [_one_ticker(g, spy).assign(ticker=t) for t, g in df[~is_market].groupby("ticker")]
    out = pd.concat(frames).reset_index()
    for w in (5, 21):
        out[f"rel_xs_{w}"] = out[f"ret_{w}"] - out.groupby("date")[f"ret_{w}"].transform("mean")
    wide = df[is_market].pivot(index="date", columns="ticker", values="ret_cc")
    ctx = pd.DataFrame(index=wide.index)
    for m in market:
        ctx[f"{m.lower()}_ret_1"] = wide[m]
        ctx[f"{m.lower()}_ret_5"] = wide[m].rolling(5).sum()
    ctx["spy_vol_21"] = np.sqrt((wide[market[0]] ** 2).rolling(21).mean())
    out = out.merge(ctx.reset_index(), on="date", how="left")
    out["dow"] = out.date.dt.dayofweek
    return out
