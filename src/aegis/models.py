"""Baseline forecasters and simple portfolio rules."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from aegis.backtest import apply_limits

HAR_COLUMNS = ["har_d", "har_w", "har_m"]
VARIANCE_FLOOR = 1e-8


class HAR:
    """Log-HAR(1, 5, 22) for realised variance, rescaled so the mean forecast matches the training mean."""

    def __init__(self, per_ticker=False):
        self.per_ticker = per_ticker
        self.models = {}

    @staticmethod
    def _x(df):
        return np.log(df[HAR_COLUMNS].clip(lower=VARIANCE_FLOOR)).to_numpy()

    def _fit_one(self, df, target):
        y = df[target].clip(lower=VARIANCE_FLOOR).to_numpy()
        model = LinearRegression().fit(self._x(df), np.log(y))
        return model, y.mean() / np.exp(model.predict(self._x(df))).mean()

    def fit(self, df, target):
        if self.per_ticker:
            self.models = {t: self._fit_one(g, target) for t, g in df.groupby("ticker")}
        else:
            self.models = {None: self._fit_one(df, target)}
        return self

    def predict(self, df):
        out = np.empty(len(df))
        keys = df.ticker.to_numpy() if self.per_ticker else np.full(len(df), None)
        for key in pd.unique(keys):
            mask = keys == key
            model, scale = self.models[key]
            out[mask] = np.exp(model.predict(self._x(df[mask]))) * scale
        return out


class ClippedRidge:
    """Ridge on standardised features. Clip bounds and scaling come from the training data only."""

    def __init__(self, alpha=1000.0, clip=0.005):
        self.alpha, self.clip = alpha, clip

    def fit(self, x, y):
        self.lower, self.upper = x.quantile(self.clip), x.quantile(1 - self.clip)
        self.scaler = StandardScaler().fit(x.clip(self.lower, self.upper, axis=1))
        self.model = Ridge(alpha=self.alpha).fit(self.scaler.transform(x.clip(self.lower, self.upper, axis=1)), y)
        return self

    def predict(self, x):
        return self.model.predict(self.scaler.transform(x.clip(self.lower, self.upper, axis=1)))


def _schedule(index, every):
    return index[::every]


def top_k_targets(df, score, k, every, sector, max_name=1.0, max_sector=1.0):
    """Hold the k best scores equally. Rows off the rebalance schedule stay NaN, meaning no new decision."""
    wide = df.pivot(index="date", columns="ticker", values=score)
    out = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
    for d in _schedule(wide.index, every):
        best = wide.loc[d].nlargest(k).index
        w = pd.Series(0.0, index=wide.columns)
        w[best] = 1 / k
        out.loc[d] = apply_limits(w, sector, max_name, max_sector)
    return out


def equal_weight_targets(dates, tickers, every):
    out = pd.DataFrame(np.nan, index=dates, columns=tickers)
    out.loc[_schedule(dates, every)] = 1 / len(tickers)
    return out


def inverse_vol_targets(df, every, sector, max_name=0.15, max_sector=0.40):
    wide = df.pivot(index="date", columns="ticker", values="vol_63")
    out = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
    for d in _schedule(wide.index, every):
        w = 1 / wide.loc[d]
        out.loc[d] = apply_limits(w / w.sum(), sector, max_name, max_sector)
    return out
