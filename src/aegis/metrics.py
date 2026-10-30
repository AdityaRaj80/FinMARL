"""Forecast scores and bootstrap intervals."""

import numpy as np
import pandas as pd


def qlike(y, h):
    ratio = np.asarray(y) / np.asarray(h)
    return float(np.mean(ratio - np.log(ratio) - 1))


def oos_r2(y, pred, bench=0.0):
    y, pred = np.asarray(y), np.asarray(pred)
    return float(1 - ((y - pred) ** 2).sum() / ((y - bench) ** 2).sum())


def daily_ic(df, pred, y):
    """Cross-sectional Spearman correlation between forecast and outcome on each date."""
    return df.groupby("date")[[pred, y]].apply(lambda g: g[pred].corr(g[y], method="spearman"))


def ic_tstat(ic, block=5):
    """t-statistic on block means, because 5-day labels on consecutive days overlap."""
    ids = np.arange(len(ic)) // block
    means = pd.Series(np.asarray(ic)).groupby(ids).mean()
    return float(means.mean() / (means.std(ddof=1) / np.sqrt(len(means))))


def gaussian_nll(y, mu, sigma):
    y, mu, sigma = map(np.asarray, (y, mu, sigma))
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)))


def pinball(y, q_pred, q):
    e = np.asarray(y) - np.asarray(q_pred)
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def block_bootstrap_ci(data, stat, block=21, n=2000, seed=0, alpha=0.05):
    """Percentile interval from a circular block bootstrap. data may be a Series or DataFrame."""
    rng = np.random.default_rng(seed)
    size = len(data)
    n_blocks = int(np.ceil(size / block))
    offsets = np.arange(block)
    stats = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, size, n_blocks)
        idx = ((starts[:, None] + offsets) % size).ravel()[:size]
        stats[i] = stat(data.iloc[idx])
    return tuple(float(x) for x in np.quantile(stats, [alpha / 2, 1 - alpha / 2]))
