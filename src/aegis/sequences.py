"""Fixed-length feature windows per stock and session, with fold-aware scaling and sample selection."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

CLIP = 0.005
FLOOR = 1e-10


class SequencePanel:
    """Dense (session, stock, feature) arrays. A window for decision d covers sessions d-lookback+1 to d."""

    def __init__(self, df, features, log_features, lookback, sector, horizon=5):
        self.lookback, self.horizon = lookback, horizon
        self.dates = pd.DatetimeIndex(sorted(df.date.unique()))
        self.tickers = sorted(df.ticker.unique())
        wide = df.set_index(["date", "ticker"]).reindex(
            pd.MultiIndex.from_product([self.dates, self.tickers]))
        shape = (len(self.dates), len(self.tickers))

        def grid(col):
            return wide[col].to_numpy().reshape(shape)

        self.x = np.stack([np.log(np.maximum(grid(f).astype(float), FLOOR)) if f in log_features
                           else grid(f).astype(float) for f in features], axis=-1).astype(np.float32)
        self.y_ret = grid(f"y_ret_{horizon}").astype(np.float32)
        self.y_rv = grid(f"y_rv_{horizon}").astype(np.float32)
        self.base = np.log(np.maximum(grid("ewma_var").astype(float) * horizon, FLOOR)).astype(np.float32)
        self.label_end = grid(f"label_end_{horizon}").astype("datetime64[ns]")
        sectors = sorted(set(sector[t] for t in self.tickers))
        self.static = np.array([[i, sectors.index(sector[t])] for i, t in enumerate(self.tickers)], dtype=np.int64)
        self.n_sectors = len(sectors)
        n = len(features)
        self.mean, self.std = np.zeros(n, np.float32), np.ones(n, np.float32)
        self.lower, self.upper = np.full(n, -np.inf, np.float32), np.full(n, np.inf, np.float32)
        self._cache = (None, None)
        finite = np.isfinite(self.x).all(axis=-1) & np.isfinite(self.base)
        counts = np.cumsum(finite, axis=0)
        lagged = np.vstack([np.zeros((lookback, shape[1]), int), counts[:-lookback]])
        self.window_ok = (counts - lagged) == lookback

    def fit_scaler(self, cut):
        """Clip bounds and z-score from sessions up to and including cut."""
        rows = self.x[self.dates <= pd.Timestamp(cut)].reshape(-1, self.x.shape[-1])
        self.lower, self.upper = (np.nanquantile(rows, q, axis=0).astype(np.float32) for q in (CLIP, 1 - CLIP))
        clipped = np.clip(rows, self.lower, self.upper)
        self.mean = np.nanmean(clipped, axis=0).astype(np.float32)
        self.std = np.maximum(np.nanstd(clipped, axis=0), 1e-6).astype(np.float32)

    def _scaled(self, device):
        key = (device, self.mean.tobytes(), self.std.tobytes(), self.lower.tobytes(), self.upper.tobytes())
        if self._cache[0] != key:
            z = (np.clip(self.x, self.lower, self.upper) - self.mean) / self.std
            self._cache = (key, dict(
                x=torch.from_numpy(np.nan_to_num(z)).to(device),
                static=torch.from_numpy(self.static).to(device),
                y_ret=torch.from_numpy(np.nan_to_num(self.y_ret)).to(device),
                y_rv=torch.from_numpy(np.nan_to_num(self.y_rv)).to(device),
                base=torch.from_numpy(np.nan_to_num(self.base)).to(device)))
        return self._cache[1]

    def batch(self, d, s, device):
        """Tensors for decisions (session d, stock s): windows, static ids, targets, log base variance."""
        t = self._scaled(device)
        d, s = d.to(device), s.to(device)
        steps = torch.arange(-self.lookback + 1, 1, device=device)
        x = t["x"][d[:, None] + steps, s[:, None]]
        return x, t["static"][s], t["y_ret"][d, s], t["y_rv"][d, s], t["base"][d, s]


@dataclass
class FoldIndex:
    fit_d: np.ndarray
    fit_s: np.ndarray
    stop_d: np.ndarray
    stop_s: np.ndarray
    scored_d: np.ndarray
    scored_s: np.ndarray
    predict_d: np.ndarray
    predict_s: np.ndarray


def fold_indices(panel, fold, early_stop_sessions):
    """Split a fold into fitting, early-stopping, scored and prediction rows, each purged against the next."""
    start, end = map(pd.Timestamp, fold["train"])
    val_start, val_end = map(pd.Timestamp, fold["val"])
    dates = panel.dates
    train_days = np.flatnonzero((dates >= start) & (dates <= end))
    stop_start = dates[train_days[-early_stop_sessions]]
    day = np.broadcast_to(dates.to_numpy()[:, None], panel.label_end.shape)
    labelled = ~np.isnat(panel.label_end) & np.isfinite(panel.y_ret) & np.isfinite(panel.y_rv) & panel.window_ok

    def rows(mask):
        return np.nonzero(mask)

    in_train = (day >= np.datetime64(start)) & (day <= np.datetime64(end))
    in_val = (day >= np.datetime64(val_start)) & (day <= np.datetime64(val_end))
    fit = rows(in_train & (day < np.datetime64(stop_start)) & labelled & (panel.label_end < np.datetime64(stop_start)))
    stop = rows(in_train & (day >= np.datetime64(stop_start)) & labelled & (panel.label_end < np.datetime64(val_start)))
    scored = rows(in_val & labelled & (panel.label_end <= np.datetime64(val_end)))
    predict = rows(in_val & panel.window_ok)
    return FoldIndex(*fit, *stop, *scored, *predict)
