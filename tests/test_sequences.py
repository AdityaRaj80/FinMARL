"""Sequence windows must end at the decision session, and scaling and sample choice must respect each fold."""

import numpy as np
import pandas as pd
import pytest
import torch

from aegis.sequences import SequencePanel, fold_indices


def panel_frame(n_dates=120, tickers=("AAA", "BBB", "CCC")):
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    rows = []
    for s, t in enumerate(tickers):
        for d, date in enumerate(dates):
            rows.append({"date": date, "ticker": t, "f1": d + 100.0 * s, "vol_21": 0.01 + d * 1e-4,
                         "ewma_var": 1e-4, "y_ret_5": 0.01 * s, "y_rv_5": 1e-3,
                         "label_end_5": dates[d + 6] if d + 6 < n_dates else pd.NaT})
    return pd.DataFrame(rows)


def make(df=None, lookback=10):
    df = panel_frame() if df is None else df
    return SequencePanel(df, features=["f1", "vol_21"], log_features=["vol_21"], lookback=lookback,
                         sector={"AAA": "x", "BBB": "x", "CCC": "y"})


def identity_scale(p):
    p.mean[:] = 0.0
    p.std[:] = 1.0
    p.lower[:] = -np.inf
    p.upper[:] = np.inf


def test_window_ends_at_the_decision_session():
    p = make()
    identity_scale(p)
    d = 30
    x, *_ = p.batch(torch.tensor([d]), torch.tensor([1]), device="cpu")
    assert x[0, -1, 0].item() == pytest.approx(d + 100.0)
    assert x[0, 0, 0].item() == pytest.approx(d - 9 + 100.0)


def test_window_has_no_value_from_after_the_decision():
    df = panel_frame()
    p = make(df)
    identity_scale(p)
    d = 30
    before, *_ = p.batch(torch.tensor([d]), torch.tensor([0]), device="cpu")
    shocked = df.copy()
    shocked.loc[shocked.date > p.dates[d], "f1"] = 1e6
    q = make(shocked)
    identity_scale(q)
    after, *_ = q.batch(torch.tensor([d]), torch.tensor([0]), device="cpu")
    assert torch.equal(before, after)


def test_log_features_are_logged_before_scaling():
    p = make()
    identity_scale(p)
    x, *_ = p.batch(torch.tensor([30]), torch.tensor([0]), device="cpu")
    assert x[0, -1, 1].item() == pytest.approx(np.log(0.01 + 30 * 1e-4), rel=1e-5)


def test_scaler_ignores_everything_after_its_fit_date():
    df = panel_frame()
    p = make(df)
    cut = p.dates[60]
    p.fit_scaler(cut)
    shocked = df.copy()
    shocked.loc[shocked.date > cut, "f1"] = 1e6
    q = make(shocked)
    q.fit_scaler(cut)
    assert np.allclose(p.mean, q.mean) and np.allclose(p.std, q.std)


def test_batch_returns_targets_and_the_base_variance_for_the_horizon():
    p = make()
    identity_scale(p)
    _, static, y_ret, y_rv, base = p.batch(torch.tensor([30]), torch.tensor([2]), device="cpu")
    assert y_ret.item() == pytest.approx(0.02)
    assert y_rv.item() == pytest.approx(1e-3)
    assert base.item() == pytest.approx(np.log(5e-4), rel=1e-5)  # log(5 * daily EWMA variance)
    assert static[0, 0].item() == 2  # ticker id
    assert static[0, 1].item() == 1  # sector id


def fold():
    return {"train": ["2019-01-01", "2019-04-30"], "val": ["2019-05-01", "2019-06-30"]}


def test_fit_labels_finish_before_early_stopping_rows_begin():
    p = make()
    idx = fold_indices(p, fold(), early_stop_sessions=20)
    fit_end = p.label_end[idx.fit_d, idx.fit_s].max()
    assert fit_end < p.dates[idx.stop_d.min()]


def test_early_stopping_rows_finish_before_the_scoring_period():
    p = make()
    idx = fold_indices(p, fold(), early_stop_sessions=20)
    assert p.label_end[idx.stop_d, idx.stop_s].max() < pd.Timestamp("2019-05-01")


def test_scored_rows_stay_inside_the_validation_period():
    p = make()
    idx = fold_indices(p, fold(), early_stop_sessions=20)
    assert p.dates[idx.scored_d].min() >= pd.Timestamp("2019-05-01")
    assert p.label_end[idx.scored_d, idx.scored_s].max() <= pd.Timestamp("2019-06-30")


def test_decisions_without_a_full_window_are_left_out():
    p = make(lookback=10)
    idx = fold_indices(p, fold(), early_stop_sessions=20)
    assert idx.fit_d.min() >= 9


def test_predict_rows_cover_every_validation_decision():
    p = make()
    idx = fold_indices(p, fold(), early_stop_sessions=20)
    days = np.unique(idx.predict_d)
    in_val = np.flatnonzero((p.dates >= "2019-05-01") & (p.dates <= "2019-06-30"))
    assert set(days) == set(in_val)
