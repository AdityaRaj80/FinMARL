"""Training must stop on the early-stopping slice, keep its best weights, and repeat exactly from a seed."""

import numpy as np
import pytest
import torch

from aegis.deep import PatchTST
from aegis.sequences import SequencePanel, fold_indices
from aegis.train import ensemble, fit_model, predict, stop_loss
from test_sequences import fold, panel_frame  # noqa: E402

CONFIG = dict(batch_size=32, max_epochs=6, patience=2, learning_rate=1e-3, weight_decay=0.0,
              grad_clip=1.0, variance_loss_weight=1.0)


def setup():
    p = SequencePanel(panel_frame(), features=["f1", "vol_21"], log_features=["vol_21"], lookback=10,
                      sector={"AAA": "x", "BBB": "x", "CCC": "y"})
    idx = fold_indices(p, fold(), early_stop_sessions=20)
    p.fit_scaler(p.dates[idx.fit_d.max()])
    return p, idx


def new_model(p):
    return PatchTST(n_features=2, n_tickers=3, n_sectors=2, lookback=10, hidden=8, heads=2, dropout=0.0,
                    patch=5, stride=5, layers=1)


def test_training_stops_once_patience_runs_out():
    p, idx = setup()
    frozen = CONFIG | {"learning_rate": 0.0}
    _, history = fit_model(lambda: new_model(p), p, idx, frozen, seed=0, device="cpu")
    assert len(history) == 1 + frozen["patience"]


def test_training_returns_the_weights_of_the_best_epoch():
    p, idx = setup()
    model, history = fit_model(lambda: new_model(p), p, idx, CONFIG, seed=0, device="cpu")
    assert stop_loss(model, p, idx, CONFIG, "cpu") == pytest.approx(min(history), rel=1e-5)


def test_the_same_seed_repeats_the_same_run():
    p, idx = setup()
    _, a = fit_model(lambda: new_model(p), p, idx, CONFIG, seed=3, device="cpu")
    _, b = fit_model(lambda: new_model(p), p, idx, CONFIG, seed=3, device="cpu")
    assert a == b


def test_different_seeds_start_from_different_weights():
    p, idx = setup()
    _, a = fit_model(lambda: new_model(p), p, idx, CONFIG, seed=3, device="cpu")
    _, b = fit_model(lambda: new_model(p), p, idx, CONFIG, seed=4, device="cpu")
    assert a != b


def test_a_non_finite_loss_stops_training_with_an_error():
    p, idx = setup()
    p.y_rv[:] = np.inf
    p._cache = (None, None)
    p.y_ret[:] = np.inf
    with pytest.raises(FloatingPointError):
        fit_model(lambda: new_model(p), p, idx, CONFIG, seed=0, device="cpu")


def test_predict_returns_positive_scale_and_variance_for_every_row():
    p, idx = setup()
    model, _ = fit_model(lambda: new_model(p), p, idx, CONFIG, seed=0, device="cpu")
    out = predict(model, p, idx.predict_d, idx.predict_s, "cpu")
    assert len(out.mu) == len(idx.predict_d)
    assert (out.sigma > 0).all() and (out.rv > 0).all() and (out.nu > 2).all()


def test_ensemble_averages_means_and_variances():
    runs = [dict(mu=np.array([0.0]), sigma=np.array([1.0]), nu=np.array([4.0]), rv=np.array([1.0])),
            dict(mu=np.array([0.2]), sigma=np.array([3.0]), nu=np.array([6.0]), rv=np.array([3.0]))]
    e = ensemble(runs)
    assert e["mu"][0] == pytest.approx(0.1)
    assert e["sigma"][0] == pytest.approx(np.sqrt(5.0))
    assert e["nu"][0] == pytest.approx(5.0)
    assert e["rv"][0] == pytest.approx(2.0)
