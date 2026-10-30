"""Gate B0: price-only baselines on the development folds (2019-2022). Nothing from 2023 onward is read.

  python scripts/run_baselines.py

Every model is fitted per fold on past data only, with fixed first-run settings (no tuning), then scored on
that fold's validation year. Strategies trade weekly at the next open under the plan's hard limits.

Outputs: reports/baselines/*.csv and data/processed/baseline_oos.parquet (out-of-sample forecasts).
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aegis.backtest import annual_sharpe, max_drawdown, simulate  # noqa: E402
from aegis.data import PROCESSED, load_prices, load_riskfree, splits, universe  # noqa: E402
from aegis.features import make_labels, price_features  # noqa: E402
from aegis.metrics import (block_bootstrap_ci, daily_ic, gaussian_nll, ic_tstat, oos_r2, pinball,  # noqa: E402
                           qlike)
from aegis.models import (HAR, VARIANCE_FLOOR, ClippedRidge, equal_weight_targets,  # noqa: E402
                          inverse_vol_targets, top_k_targets)
from aegis.splits import fold_frames  # noqa: E402

OUT = ROOT / "reports" / "baselines"
SEED = 0
TOP_K = 8                 # 12.5% per name, inside the 15% name cap
REBALANCE_EVERY = 5       # matches the 5-day label
MONTHLY = 21
MAX_NAME, MAX_SECTOR, MAX_TURNOVER = 0.15, 0.40, 0.20
BASE_COST_BPS = 5
COST_GRID = (5, 10, 20, 50)
N_RANDOM = 200
Z90 = 1.645
LGBM = dict(n_estimators=200, learning_rate=0.02, num_leaves=15, min_child_samples=500, subsample=0.7,
            subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, random_state=SEED, n_jobs=-1,
            deterministic=True, force_row_wise=True, verbose=-1)
RETURN_MODELS = ["zero", "train_mean", "ridge_pooled", "ridge_stock", "lgbm_pooled", "lgbm_xs", "lgbm_shuffled"]
SKIP = {"date", "ticker", "har_d", "har_w", "har_m", "ewma_var"}


def build_panel():
    cfg = universe()
    tickers = [t for names in cfg["universe"].values() for t in names]
    bench = cfg["benchmarks"]
    px = load_prices()
    px = px[px.ticker.isin(tickers + bench)]
    panel = price_features(px, market=tuple(bench)).merge(
        make_labels(px[px.ticker.isin(tickers)], horizons=(5, 1)), on=["date", "ticker"])
    panel = panel[panel.date >= splits()["periods"]["train"][0].isoformat()].copy()
    features = [c for c in panel.columns if c not in SKIP and not c.startswith(("y_", "label_end"))]
    for h in (5, 1):
        panel[f"y_xs_{h}"] = panel[f"y_ret_{h}"] - panel.groupby("date")[f"y_ret_{h}"].transform("mean")
    panel = panel.dropna(subset=features).reset_index(drop=True)
    return panel, features, tickers, px, {t: s for s, names in cfg["universe"].items() for t in names}


def fit_log_variance(model, train, test, features, target):
    """Fit on log variance, then rescale so the mean forecast matches the training mean."""
    y = train[target].clip(lower=VARIANCE_FLOOR)
    model.fit(train[features], np.log(y))
    return np.exp(model.predict(test[features])) * y.mean() / np.exp(model.predict(train[features])).mean()


def shuffled_target(train, target, seed):
    """Control: keep each day's cross-section together but move it to a random date."""
    dates = train.date.drop_duplicates().to_numpy()
    moved = dict(zip(dates, np.random.default_rng(seed).permutation(dates)))
    table = train.set_index(["date", "ticker"])[target]
    keys = pd.MultiIndex.from_arrays([train.date.map(moved), train.ticker])
    return table.reindex(keys).fillna(0.0).to_numpy()


def fold_forecasts(panel, features, fold, h):
    train, _ = fold_frames(panel, fold, f"label_end_{h}")
    span = pd.to_datetime(fold["val"])
    test = panel[panel.date.between(span[0], span[1])].copy()
    y = f"y_ret_{h}"
    test["zero"] = 0.0
    test["train_mean"] = train[y].mean()
    test["ridge_pooled"] = ClippedRidge().fit(train[features], train[y]).predict(test[features])
    stock = pd.Series(0.0, index=test.index)
    for t, g in train.groupby("ticker"):
        rows = test.ticker == t
        stock[rows] = ClippedRidge().fit(g[features], g[y]).predict(test.loc[rows, features])
    test["ridge_stock"] = stock
    test["lgbm_pooled"] = lgb.LGBMRegressor(**LGBM).fit(train[features], train[y]).predict(test[features])
    test["lgbm_xs"] = lgb.LGBMRegressor(**LGBM).fit(train[features], train[f"y_xs_{h}"]).predict(test[features])
    fake = shuffled_target(train, y, SEED)
    test["lgbm_shuffled"] = lgb.LGBMRegressor(**LGBM).fit(train[features], fake).predict(test[features])

    rv = f"y_rv_{h}"
    test["ewma"] = test.ewma_var * h
    test["roll_21"] = test.vol_21 ** 2 * h
    test["har_pooled"] = HAR().fit(train, rv).predict(test)
    test["har_stock"] = HAR(per_ticker=True).fit(train, rv).predict(test)
    test["ridge_vol"] = fit_log_variance(ClippedRidge(), train, test, features, rv)
    test["lgbm_vol"] = fit_log_variance(lgb.LGBMRegressor(**LGBM), train, test, features, rv)
    test["scored"] = test[f"label_end_{h}"].notna() & (test[f"label_end_{h}"] <= span[1])
    return test.assign(fold=span[0].year)


def score_returns(oos, h):
    y = f"y_ret_{h}"
    rows = []
    for name, group in [("all", oos), *oos.groupby("fold")]:
        s = group[group.scored]
        for m in RETURN_MODELS + ["momentum"]:
            pred = "ret_21" if m == "momentum" else m
            row = {"horizon": h, "fold": name, "model": m}
            if m == "lgbm_xs":  # forecasts the cross-sectional return, so it is scored against that target
                row["r2_vs_zero"] = oos_r2(s[f"y_xs_{h}"], s[pred])
            elif m != "momentum":
                row["r2_vs_zero"] = oos_r2(s[y], s[pred])
                row["r2_vs_train_mean"] = oos_r2(s[y], s[pred], s.train_mean)
            if m not in ("zero", "train_mean"):
                ic = daily_ic(s, pred, y).dropna()
                row["ic"], row["ic_t"] = ic.mean(), ic_tstat(ic, block=max(h, 1))
            rows.append(row)
    return pd.DataFrame(rows)


def score_variance(oos, h):
    y = f"y_rv_{h}"
    rows = []
    for name, group in [("all", oos), *oos.groupby("fold")]:
        s = group[group.scored]
        for m in ("ewma", "roll_21", "har_pooled", "har_stock", "ridge_vol", "lgbm_vol"):
            rows.append({"horizon": h, "fold": name, "model": m, "qlike": qlike(s[y].clip(lower=VARIANCE_FLOOR), s[m]),
                         "mae_x1e4": (s[y] - s[m]).abs().mean() * 1e4,
                         "r2_log": oos_r2(np.log(s[y].clip(lower=VARIANCE_FLOOR)), np.log(s[m]),
                                          np.log(s[y].clip(lower=VARIANCE_FLOOR)).mean())})
    return pd.DataFrame(rows)


def variance_gaps(oos, reference="har_pooled"):
    """Paired block bootstrap of the daily QLIKE gap to the reference. Negative means better than the reference."""
    s = oos[oos.scored]
    y = s.y_rv_5.clip(lower=VARIANCE_FLOOR)

    def daily_loss(m):
        ratio = y / s[m]
        return (ratio - np.log(ratio) - 1).groupby(s.date).mean()

    rows = []
    for m in ("ewma", "roll_21", "har_stock", "ridge_vol", "lgbm_vol"):
        gap = daily_loss(m) - daily_loss(reference)
        lo, hi = block_bootstrap_ci(gap, lambda x: x.mean(), seed=SEED)
        rows.append({"model": m, "reference": reference, "qlike_gap": gap.mean(), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def score_distribution(oos, vol_model):
    """The bar for the Student-t head: a plain Gaussian built from the best variance baseline."""
    s = oos[oos.scored]
    sigma = np.sqrt(s[vol_model])
    rows = []
    for mu_name in ("zero", "train_mean", "ridge_pooled"):
        mu = s[mu_name]
        q05, q95 = mu - Z90 * sigma, mu + Z90 * sigma
        rows.append({"mu": mu_name, "sigma": vol_model, "gaussian_nll": gaussian_nll(s.y_ret_5, mu, sigma),
                     "coverage_90": ((s.y_ret_5 >= q05) & (s.y_ret_5 <= q95)).mean(),
                     "pinball_avg": np.mean([pinball(s.y_ret_5, q05, 0.05), pinball(s.y_ret_5, mu, 0.5),
                                             pinball(s.y_ret_5, q95, 0.95)])})
    return pd.DataFrame(rows)


def strategy_targets(oos, tickers, sector):
    limits = dict(sector=sector, max_name=MAX_NAME, max_sector=MAX_SECTOR)
    dates = pd.DatetimeIndex(sorted(oos.date.unique()))
    out = {"equal_weight": equal_weight_targets(dates, tickers, MONTHLY),
           "inverse_vol": inverse_vol_targets(oos, MONTHLY, **limits),
           "momentum": top_k_targets(oos, "ret_21", TOP_K, REBALANCE_EVERY, **limits)}
    for m in ("ridge_pooled", "ridge_stock", "lgbm_pooled", "lgbm_xs"):
        out[m] = top_k_targets(oos, m, TOP_K, REBALANCE_EVERY, **limits)
    return out


def random_targets(oos, tickers, sector, seed):
    rng = np.random.default_rng(seed)
    s = oos[["date", "ticker"]].assign(score=rng.random(len(oos)))
    return top_k_targets(s, "score", TOP_K, REBALANCE_EVERY, sector, MAX_NAME, MAX_SECTOR)


def stats(res, rf, spy_ret):
    r = res.returns
    years = len(r) / 252
    excess = r - rf.reindex(r.index)
    lo, hi = block_bootstrap_ci(excess, lambda x: x.mean() / x.std() * np.sqrt(252), seed=SEED)
    beta = r.cov(spy_ret.reindex(r.index)) / spy_ret.reindex(r.index).var()
    return {"ann_return": res.nav.iloc[-1] ** (1 / years) - 1, "ann_vol": r.std() * np.sqrt(252),
            "sharpe": annual_sharpe(r, rf), "sharpe_lo": lo, "sharpe_hi": hi, "max_drawdown": max_drawdown(res.nav),
            "turnover_per_year": res.turnover.sum() / years, "cost_per_year": res.costs.sum() / years,
            "beta_to_spy": beta}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    panel, features, tickers, px, sector = build_panel()
    assert panel.date.max() <= pd.Timestamp("2022-12-31"), "frozen data reached the baseline run"
    print(f"panel {panel.shape}, {len(features)} features, {panel.date.min().date()} -> {panel.date.max().date()}")

    folds = splits()["folds"]
    forecasts = {h: pd.concat([fold_forecasts(panel, features, f, h) for f in folds], ignore_index=True) for h in (5, 1)}
    oos = forecasts[5]
    keep = ["date", "ticker", "fold", "scored", "y_ret_5", "y_rv_5", *RETURN_MODELS, "ewma", "roll_21", "har_pooled",
            "har_stock", "ridge_vol", "lgbm_vol"]
    oos[keep].to_parquet(PROCESSED / "baseline_oos.parquet", index=False)

    ret = pd.concat([score_returns(forecasts[h], h) for h in (5, 1)]).round(5)
    var = pd.concat([score_variance(forecasts[h], h) for h in (5, 1)]).round(5)
    ret.to_csv(OUT / "return_scores.csv", index=False)
    var.to_csv(OUT / "variance_scores.csv", index=False)
    print("\n" + ret[ret.fold == "all"].to_string(index=False))
    print("\n" + ret[(ret.fold != "all") & (ret.horizon == 5)].pivot(index="model", columns="fold", values="ic").round(4).to_string())
    print("\n" + var[var.fold == "all"].to_string(index=False))
    print("\n" + var[var.horizon == 5].pivot(index="model", columns="fold", values="qlike").to_string())

    gaps = variance_gaps(oos).round(5)
    gaps.to_csv(OUT / "variance_gaps.csv", index=False)
    print("\n" + gaps.to_string(index=False))

    h5 = var[(var.fold == "all") & (var.horizon == 5)]
    best_vol = h5.loc[h5.qlike.idxmin(), "model"]
    dist = score_distribution(oos, best_vol).round(5)
    dist.to_csv(OUT / "distribution_scores.csv", index=False)
    print("\n" + dist.to_string(index=False))

    rf = load_riskfree().rf_period
    dates = pd.DatetimeIndex(sorted(oos.date.unique()))
    opens = px[px.ticker.isin(tickers)].pivot(index="date", columns="ticker", values="adj_open").loc[dates, tickers]
    spy_open = px[px.ticker == "SPY"].set_index("date").adj_open.loc[dates].to_frame("SPY")
    spy_ret = spy_open.SPY.pct_change().dropna()
    run = dict(max_turnover=MAX_TURNOVER, fund_first_trade=True)

    spy = simulate(pd.DataFrame(1.0, index=dates[:1], columns=["SPY"]), spy_open, rf, BASE_COST_BPS)
    cash = simulate(pd.DataFrame(0.0, index=dates, columns=tickers), opens, rf, 0)
    results = {"cash": cash, "spy_buy_hold": spy}
    targets = strategy_targets(oos, tickers, sector)
    costs = []
    for name, t in targets.items():
        results[name] = simulate(t, opens, rf, BASE_COST_BPS, **run)
        costs.append({"strategy": name, **{f"sharpe_{c}bps": annual_sharpe(
            simulate(t, opens, rf, c, **run).returns, rf) for c in COST_GRID}})

    table = pd.DataFrame({k: stats(v, rf, spy_ret) for k, v in results.items() if k != "cash"}).T
    table.loc["cash", ["ann_return", "sharpe"]] = [cash.nav.iloc[-1] ** (252 / (len(dates) - 1)) - 1, np.nan]

    base = results["equal_weight"].returns - rf.reindex(results["equal_weight"].returns.index)
    gap = {}
    for name, res in results.items():
        if name in ("cash", "equal_weight"):
            continue
        x = pd.DataFrame({"a": res.returns - rf.reindex(res.returns.index), "b": base}).dropna()
        sr = lambda d: (d.a.mean() / d.a.std() - d.b.mean() / d.b.std()) * np.sqrt(252)  # noqa: E731
        gap[name] = (sr(x), *block_bootstrap_ci(x, sr, seed=SEED))
    gap = pd.DataFrame(gap, index=["sharpe_minus_ew", "lo", "hi"]).T
    table = table.join(gap)
    table.round(4).to_csv(OUT / "strategy_scores.csv")
    print("\n" + table.round(3).to_string())

    yearly = pd.DataFrame({k: v.returns.groupby(v.returns.index.year).apply(lambda r: (1 + r).prod() - 1)
                           for k, v in results.items()}).round(4)
    yearly.to_csv(OUT / "strategy_by_year.csv")
    print("\n" + yearly.to_string())
    pd.DataFrame(costs).round(3).to_csv(OUT / "cost_sensitivity.csv", index=False)
    print("\n" + pd.DataFrame(costs).round(3).to_string(index=False))

    random_sharpes = np.array([annual_sharpe(simulate(random_targets(oos, tickers, sector, i), opens, rf,
                                                      BASE_COST_BPS, **run).returns, rf) for i in range(N_RANDOM)])
    control = table[["sharpe"]].dropna().copy()
    control["pct_of_random"] = [(random_sharpes < s).mean() for s in control.sharpe]
    control.round(3).to_csv(OUT / "random_control.csv")
    pd.Series(random_sharpes).describe(percentiles=[0.025, 0.5, 0.975]).round(3).to_csv(OUT / "random_sharpes.csv")
    print(f"\nrandom top-{TOP_K}: median {np.median(random_sharpes):.3f}, "
          f"95% range {np.quantile(random_sharpes, 0.025):.3f} to {np.quantile(random_sharpes, 0.975):.3f}")
    print(control.round(3).to_string())


if __name__ == "__main__":
    main()
