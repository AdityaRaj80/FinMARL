"""TFT vs PatchTST on the development folds, under the rules fixed in configs/forecasters.yaml.

  python scripts/run_forecasters.py --smoke     2 stocks, last fold, 2 epochs; prints seconds per epoch
  python scripts/run_forecasters.py --train     every model, fold, configuration and seed (resumes if stopped)
  python scripts/run_forecasters.py --score     scores finished runs, picks the winner, checks the bar

Runs are saved in data/interim/forecasters/ and checkpoints in data/models/ (both local only).
Nothing from 2023 onward is read: prices come through the guarded loader in aegis.data.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from aegis.backtest import annual_sharpe, simulate  # noqa: E402
from aegis.data import PROCESSED, load_prices, load_riskfree, splits, universe  # noqa: E402
from aegis.deep import PatchTST, TFT  # noqa: E402
from aegis.features import make_labels, price_features  # noqa: E402
from aegis.metrics import block_bootstrap_ci, daily_ic, ic_tstat, oos_r2, pinball, qlike  # noqa: E402
from aegis.models import VARIANCE_FLOOR, inverse_vol_targets, top_k_targets  # noqa: E402
from aegis.sequences import SequencePanel, fold_indices  # noqa: E402
from aegis.train import ensemble, fit_model, predict  # noqa: E402
from run_baselines import (BASE_COST_BPS, COST_GRID, MAX_NAME, MAX_SECTOR, MAX_TURNOVER, MONTHLY,  # noqa: E402
                           REBALANCE_EVERY, SEED, SKIP, TOP_K, stats)

RUNS = ROOT / "data" / "interim" / "forecasters"
MODELS = ROOT / "data" / "models"
OUT = ROOT / "reports" / "forecasters"
LOG_FEATURES = ["vol_5", "vol_21", "vol_63", "range_21", "spy_vol_21", "ewma_var"]
SMOKE_TICKERS = ["AAPL", "XOM"]


def config():
    return yaml.safe_load((ROOT / "configs" / "forecasters.yaml").read_text())


def build_panel(tickers_only=None):
    cfg = universe()
    tickers = [t for names in cfg["universe"].values() for t in names]
    sector = {t: s for s, names in cfg["universe"].items() for t in names}
    bench = cfg["benchmarks"]
    px = load_prices()
    px = px[px.ticker.isin(tickers + bench)]
    frame = price_features(px, market=tuple(bench)).merge(
        make_labels(px[px.ticker.isin(tickers)], horizons=(5, 1)), on=["date", "ticker"])
    if tickers_only:
        frame = frame[frame.ticker.isin(tickers_only)]
    features = [c for c in frame.columns if (c not in SKIP or c == "ewma_var")
                and not c.startswith(("y_", "label_end"))]
    shared = config()["shared"]
    panel = SequencePanel(frame, features, LOG_FEATURES, shared["lookback_sessions"], sector,
                          shared["horizon_sessions"])
    return panel, features


def build_model(name, params, panel, n_features):
    common = dict(n_features=n_features, n_tickers=len(panel.tickers), n_sectors=panel.n_sectors)
    if name == "tft":
        return TFT(**common, **params)
    return PatchTST(**common, lookback=panel.lookback, **params)


def run_one(name, year, c, params, seed, panel, idx, n_features, shared, device, tag=""):
    path = RUNS / f"{tag}{name}_{year}_c{c}_s{seed}.npz"
    if path.exists():
        return float(np.load(path)["best"])
    start = time.time()
    model, history = fit_model(lambda: build_model(name, params, panel, n_features), panel, idx, shared, seed, device)
    out = predict(model, panel, idx.predict_d, idx.predict_s, device)
    RUNS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    np.savez(path, d=idx.predict_d, s=idx.predict_s, mu=out.mu, sigma=out.sigma, nu=out.nu, rv=out.rv,
             history=np.array(history), best=min(history), seconds=time.time() - start)
    torch.save(model.state_dict(), MODELS / path.with_suffix(".pt").name)
    print(f"{name} {year} c{c} s{seed}: best {min(history):.4f} at epoch {int(np.argmin(history)) + 1}"
          f"/{len(history)}, {time.time() - start:.0f}s", flush=True)
    return min(history)


def train(smoke=False):
    cfg = config()
    shared = dict(cfg["shared"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel, features = build_panel(SMOKE_TICKERS if smoke else None)
    folds = splits()["folds"]
    tag = ""
    if smoke:
        shared |= {"max_epochs": 2, "patience": 2}
        folds, tag = folds[-1:], "smoke_"
    print(f"{len(features)} features, {len(panel.tickers)} stocks, device {device}", flush=True)
    for name, grid in cfg["models"].items():
        for fold in folds:
            year = pd.Timestamp(fold["val"][0]).year
            idx = fold_indices(panel, fold, shared["early_stop_sessions"])
            panel.fit_scaler(panel.dates[idx.fit_d.max()])
            losses = [run_one(name, year, c, p, shared["seeds"][0], panel, idx, len(features), shared, device, tag)
                      for c, p in enumerate(grid)]
            best = int(np.argmin(losses))
            if smoke:
                continue
            for seed in shared["seeds"][1:]:
                run_one(name, year, best, grid[best], seed, panel, idx, len(features), shared, device)


def t_nll(y, mu, sigma, nu):
    return -sps.t.logpdf(y, df=nu, loc=mu, scale=sigma)


def load_runs(panel, name, year, shared, grid):
    """All seeds of the configuration that won on the early-stopping loss for this fold."""
    first = [np.load(RUNS / f"{name}_{year}_c{c}_s{shared['seeds'][0]}.npz") for c in range(len(grid))]
    best = int(np.argmin([float(f["best"]) for f in first]))
    runs = [dict(np.load(RUNS / f"{name}_{year}_c{best}_s{s}.npz")) for s in shared["seeds"]]
    return best, runs


def forecast_frame(panel, runs, year, name):
    r = runs[0]
    rows = pd.DataFrame({"date": panel.dates[r["d"]], "ticker": np.array(panel.tickers)[r["s"]]})
    return rows.assign(**{f"{name}_{k}": v for k, v in ensemble(runs).items()}, fold=year)


def score_family(s, name):
    """Scores for one model on the scored rows s, overall and by fold."""
    rows = []
    for fold, g in [("all", s), *s.groupby("fold")]:
        mu, sigma, nu, rv = (g[f"{name}_{k}"].to_numpy() for k in ("mu", "sigma", "nu", "rv"))
        y, y_rv = g.y_ret_5.to_numpy(), g.y_rv_5.clip(lower=VARIANCE_FLOOR).to_numpy()
        ic = daily_ic(g, f"{name}_mu", "y_ret_5").dropna()
        q = sps.t.ppf(0.95, nu)
        low, high = mu - q * sigma, mu + q * sigma
        rows.append({"model": name, "fold": fold, "t_nll": t_nll(y, mu, sigma, nu).mean(), "qlike": qlike(y_rv, rv),
                     "primary": t_nll(y, mu, sigma, nu).mean() + qlike(y_rv, rv),
                     "r2_vs_zero": oos_r2(y, mu), "r2_vs_train_mean": oos_r2(y, mu, g.train_mean),
                     "ic": ic.mean(), "ic_t": ic_tstat(ic, 5),
                     "coverage_90": ((y >= low) & (y <= high)).mean(),
                     "pinball_avg": np.mean([pinball(y, low, 0.05), pinball(y, mu, 0.5), pinball(y, high, 0.95)]),
                     "nu": nu.mean()})
    return pd.DataFrame(rows)


def daily_primary(s, name):
    mu, sigma, nu, rv = (s[f"{name}_{k}"].to_numpy() for k in ("mu", "sigma", "nu", "rv"))
    ratio = s.y_rv_5.clip(lower=VARIANCE_FLOOR).to_numpy() / rv
    loss = t_nll(s.y_ret_5.to_numpy(), mu, sigma, nu) + ratio - np.log(ratio) - 1
    return pd.Series(loss, index=s.index).groupby(s.date).mean()


def daily_qlike(s, column):
    ratio = s.y_rv_5.clip(lower=VARIANCE_FLOOR) / s[column]
    return (ratio - np.log(ratio) - 1).groupby(s.date).mean()


def strategies(oos, name, sector, opens, rf):
    limits = dict(sector=sector, max_name=MAX_NAME, max_sector=MAX_SECTOR)
    top = top_k_targets(oos, f"{name}_mu", TOP_K, REBALANCE_EVERY, **limits)
    inv = inverse_vol_targets(oos.assign(vol_63=oos[f"{name}_sigma"]), MONTHLY, **limits)
    return {f"{name}_top8_mu": top, f"{name}_inverse_sigma": inv}


def score():
    cfg = config()
    shared, grids = cfg["shared"], cfg["models"]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    panel, features = build_panel()
    names = list(grids)
    base = pd.read_parquet(PROCESSED / "baseline_oos.parquet")
    frames, chosen, seed_rows = {}, [], []
    for name in names:
        parts = []
        for fold in splits()["folds"]:
            year = pd.Timestamp(fold["val"][0]).year
            best, runs = load_runs(panel, name, year, shared, grids[name])
            parts.append(forecast_frame(panel, runs, year, name))
            chosen.append({"model": name, "fold": year, "config": best, **grids[name][best],
                           "stop_loss_by_seed": [round(float(r["best"]), 4) for r in runs],
                           "epochs": [len(r["history"]) for r in runs],
                           "seconds": round(sum(float(r["seconds"]) for r in runs))})
            for seed, r in zip(shared["seeds"], runs):
                f = forecast_frame(panel, [r], year, "seed")
                seed_rows.append(f.assign(model=name, seed=seed))
        frames[name] = pd.concat(parts, ignore_index=True)
    oos = base.merge(frames[names[0]].drop(columns="fold"), on=["date", "ticker"]).merge(
        frames[names[1]].drop(columns="fold"), on=["date", "ticker"])
    assert len(oos) == len(base), "forecast rows do not line up with the baseline rows"
    oos.to_parquet(PROCESSED / "forecaster_oos.parquet", index=False)
    s = oos[oos.scored]

    scores = pd.concat([score_family(s, n) for n in names]).round(5)
    scores.to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(chosen).to_csv(OUT / "chosen_configs.csv", index=False)
    print(scores.to_string(index=False))
    print("\n" + pd.DataFrame(chosen).drop(columns=["heads"]).to_string(index=False))

    seeds = pd.concat(seed_rows, ignore_index=True).merge(base[["date", "ticker", "scored", "y_ret_5", "y_rv_5"]],
                                                          on=["date", "ticker"])
    seeds = seeds[seeds.scored]
    per_seed = []
    for (m, sd), g in seeds.groupby(["model", "seed"]):
        ic = daily_ic(g, "seed_mu", "y_ret_5").dropna()
        per_seed.append({"model": m, "seed": sd, "primary": daily_primary(g, "seed").mean(),
                         "qlike": qlike(g.y_rv_5.clip(lower=VARIANCE_FLOOR), g.seed_rv), "ic": ic.mean()})
    per_seed = pd.DataFrame(per_seed).round(4)
    per_seed.to_csv(OUT / "per_seed.csv", index=False)
    print("\n" + per_seed.to_string(index=False))

    gap = daily_primary(s, names[0]) - daily_primary(s, names[1])
    lo, hi = block_bootstrap_ci(gap, lambda x: x.mean(), seed=SEED)
    winner = (names[0] if gap.mean() < 0 else names[1]) if lo > 0 or hi < 0 else cfg["selection"]["tie_break"]
    decided = "tie-break" if lo <= 0 <= hi else "lower primary loss"
    print(f"\nprimary gap {names[0]} - {names[1]}: {gap.mean():.5f} ({lo:.5f} to {hi:.5f}) -> {winner} by {decided}")

    gaps = []
    for n in names:
        for ref in ("har_pooled", "lgbm_vol"):
            g = daily_qlike(s, f"{n}_rv") - daily_qlike(s, ref)
            gaps.append({"model": n, "reference": ref, "qlike_gap": g.mean(),
                         **dict(zip(("lo", "hi"), block_bootstrap_ci(g, lambda x: x.mean(), seed=SEED)))})
    gaps = pd.DataFrame(gaps).round(5)
    gaps.to_csv(OUT / "variance_gaps.csv", index=False)
    print("\n" + gaps.to_string(index=False))
    har = s.groupby("fold")[["y_rv_5", "har_pooled"]].apply(
        lambda g: qlike(g.y_rv_5.clip(lower=VARIANCE_FLOOR), g.har_pooled))

    px = load_prices()
    cfg_u = universe()
    tickers = [t for ns in cfg_u["universe"].values() for t in ns]
    sector = {t: k for k, ns in cfg_u["universe"].items() for t in ns}
    dates = pd.DatetimeIndex(sorted(oos.date.unique()))
    opens = px[px.ticker.isin(tickers)].pivot(index="date", columns="ticker", values="adj_open").loc[dates, tickers]
    spy_ret = px[px.ticker == "SPY"].set_index("date").adj_open.loc[dates].pct_change().dropna()
    rf = load_riskfree().rf_period
    run = dict(max_turnover=MAX_TURNOVER, fund_first_trade=True)
    ew_costs = pd.read_csv(ROOT / "reports" / "baselines" / "cost_sensitivity.csv", index_col=0).loc["equal_weight"]
    from aegis.models import equal_weight_targets
    base_r = simulate(equal_weight_targets(dates, tickers, MONTHLY), opens, rf, BASE_COST_BPS, **run).returns
    base_x = base_r - rf.reindex(base_r.index)
    table, costs, yearly = {}, [], {}
    for n in names:
        for key, t in strategies(oos, n, sector, opens, rf).items():
            res = simulate(t, opens, rf, BASE_COST_BPS, **run)
            row = stats(res, rf, spy_ret)
            x = pd.DataFrame({"a": res.returns - rf.reindex(res.returns.index), "b": base_x}).dropna()
            sr = lambda d: (d.a.mean() / d.a.std() - d.b.mean() / d.b.std()) * np.sqrt(252)  # noqa: E731
            row["sharpe_minus_ew"], (row["lo"], row["hi"]) = sr(x), block_bootstrap_ci(x, sr, seed=SEED)
            x10 = simulate(t, opens, rf, 10, **run).returns
            row["sharpe_minus_ew_10bps"] = annual_sharpe(x10, rf) - ew_costs["sharpe_10bps"]
            table[key] = row
            costs.append({"strategy": key, **{f"sharpe_{c}bps": annual_sharpe(
                simulate(t, opens, rf, c, **run).returns, rf) for c in COST_GRID}})
            yearly[key] = res.returns.groupby(res.returns.index.year).apply(lambda r: (1 + r).prod() - 1)
    table = pd.DataFrame(table).T
    table.round(4).to_csv(OUT / "strategy_scores.csv")
    pd.DataFrame(costs).round(3).to_csv(OUT / "cost_sensitivity.csv", index=False)
    pd.DataFrame(yearly).round(4).to_csv(OUT / "strategy_by_year.csv")
    print("\n" + table.round(3).to_string())
    print("\n" + pd.DataFrame(costs).round(3).to_string(index=False))
    print("\n" + pd.DataFrame(yearly).round(3).to_string())

    random_975 = float(pd.read_csv(ROOT / "reports" / "baselines" / "random_sharpes.csv", index_col=0).loc["97.5%"].iloc[0])
    controls = pd.read_csv(ROOT / "reports" / "baselines" / "return_scores.csv")
    control_ic = controls.query("horizon == 5 and fold == 'all' and model == 'lgbm_shuffled'").ic.iloc[0]
    bar = []
    for n in names:
        a = scores[(scores.model == n) & (scores.fold == "all")].iloc[0]
        f = scores[(scores.model == n) & (scores.fold != "all")].set_index("fold")
        f.index = f.index.astype(int)
        beats = f.qlike < har.reindex(f.index)
        top = table.loc[f"{n}_top8_mu"]
        inv = table.loc[f"{n}_inverse_sigma"]
        bar.append({"model": n,
                    "return_ic_gt_0.02_t_gt_2": bool(a.ic > 0.02 and a.ic_t > 2),
                    "return_ic_above_shuffled_control": bool(a.ic > control_ic),
                    "return_r2_gt_0": bool(a.r2_vs_zero > 0),
                    "qlike_lt_0.50": bool(a.qlike < 0.50),
                    "beats_har_folds": int(beats.sum()), "beats_har_2020": bool(beats.get(2020, False)),
                    "nll_lt_-1.68": bool(a.t_nll < -1.68), "coverage_88_92": bool(0.88 <= a.coverage_90 <= 0.92),
                    "top8_gap_above_0": bool(top.lo > 0), "top8_ahead_at_10bps": bool(top.sharpe_minus_ew_10bps > 0),
                    "top8_above_random_975": bool(top.sharpe > random_975),
                    "inverse_sigma_gap_above_0": bool(inv.lo > 0)})
    bar = pd.DataFrame(bar).set_index("model").T
    bar.to_csv(OUT / "bar_check.csv")
    print("\n" + bar.to_string())

    (OUT / "selection.json").write_text(json.dumps({
        "gap_tft_minus_patchtst": round(float(gap.mean()), 5), "lo": round(lo, 5), "hi": round(hi, 5),
        "winner": winner, "decided_by": decided}, indent=1))
    robustness(s, seeds, gap)
    importance(panel, features, grids, shared)


def robustness(s, seeds, gap):
    """Checks added after the first scoring run. They explain the result and change no rule or verdict."""
    columns = {"pooled HAR": "har_pooled", "LightGBM": "lgbm_vol", "TFT": "tft_rv", "PatchTST": "patchtst_rv"}
    by_fold = pd.DataFrame({k: s.groupby("fold")[["y_rv_5", c]].apply(
        lambda g, c=c: qlike(g.y_rv_5.clip(lower=VARIANCE_FLOOR), g[c])) for k, c in columns.items()}).round(4)
    by_fold.to_csv(OUT / "qlike_by_fold.csv")
    print("\n" + by_fold.to_string())

    seed_fold = seeds.groupby(["model", "seed", "fold"])[["y_rv_5", "seed_rv"]].apply(
        lambda g: qlike(g.y_rv_5.clip(lower=VARIANCE_FLOOR), g.seed_rv)).unstack().round(4)
    seed_fold.to_csv(OUT / "qlike_by_seed.csv")
    print("\n" + seed_fold.to_string())

    # The bar's NLL came from a Gaussian, so a fat-tailed model gains from its shape alone.
    nu = 6.0
    shape = []
    for k, c in columns.items():
        scale = np.sqrt(s[c].to_numpy() * (nu - 2) / nu)
        shape.append({"variance_forecast": k, "gaussian_nll": -sps.norm.logpdf(s.y_ret_5, scale=np.sqrt(s[c])).mean(),
                      "student_t6_nll": -sps.t.logpdf(s.y_ret_5, df=nu, scale=scale).mean()})
    shape = pd.DataFrame(shape).round(4)
    shape.to_csv(OUT / "nll_same_shape.csv", index=False)
    print("\n" + shape.to_string(index=False))

    spread = [{"block": block, "seed": seed, **dict(zip(("lo", "hi"), block_bootstrap_ci(
        gap, lambda x: x.mean(), block=block, seed=seed)))} for block in (10, 21, 42) for seed in range(4)]
    spread = pd.DataFrame(spread).round(5).assign(excludes_zero=lambda d: (d.lo > 0) | (d.hi < 0))
    spread.to_csv(OUT / "primary_gap_bootstrap.csv", index=False)
    print("\n" + spread.to_string(index=False))

    means = pd.DataFrame({n: s[f"{n}_mu"].describe()[["mean", "std"]] for n in ("tft", "patchtst")}).T
    means["share_positive"] = [(s[f"{n}_mu"] > 0).mean() for n in means.index]
    means.round(5).to_csv(OUT / "mean_forecast_spread.csv")
    print("\n" + means.round(5).to_string())


def importance(panel, features, grids, shared):
    """Average TFT variable weights on the 2022 fold, for the report and the demo."""
    cfg_fold = splits()["folds"][-1]
    year = pd.Timestamp(cfg_fold["val"][0]).year
    idx = fold_indices(panel, cfg_fold, shared["early_stop_sessions"])
    panel.fit_scaler(panel.dates[idx.fit_d.max()])
    best, runs = load_runs(panel, "tft", year, shared, grids["tft"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for seed in shared["seeds"]:
        state = torch.load(MODELS / f"tft_{year}_c{best}_s{seed}.pt", map_location=device)
        model = build_model("tft", grids["tft"][best], panel, len(features)).to(device).eval()
        model.load_state_dict(state)
        with torch.no_grad():
            d, s = torch.from_numpy(idx.scored_d), torch.from_numpy(idx.scored_s)
            parts = []
            for i in range(0, len(d), 2048):
                x, static, *_ = panel.batch(d[i:i + 2048], s[i:i + 2048], device)
                parts.append(model.variable_weights(x, static).mean(dim=(0, 1)).cpu().numpy())
            out.append(np.mean(parts, axis=0))
    table = pd.Series(np.mean(out, axis=0), index=features).sort_values(ascending=False).round(4)
    table.rename("mean_weight").to_csv(OUT / "tft_variable_weights.csv")
    print("\n" + table.head(10).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.smoke or args.train:
        train(smoke=args.smoke)
    if args.score:
        score()


if __name__ == "__main__":
    main()
