# Price-Only Baselines

**Date:** 2026-10-30 · **Reproduce:** `python scripts/build_riskfree.py && python scripts/run_baselines.py` · **Tests:** `python -m pytest`
**Tables:** `reports/baselines/*.csv` · **Data used:** 2015-01-02 to 2022-12-30. Nothing from 2023 onward was read.

## Bottom line

- **Next-week stock returns were not reliably predictable from prices.** The best model's daily rank correlation (IC) was 0.018 with t = 1.19, and no model beat a zero forecast on R².
- **Volatility was predictable.** LightGBM and pooled HAR are effectively tied, and pooled HAR clearly beats a plain 21-day rolling estimate.
- **No active strategy clearly beat holding the 25 stocks equally.** Every Sharpe gap to equal weight has an interval that includes zero.

These results set the bar in the [last section](#the-bar-for-later-models). The volatility side has a real baseline to beat, and any claim about return prediction has to clear a demanding standard.

## Setup

| Item | Choice |
|---|---|
| Universe | The 25 stocks in `configs/universe.yaml` |
| Decisions | 44,050 stock-days, 2016-01-04 to 2022-12-30 |
| Evaluation | Four expanding folds, validating on 2019, 2020, 2021 and 2022. Training labels must finish before the validation year starts |
| Labels | Adjusted open-to-open: enter at t+1, exit at t+6 (H = 5); H = 1 also run |
| Features | 28 inputs from returns, volatility, range, liquidity, beta, market context and weekday |
| Tuning | None. All models use fixed first-run settings |

**Strategies:** the model strategies and momentum hold the top 8 stocks by score and rebalance weekly. Equal weight and inverse volatility rebalance monthly. All use the plan's limits: 15% per stock, 40% per sector, long or flat, and at most 20% of NAV traded per day. The base cost is 5 bps per side.

## Return forecasts (H = 5)

| Model | R² vs zero | IC | IC t-stat | IC 2019 | IC 2020 | IC 2021 | IC 2022 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training mean | −0.0004 | — | — | — | — | — | — |
| Ridge, pooled | −0.0179 | 0.001 | 0.06 | −0.007 | 0.016 | −0.012 | 0.006 |
| Ridge, per stock | −0.0215 | 0.010 | 0.74 | 0.050 | 0.017 | −0.002 | −0.026 |
| LightGBM, pooled | −0.0415 | 0.006 | 0.35 | 0.018 | −0.022 | 0.033 | −0.005 |
| LightGBM, cross-sectional | −0.0088\* | 0.018 | 1.19 | 0.023 | −0.001 | 0.020 | 0.029 |
| 21-day momentum | — | −0.000 | −0.01 | −0.067 | 0.061 | −0.023 | 0.028 |
| **Control: LightGBM on shuffled targets** | −0.0110 | 0.013 | 1.01 | 0.028 | 0.020 | 0.004 | −0.001 |

\* Scored against the cross-sectional return, because that is what it predicts.

The t-statistics use 5-session blocks, because 5-day labels on consecutive days overlap. The control is the important row: a model trained on dates shuffled at random scored almost the same IC as the real models. So these ICs are within what noise and fixed stock characteristics can produce.

H = 1 tells the same story. The best IC is 0.015 with t = 1.73, against 0.013 for the control, and no model beats zero on R².

## Volatility forecasts (H = 5)

Lower QLIKE is better.

| Model | QLIKE, all | 2019 | 2020 | 2021 | 2022 | Gap to pooled HAR (95% interval) |
|---|---:|---:|---:|---:|---:|---:|
| Rolling 21-day | 0.590 | 0.564 | 0.805 | 0.444 | 0.547 | +0.081 (0.050 to 0.113) |
| EWMA (λ = 0.94) | 0.530 | 0.482 | 0.753 | 0.396 | 0.488 | +0.021 (−0.010 to 0.055) |
| **HAR, pooled** | **0.510** | 0.487 | 0.679 | 0.408 | 0.463 | — |
| HAR, per stock | 0.763 | 0.451 | 1.747 | 0.398 | 0.449 | +0.253 (0.058 to 0.531) |
| Ridge | 0.545 | 0.437 | 0.809 | 0.422 | 0.511 | +0.035 (−0.004 to 0.078) |
| **LightGBM** | **0.502** | 0.440 | 0.758 | 0.387 | 0.421 | −0.008 (−0.035 to 0.022) |

- **LightGBM has the lowest QLIKE overall**, but its lead over pooled HAR isn't statistically clear.
- **2020 separates them.** LightGBM beats pooled HAR in 2019, 2021 and 2022, while pooled HAR has the best 2020 of any model. Per-stock HAR breaks down in 2020, so per-stock fitting is dropped.

**Distribution check:** a plain Gaussian with mean zero and LightGBM volatility gives NLL −1.680, pinball 0.0095 and 90.1% coverage of the 90% interval. Adding a Ridge mean makes it worse (NLL −1.663).

## Strategies, 2019–2022

| Strategy | Return/yr | Vol | Sharpe (95% interval) | Max drawdown | Turnover/yr | Sharpe minus EW (95% interval) |
|---|---:|---:|---:|---:|---:|---:|
| Cash | 1.1% | — | — | — | — | — |
| SPY buy and hold | 13.1% | 20.8% | 0.64 (−0.42 to 1.74) | −32.0% | 0.3 | +0.03 (−0.26 to 0.31) |
| **Equal weight** | 13.2% | 22.3% | 0.62 (−0.48 to 1.70) | −35.0% | 0.9 | — |
| Inverse volatility | 13.4% | 20.2% | 0.67 (−0.45 to 1.80) | −33.9% | 1.5 | +0.05 (−0.08 to 0.21) |
| Momentum top 8 | 14.6% | 23.7% | 0.65 (−0.22 to 1.55) | −27.2% | 32.5 | +0.03 (−0.49 to 0.53) |
| Ridge, pooled | 12.6% | 28.6% | 0.52 (−0.54 to 1.55) | −43.5% | 47.3 | −0.10 (−0.37 to 0.15) |
| Ridge, per stock | 9.6% | 26.2% | 0.44 (−0.68 to 1.51) | −39.4% | 41.6 | −0.18 (−0.53 to 0.12) |
| LightGBM, pooled | 10.7% | 29.5% | 0.45 (−0.71 to 1.61) | −50.6% | 44.9 | −0.16 (−0.51 to 0.23) |
| LightGBM, cross-sectional | 17.6% | 28.0% | 0.68 (−0.40 to 1.70) | −40.5% | 44.1 | +0.06 (−0.30 to 0.39) |

Sharpe here is the realised value from daily net P&L, not a model estimate. Turnover counts buys plus sells as a multiple of NAV. Four years leave very wide intervals, so these rankings shouldn't be read closely.

### Return by calendar year

| Strategy | 2019 | 2020 | 2021 | 2022 |
|---|---:|---:|---:|---:|
| SPY | 31.5% | 18.2% | 29.6% | −18.7% |
| Equal weight | 26.4% | 13.5% | 33.1% | −14.0% |
| Inverse volatility | 23.9% | 13.7% | 31.0% | −10.5% |
| Momentum | 13.7% | 35.0% | 24.6% | −9.7% |
| LightGBM, cross-sectional | 38.1% | 27.3% | 38.3% | −21.4% |

### Sharpe at higher costs

| Strategy | 5 bps | 10 bps | 20 bps | 50 bps |
|---|---:|---:|---:|---:|
| Equal weight | 0.62 | 0.62 | 0.61 | 0.60 |
| Inverse volatility | 0.67 | 0.66 | 0.65 | 0.63 |
| Momentum | 0.65 | 0.58 | 0.44 | 0.03 |
| LightGBM, cross-sectional | 0.68 | 0.60 | 0.44 | −0.03 |
| Ridge, pooled | 0.52 | 0.44 | 0.27 | −0.23 |

The weekly strategies lose their small advantage at 10 bps, while equal weight barely changes.

### Random control

200 random top-8 portfolios, rebalanced weekly under the same rules and costs, gave a median Sharpe of 0.49, with a 95% range of 0.28 to 0.68. That is the fair comparison for any weekly top-8 strategy.

LightGBM cross-sectional sits at the 97th percentile. That looks good alone, but seven strategies were tried and its interval against equal weight includes zero. So it counts as a weak lead, not as evidence of skill.

## The bar for later models

Fixed now, before any later model runs. All are measured on the same four folds, net of costs where relevant.

| Area | A later model must |
|---|---|
| Return forecast | Reach IC above 0.02 with t above 2, beat the shuffled-target control, and have R² above zero |
| Volatility | Get QLIKE below 0.50 overall, and beat pooled HAR in at least 3 of 4 folds, including 2020 unless it explains why not |
| Distribution | Get NLL below −1.68, with 88–92% coverage of the 90% interval |
| Strategy | Beat equal weight with a Sharpe-gap interval above zero at 5 bps, and stay ahead at 10 bps |
| Weekly top-8 strategy | Also finish above the random control's 97.5th percentile (Sharpe 0.68) |

Equal weight, at Sharpe 0.62, stays the main benchmark. If nothing clears the strategy rows, the honest result is that the model adds no trading value, and the forecasts are reported for risk use only.

## Checks

- 50 automated tests pass, covering label timing, feature leakage, purging, costs, cash, position limits and Sharpe arithmetic.
- **Future-data guard:** the loader raises an error for any date after 2022, and the baseline run asserts this again.
- **Independent P&L reference:** on real data, a zero-cost strategy's weekly return matches the mean of its stocks' labels to 1e-9.
- **Deliberate bug:** moving execution one day earlier made three tests fail, including that reference test.
- **Benchmark sanity:** SPY's calendar-year returns are close to its published total returns. The small gaps come from measuring open to open.

## Limitations

- **Short, single sample.** Four years cover a bull run, COVID and the 2022 decline. They aren't enough to rank strategies confidently.
- **Survivor bias remains.** The 25 names are point-in-time picks minus later delistings, as described in `universe_selection.md`.
- **The turnover cap binds.** The 20% daily cap counts buys plus sells, so replacing a whole portfolio takes about 10 sessions. Weekly top-8 strategies can't fully reach each target, which may weaken a real weekly signal.
- **No tuning.** Results may improve slightly with tuning, which is why the bar requires a clear win.
- **Feature gaps.** Days since the last earnings filing isn't included yet. Fundamentals and macro data aren't used.
- **Approximate return uncertainty.** The distribution check uses the square root of forecast variance as σ. The plan warns these two aren't identical, so the main model should predict σ directly.
- **Cost model.** A flat cost per side ignores market impact, though these are very liquid stocks and the traded size is small.

## Status of gate B0

The baseline half is complete. B0 stays open until one deep model has been run on the same folds and compared with this bar.
