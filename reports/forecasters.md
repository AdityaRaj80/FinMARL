# TFT vs PatchTST

Comparison of the two deep forecasters on the 2019–2022 development folds, under rules fixed in [forecasters.yaml](../configs/forecasters.yaml) before either model ran. Nothing from 2023 onward was read. The code is in [run_forecasters.py](../scripts/run_forecasters.py), and every table comes from [reports/forecasters/](forecasters/).

## Bottom line

- **TFT wins the comparison.** Its primary loss is lower in all four folds. The result would be the same if the gap counted as a tie, because the fixed tie-break is also TFT.
- **TFT doesn't clear the full bar.** It passes the volatility and distribution rows and fails the return and strategy rows. PatchTST passes only the distribution row.
- **LightGBM and pooled HAR stay the default**, as the fallback rule requires. TFT moves forward as the challenger for the news stage, and PatchTST is dropped.
- **TFT adds no proven trading value.** Under the bar's wording, its forecasts are for risk use only.
- **The 2023 test year stays unread.**

## Setup

| Item | Choice |
|---|---|
| Data | 25 stocks, 29 price features, 60-session lookback |
| Target | 5-day return distribution and 5-day realised variance |
| Folds | 2019, 2020, 2021, 2022, expanding from 2016 |
| Shared design | Same head, loss, optimiser, early stopping and training budget |
| Configurations | Three per model, chosen on each fold by early-stopping loss with seed 0 |
| Scored forecast | Three-seed ensemble of the chosen configuration |
| Runs | 40 in total. TFT took 128 minutes and PatchTST 23 minutes on an RTX 5060 laptop GPU |

## Selection result

| Measure | Result |
|---|---|
| Primary loss | TFT −1.250, PatchTST −1.217 |
| Daily gap, TFT minus PatchTST | −0.033 (95% interval −0.078 to −0.0004) |
| Gap by fold | 2019 −0.015, 2020 −0.044, 2021 −0.032, 2022 −0.040 |
| Winner | TFT, by lower primary loss |

The interval only just excludes zero. With other bootstrap seeds and block lengths, it excluded zero in 4 of 12 settings ([primary_gap_bootstrap.csv](forecasters/primary_gap_bootstrap.csv)). This should be read as a narrow, consistent lead rather than a clear win. It doesn't affect the choice, since a tie also goes to TFT.

## Scores

Scored on all four folds. Lower is better for QLIKE and NLL.

| Measure | TFT | PatchTST | LightGBM | Pooled HAR |
|---|---|---|---|---|
| QLIKE | 0.487 | 0.512 | 0.502 | 0.510 |
| NLL, model's own forecast | −1.737 | −1.729 | — | — |
| NLL, Student-t with ν = 6, mean zero | −1.737 | −1.731 | −1.739 | −1.737 |
| NLL, Gaussian, mean zero | −1.688 | −1.675 | −1.680 | −1.679 |
| 90% interval coverage | 88.5% | 88.3% | 90.1% | — |
| Return IC (t) | 0.020 (1.50) | 0.010 (0.60) | 0.018 (1.19) | — |
| Return R² vs zero | 0.0001 | −0.0040 | −0.0088 | — |

The LightGBM return row is the cross-sectional model, and the variance columns use the variance LightGBM. The shuffled-target control has IC 0.013 (t = 1.01).

### By fold

| Fold | QLIKE: TFT | PatchTST | LightGBM | Pooled HAR | IC: TFT | PatchTST | Coverage: TFT | PatchTST |
|---|---|---|---|---|---|---|---|---|
| 2019 | 0.455 | 0.473 | 0.440 | 0.487 | 0.017 | 0.066 | 90.2% | 90.6% |
| 2020 | 0.676 | 0.705 | 0.758 | 0.679 | 0.034 | 0.055 | 88.3% | 86.4% |
| 2021 | 0.391 | 0.413 | 0.387 | 0.408 | 0.032 | 0.005 | 90.8% | 92.0% |
| 2022 | 0.426 | 0.456 | 0.421 | 0.463 | −0.003 | −0.085 | 84.5% | 84.1% |

## Bar check

| Row | Requirement | TFT | PatchTST |
|---|---|---|---|
| Return | IC above 0.02 with t above 2 | **Fail**: 0.020, t = 1.50 | **Fail**: 0.010, t = 0.60 |
| | Beat the shuffled control | Pass | Fail |
| | R² above zero | Pass: 0.0001 | Fail |
| Volatility | QLIKE below 0.50 | Pass: 0.487 | **Fail**: 0.512 |
| | Beat pooled HAR in 3 of 4 folds, including 2020 | Pass: 4 of 4 | **Fail**: 2 of 4, not 2020 |
| Distribution | NLL below −1.68 | Pass: −1.737 | Pass: −1.729 |
| | 88–92% coverage | Pass: 88.5% | Pass: 88.3% |
| Strategy | Sharpe-gap interval above zero at 5 bps | **Fail** | **Fail** |
| | Stay ahead of equal weight at 10 bps | **Fail** for top-8, pass for inverse sigma | **Fail** for top-8, pass for inverse sigma |
| Weekly top-8 | Above random 97.5th percentile (0.679) | Pass: 0.683 | **Fail**: 0.360 |

| Row | TFT | PatchTST |
|---|---|---|
| Return | Fail | Fail |
| Volatility | Pass | Fail |
| Distribution | Pass | Pass |
| Strategy | Fail | Fail |

## How strong the passes are

**Volatility.** This is TFT's real result, but it isn't a clean win.

- Its gap to pooled HAR is −0.022 (interval −0.055 to 0.008), and to LightGBM −0.014 (−0.059 to 0.017). Both intervals include zero.
- LightGBM is better in 2019, 2021 and 2022. TFT's overall lead comes from 2020, where LightGBM struggled (0.676 vs 0.758).
- TFT beat pooled HAR in 2020 by only 0.003.
- The pass depends on the ensemble. Single seeds scored 0.540, 0.501 and 0.506 overall, all above the 0.50 limit, and ranged from 0.65 to 0.82 in 2020.

**Distribution.** The NLL pass says little about skill.

- The bar was set by a Gaussian baseline, so fat tails alone improve the score. Give every variance forecast the same Student-t shape, and LightGBM (−1.739) and pooled HAR (−1.737) match TFT (−1.737).
- Coverage passes overall, but it falls to 84.5% in 2022, so the intervals were too narrow in that year.
- ν stayed between 5.9 and 6.1, close to its starting value of 6. Short training gave it little chance to move.

**Return.** There is no reliable return signal.

- TFT's mean forecast is close to a constant: 0.37% a week on average, spread by only 0.10% across rows, and positive on 99.9% of rows.
- R² above zero therefore reflects that steady upward drift and isn't evidence of stock selection.
- ICs are unstable. PatchTST went from 0.066 in 2019 to −0.085 in 2022, and TFT's single-seed ICs ranged from 0.004 to 0.021.

**Strategy.** Neither model beat equal weight.

| Strategy | Sharpe, 5 bps | Gap to equal weight (interval) | Sharpe, 10 bps | Sharpe, 50 bps |
|---|---|---|---|---|
| TFT top-8 weekly | 0.68 | +0.07 (−0.27 to 0.42) | 0.60 | −0.02 |
| TFT inverse sigma monthly | 0.68 | +0.06 (−0.04 to 0.19) | 0.67 | 0.64 |
| PatchTST top-8 weekly | 0.36 | −0.26 (−0.64 to 0.12) | 0.29 | −0.29 |
| PatchTST inverse sigma monthly | 0.66 | +0.04 (−0.08 to 0.20) | 0.65 | 0.61 |
| Equal weight | 0.62 | — | 0.62 | 0.60 |
| Inverse volatility baseline | 0.67 | — | 0.66 | 0.63 |
| LightGBM cross-sectional | 0.68 | — | 0.60 | −0.03 |

TFT's top-8 clears the random control by 0.004 and loses its lead over equal weight at 10 bps, just as LightGBM did. TFT inverse sigma is close to the plain inverse-volatility baseline.

## Training behaviour

- **Early stopping came quickly.** 31 of 40 runs reached their best early-stopping loss by epoch 2, and 17 of them at epoch 1. Later epochs fit the training data without helping on the following year.
- **The forecasts still moved away from their starting point.** TFT's variance forecast differs from EWMA by about 30% on a typical row, and it improved QLIKE from 0.530 to 0.487.
- **Variable weights are spread out.** On the 2022 fold, market-wide inputs lead: SPY 21-day volatility 7.6%, TLT 5-day return 6.9%, and the stock's own 21-day volatility 6.8%. An even split would be 3.4%. The weights show what TFT used, not what caused returns.

## What this means for later stages

- News has to add something real. TFT with news must beat this price-only TFT, and LightGBM and pooled HAR, on the same folds.
- If a volatility forecast is needed now, LightGBM or pooled HAR is the simpler and equally defensible choice.
- TFT's single-seed variance is large, so any later TFT result must use the ensemble and report the seed spread.
- The 2022 under-coverage makes calibration worth doing before any ex-ante Sharpe is computed. That step is still open.

## Limitations

- **One short sample.** Four years cover one crash and one bear market, so a small gap in one year can move the overall ranking.
- **Light tuning by design.** Three configurations each. More search might help, but it would also spend the validation years.
- **Rule wording.** "Clears the bar" is read here as every row. Passing some rows doesn't make TFT the default.
- **Same-shape NLL is a later check.** It was added after the first scoring run to explain the result, and the bar itself is unchanged.
- **Survivor bias and the turnover cap** apply here as they do in [baselines.md](baselines.md).

## Checks

- **Independent recomputation:** a separate script reproduced NLL, QLIKE, IC and R² for both models to five decimals.
- **Fold boundaries:** on every fold, the last training label ends before the early-stopping rows start, and the last early-stopping label ends before the fold's scoring year. The scaler uses only sessions up to the last training date.
- **Date range:** forecasts end on 2022-12-30, and the last scored row's label ends on the same day.
- **Completeness:** 25,200 forecast rows, matching the baseline rows, with no duplicates or missing values.
- **Changes after the first scoring run:** the 10 bps check compared against equal weight's 5 bps Sharpe and now uses its 10 bps Sharpe (0.615), and the shuffled-control row was missing and was added. Neither changed a verdict. The rules in the configuration and the bar were not edited.

## Status of gate B0

B0's comparison is complete: the deep model was run on the same folds and scored against the bar. The result is that it doesn't clear the full bar, so the simpler models stay the default. Calibration and the ex-ante Sharpe remain open in Step 2.
