# Universe Selection — Point-in-Time, Pre-Registered

**Date:** 2026-10-29 · **Reproduce:** `python scripts/build_universe.py` · **Config:** `configs/universe.yaml`
**Tables:** `reports/universe_selection.csv`, `reports/data_audit/universe_news_coverage.csv`, `data/external/universe/sp500_2015-12-31.csv`

## Why the original 8+4 was replaced

The original list (AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM + AMD, BAC, JNJ, WMT) was picked in 2026, so it contains names we already know became huge winners. The plan itself rates this High risk (S04, "survivor/selection bias from 2026 household names"). Two examples:

- **AMD was not in the S&P 500 on 2015-12-31.** It was removed in 2013 and re-added in 2017.
- **NVDA ranked 166th of the 2015 S&P 500 by liquidity** ($185M/day; 21st in its sector). An investor at the start of 2016 would have had no reason to single it out.

Back-testing an RL agent on that list would mix any learned skill with the selection hindsight. Choosing the universe the way an investor could have on 2015-12-31 removes most of that.

## The rule (decided before any returns were examined)

1. **Candidates:** S&P 500 members on 2015-12-31, the day before training starts. Reconstructed from a pinned Wikipedia revision (2026-05-23, revid 1355685534) by undoing the 230 changes dated after 2015-12-31. That gives 509 members, matching the index's ~505 companies plus dual share classes.
2. **Liquidity:** mean daily dollar volume over calendar 2015 (Yahoo). One share class per issuer (GOOGL, not GOOG).
3. **Selection:** the 2 most liquid names in each of the 11 GICS sectors (22), then filled to **25** with the next most liquid names in any sector. The following 5 are **reserves**, used in order only if a pick fails a data gate.
4. **Price gate:** a continuous adjusted history from 2015-01-02, with every one-day move over 25% explained by a dated market event.

No news counts and no returns were used to choose names.

## Result

| Sector | Names | 2015 $ volume ($M/day) |
|---|---|---|
| Communication Services | META, NFLX, GOOGL* | 2,399 · 1,720 · 1,353 |
| Consumer Discretionary | AMZN, BKNG | 1,847 · 808 |
| Consumer Staples | PG, WMT | 696 · 634 |
| Energy | XOM, CVX | 1,159 · 870 |
| Financials | BAC, JPM, C* | 1,395 · 1,005 · 978 |
| Health Care | GILD, PFE | 1,191 · 965 |
| Industrials | GE, BA | 1,451 · 560 |
| Information Technology | AAPL, MSFT, INTC* | 6,191 · 1,676 · 965 |
| Materials | DD, FCX | 506 · 387 |
| Real Estate | SPG, AMT | 259 · 228 |
| Utilities | DUK, EXC | 260 · 241 |

\* liquidity fill. **Reserves, in order:** T, DIS, WFC, CMCSA, JNJ.

Seven of the original twelve survive the rule: AAPL, MSFT, AMZN, GOOGL, META, JPM, XOM, plus BAC and WMT from the extension. NVDA and AMD drop out for the reasons above; JNJ is the 5th reserve.

**Sector balance.** The largest sector holds 3 of 25 names (12%), so the plan's 40% sector cap and 15% single-name cap bind rarely. Tech-like exposure (IT + Communication Services) is 6 of 25 names. In the original list it was 6 of 12.

### Price gate

All 25 names and 5 reserves have 2,973 consecutive bars from 2015-01-02 to 2026-10-28, with no gaps. The one-day moves over 25% are all real events:

| Ticker | Date | Move | Event |
|---|---|---:|---|
| META | 2022-02-03 | −26% | Q4-21 earnings, first user decline |
| NFLX | 2022-04-20 | −35% | Q1-22 subscriber loss |
| FCX | 2015-08-27 / 2020-03-24 | +29% / +30% | Commodity rally; COVID rebound |
| SPG | 2020-03-16 / 2020-11-09 | −27% / +28% | COVID crash; vaccine announcement |
| INTC | 2024-08-02 | −26% | Q2-24 earnings, dividend suspension |

DD (DowDuPont merger 2017, Dow/Corteva spin-offs 2019) and GE (GE HealthCare 2023, GE Vernova 2024 spin-offs) show no artificial jumps, so Yahoo's history is spin-off adjusted. Their news needs the aliases `DWDP → DD` and `PCLN → BKNG` (Priceline until 2018); both are in the config.

## Survivorship: what is still biased

111 of the 509 members on 2015-12-31 have no free Yahoo history. Most were acquired or delisted later, and a few listed mid-2015 (PYPL, KHC, HPE). Several of those would probably have made a sector top 2 on 2015 liquidity, notably **Allergan and Celgene** (Health Care), **Time Warner**, **Yahoo**, **EMC**, **Express Scripts**, **Monsanto** and **Walgreens**. So the universe is "point-in-time candidates, minus later deaths". That is much less biased than "2026 household names", but not bias-free. Report it as such, and use the **equal-weight universe portfolio** (not SPY alone) as the main benchmark, so the agent is judged on allocation skill rather than on which names are in the list.

Other caveats:
- Sectors use today's GICS labels (Communication Services was created in 2018). This affects grouping only, not selection by performance.
- Wikipedia's change list is crowd-maintained. Spot checks (AMD absent, TSLA absent, PYPL present from July 2015) all agree with known history.

## News coverage (Alpaca, share of weekdays with at least one article)

| Tier | Names | 2020–23 news days |
|---|---|---|
| Dense | AAPL, AMZN, META, GOOGL, MSFT, NFLX, PFE, BA, WMT, INTC, JPM | 76–99% |
| Medium | XOM, BAC, C, CVX, GILD, GE | 57–65% |
| Sparse | BKNG, FCX, PG, SPG, AMT, DUK, DD, EXC | 18–40% |

Sparse names stay in, per the plan's universe gate: no minimum news count, and missing news is explicit. The three tiers also make a natural experiment: if news features help, the gain should be largest in the dense tier and close to zero in the sparse tier. FNSPID adds 2017–2023 coverage for AMT, DUK, GILD and CVX, but has none after 2020 for SPG, FCX and PG.
