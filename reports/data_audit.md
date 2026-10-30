# FNSPID Data Audit — Gate F0

**Date:** 2026-10-29 · **Scope:** local FNSPID copy, filtered to the planned 12-name universe (+ FB/GOOG aliases) and 5 benchmarks
**Reproduce:** `python scripts/build_fnspid_subset.py` then `python scripts/audit_fnspid.py` (~1 min total on this machine)
**Detailed tables:** `reports/data_audit/*.csv`, `reports/data_audit/summary.json`

> **Follow-up (same day):** Alpaca/Benzinga news was then tested and adopted as the primary news source (exact timestamps, 2015–2026, covers the original universe); see source_decision.md §5. Before that, date accuracy was then validated against SEC 8-K earnings timestamps (0 genuine leaks in 1,207 checks). Source choices and alternatives are in [source_decision.md](source_decision.md).

## Verdict: F0 FAILS for the planned universe and splits; it PASSES for a re-scoped design

FNSPID is usable as a **2018–2023 ticker-tagged news source** with conservative date-only timing. It is **not usable as a price source**, and its news coverage for the planned universe is too patchy to support the planned 2016–2020 / 2021 / 2022–23 splits. Two decisions are needed before modelling (see §6).

| Area | Result | Consequence |
|---|---|---|
| Raw files | Hashes recorded; 2,166 malformed rows out of 15.5M, none mention a universe ticker | OK |
| Price data | **Stitched series**: every symbol switches source on 2020-07-06; older segment unadjusted for later splits/dividends; several symbols end in 2020 | **Fail** — replace with an independent adjusted-price source |
| News timestamps | 99.7% are date-only (`00:00:00 UTC`) | Pass with the conservative next-session rule; no intraday claims possible |
| `All_external.csv` | For our tickers, 100% of its URLs also appear in the Nasdaq file | Drop it; it adds nothing for this universe |
| News coverage | Only 5 of 12 names have continuous coverage across 2019–2023 | **Fail** for planned universe; re-scope (§5) |
| Untagged news (9.8M rows) | Reuters/Bloomberg/lenta.ru, almost all 2007–2016 | Doesn't fill the 2016–2023 gaps |

## 1. Source files

| File | Bytes | SHA-256 |
|---|---:|---|
| `Stock_news/nasdaq_exteral_data.csv` | 23,232,979,597 | `1a7a3eb8…a5c4da` |
| `Stock_news/All_external.csv` | 5,731,397,037 | `5d4c0180…35c3` |
| `Stock_price/full_history.zip` | 589,525,596 | `03da4fce…ab91` |

Full hashes are in `data/interim/fnspid/build_manifest.json`. The Nasdaq file has 15,549,305 parsed rows, 9,804,627 of them with no `Stock_symbol`. DuckDB rejected 2,166 rows (2,086 missing columns, 75 too many, 5 unquoted values), mostly broken earnings-call transcripts; none contain a universe ticker. A few rows have text in the `Date` column (column shift). They are filtered out by the symbol join.

The filtered subset is 95,737 news rows (79,977 Nasdaq + 15,760 external), stored in `data/interim/fnspid/news_raw.parquet` (89 MB), plus 144,805 price bars.

## 2. Prices — not usable as the canonical series

**Stitching.** Every series looks like a 2020-07-02 snapshot with a later download appended from 2020-07-06:

| Symbol | 2020-07-02 close | 2020-07-06 close | Cause |
|---|---:|---:|---|
| AAPL | 364.11 | 93.46 | Old segment not adjusted for the Aug-2020 4:1 split |
| AMZN | 2,890.30 | 152.85 | Old segment not adjusted for the Jun-2022 20:1 split |
| MSFT | 206.26 (adj = close) | 210.70 (adj 204.16) | Old segment has no dividend adjustment |
| NVDA | 96.12 | 98.39 | Consistent here (older segment happens to be split-adjusted) |

Using these bars would create fake −75% / −95% returns in the middle of the training window.

**Truncated series.** BAC, FB, IWM end 2020-07-02 and GOOGL ends 2020-04-01. There is no `META.csv`. Nothing exists after 2023-12-28.

**Other.** AMD/JPM/BAC/JNJ have zero-price rows (pre-1984 only, irrelevant). There is no adjusted open column.

**Decision:** source all bars from one independent adjusted provider (yfinance for the non-commercial prototype, with Stooq as a cross-check). Keep FNSPID bars only to cross-check the post-2020-07-06 segment. This also covers the 2024–2026 bridge from the same source, so no stitching is needed.

## 3. News timestamps and eligibility

| Source | Midnight UTC (date-only) | Whole hour ≠ 00 | Sub-hour precise |
|---|---:|---:|---:|
| nasdaq | 99.69% | 0.16% | 0.15% |
| external | 99.24% | — | 0.76% |

- The plan's assumption holds. Treat effectively all records as **date-only**. A record dated D becomes eligible at the 17:00 NY decision of the first trading session **strictly after D**. Whole-hour stamps at or after 12:00 UTC (e.g. `23:00 UTC`, likely local midnight of D+1) are rolled forward one day first. The rare precise stamps use the same-session-if-before-17:00 rule.
- Resulting lag: mean 1.0 calendar day, max 3 days over a weekend (stratified 120-row sample in `temporal_join_sample.csv`).
- Benzinga URLs embed `/YY/MM/`. Only 0.09% disagree with the timestamp's month, so dates look trustworthy at day level.
- **Hindsight headlines** ("shares rose", "biggest movers", "pre-market…") are 5.6% of rows overall and up to 37% for AMD's external rows. They are flagged (`hindsight_title`) for exclusion in strict experiments.
- **Not yet done:** manually checking the 120 sampled joins against true publication times. The automated rule is conservative by construction (date-only ⇒ next session), so leakage risk is low, but the plan's ">100 manual joins" item is still open.

## 4. News content quality (Nasdaq file, universe tickers)

| Ticker | Rows | First | Last | Body % | Dup-URL % | Hindsight % |
|---|---:|---|---|---:|---:|---:|
| AAPL | 9,338 | 2020-03 | 2023-12 | 95 | 24 | 3 |
| MSFT | 8,737 | **2022-04** | 2023-12 | 100 | 27 | 3 |
| NVDA | 11,862 | 2011-03 | 2023-12 | 74 | 17 | 7 |
| AMZN | 5,060 | 2020-04 | 2023-12 | 95 | 34 | 2 |
| GOOGL (+GOOG) | 11,684 | 2018-07 | 2023-12 | 75 | 17 | 4 |
| META (FB) | 433 | 2020-02 | **2020-06** | 0 | 41 | 12 |
| JPM | 2,883 | 2018-09 | **2020-06** | 0 | 2 | 2 |
| XOM | 7,346 | 2018-06 | 2023-12 | 92 | 5 | 4 |
| AMD | 9,209 | 2016-12 | 2024-01 | 97 | 13 | 7 |
| BAC | 1,811 | 2014-10 | **2020-06** | 0 | 7 | 19 |
| JNJ | 2,928 | 2011-06 | **2020-06** | 0 | 1 | 7 |
| WMT | 8,686 | 2018-12 | 2023-12 | 100 | 6 | 3 |

- Bodies exist only on nasdaq.com articles. Benzinga-style rows (and everything in `All_external.csv`) are headline-only.
- Duplicate URLs within the file are common (13–41%). Deduplication must run before any count-based features.
- `Lsa_summary` exists whenever a body exists. Its provenance is unknown, so per the plan it is excluded.

**Zero-news sessions** (share of trading days with no eligible tagged article, Nasdaq file):

| Year | AAPL | MSFT | NVDA | AMZN | GOOGL | META | JPM | XOM | AMD | WMT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 1.00 | 1.00 | 0.27 | 1.00 | 0.02 | 1.00 | 0.02 | 0.29 | 0.06 | 0.00 |
| 2020 | 0.76 | 1.00 | 0.62 | 0.89 | 0.00 | 0.71 | 0.57 | 0.05 | 0.05 | 0.01 |
| 2021 | 1.00 | 1.00 | 0.63 | 1.00 | 0.02 | 1.00 | 1.00 | 0.02 | 0.09 | 0.02 |
| 2022 | 0.42 | 0.32 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 | 0.04 | 0.00 | 0.00 |
| 2023 | 0.03 | 0.03 | 0.03 | 0.23 | 0.03 | 1.00 | 1.00 | 0.04 | 0.01 | 0.04 |

Only **GOOGL, XOM, AMD, WMT** (and NVDA with gaps in 2020–21) have usable coverage through a train/val/test span. AAPL, MSFT and AMZN only become dense in 2022–23. META, JPM, BAC and JNJ have no news after mid-2020.

## 5. The coverage problem is the universe, not FNSPID

FNSPID's coverage is uneven by ticker, not thin overall. Ranking all symbols by their *minimum* yearly tagged-article count over 2018–2023 (`symbol_year_coverage.csv`) gives many names with ≥300 articles in **every** year, across sectors:

| Sector | Names with ≥300 articles/yr, 2018–2023 |
|---|---|
| Semis / tech | INTC, AMD, QCOM, MU, CRM, ORCL, ADBE, NVDA |
| Banks / financials | WFC, GS, MS, BX, V |
| Energy | CVX, OXY, COP |
| Healthcare | MRK, ABBV, GILD, BIIB, AMGN |
| Consumer | SBUX, KO, NKE, PEP, COST |
| Telecom / media | T, CMCSA, TMUS |
| Industrials | GE, CAT, FDX, MMM |

Choosing names by **news coverage only**, before looking at any returns, is a legitimate data-quality rule. It doesn't cherry-pick performance, and it keeps the plan's no-backtest-picking principle intact as long as the rule is written down first.

## 6. Decisions needed

1. **Universe.** Recommended: replace the hand-picked 8+4 with a **mechanical rule**: "≥300 tagged Nasdaq articles in every year 2018–2023, at most 2 names per sector, ranked by that minimum". That yields roughly 10 diversified names (e.g. INTC, AMD, WFC, GS, CVX, MRK, KO, SBUX, T, GE). Optionally keep AAPL/MSFT as "late-coverage" names with explicit `news_missing` flags before 2022, because recruiters recognise them.
2. **Splits.** Dense news starts in 2018, so use **train 2018–2021, validation 2022, frozen test 2023** for news-conditioned models. Price-only baselines can train on 2010+ from the independent price source. 2024–2026 stays the price/SEC-only external period described in plan §4 Option B/C.
3. **Prices.** Pull adjusted daily OHLCV for the universe and benchmarks from yfinance (2010–today) and reconcile against FNSPID's post-2020-07 segment. Adjusted open = `open × adj_close / close`.

## 7. What was produced

- `configs/universe.yaml` — the original plan universe (to be replaced once decision 1 is made)
- `scripts/build_fnspid_subset.py` — streaming filter of raw CSV/ZIP into Parquet + manifest
- `scripts/audit_fnspid.py` — this audit's tables
- `data/interim/fnspid/{news_raw,prices_raw}.parquet`, `build_manifest.json`
- `reports/data_audit/` — `news_by_ticker_year`, `news_quality`, `timestamp_profile`, `hour_profile`, `zero_news_sessions`, `price_coverage`, `price_anomalies`, `temporal_join_sample`, `symbol_year_coverage`, `summary.json`
