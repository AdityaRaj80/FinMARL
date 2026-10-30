# Canonical Data Tables — Prices, News, Splits

**Date:** 2026-10-29 · **Snapshot end:** 2026-10-28 (last complete session)
**Reproduce:** `python scripts/build_prices.py && python scripts/build_news.py` (after `build_fnspid_subset.py --skip-hash` and the Alpaca pull)
**Configs:** `configs/universe.yaml`, `configs/splits.yaml` · **Check tables:** `reports/data_tables/*.csv`

All tables live under `data/processed/` and stay local. Article text is never published; only code and aggregate numbers are.

## 1. Splits (`configs/splits.yaml`)

| Period | Sessions | Use |
|---|---|---|
| Warm-up | 2015 | Lookback features only; nothing is scored |
| Train | 2016-01-01 → 2021-12-31 | Fitting; rolling-origin folds validate on 2019, 2020, 2021 |
| Validation | 2022 | Model selection, early stopping |
| Test (frozen) | 2023 | Run once per finished model; never used to choose anything |
| External | 2024-01-01 → 2026-10-28 | Out-of-time check on the same sources |
| Forward shadow | 2026-10-29 → | Paper only, live data |

The decision is at 17:00 NY and execution is at the next session's open. Labels are adjusted open-to-open returns, H = 5 (primary) and H = 1. There is a 6-session embargo (H + 1) at every boundary. All periods use the same price and news sources, so no split boundary is also a change of data vendor.

## 2. Prices (`data/processed/prices.parquet`)

- **104,055 rows:** 25 universe names, 5 reserves and 5 benchmarks × 2,973 sessions (2015-01-02 → 2026-10-28). The trading calendar is SPY's bar dates.
- **Columns:**
  - Yahoo's split-adjusted `open/high/low/close/volume`;
  - `adj_close`;
  - `adj_factor = adj_close / close`;
  - `adj_open/high/low`;
  - log returns `ret_cc` (close-to-close) and `ret_oo` (open-to-open);
  - `dollar_volume`, `dividends`, `split`;
  - `available_at_utc` (16:00 NY on the session date);
  - `quality_flags`, `vendor`, `downloaded_at_utc`.
- **Frozen snapshot.** The raw download is saved as `data/external/yfinance/raw_2026-10-28.parquet` and reused on later runs. This matters because Yahoo silently revises its adjustments.
- **Point-in-time rule.** Adjusted price *levels* include dividends paid later, so features must use returns and ratios, never adjusted levels.

**Gate results** (`price_checks.csv`): for every ticker, 0 missing sessions, 0 duplicate sessions, 0 non-positive prices, 0 OHLC inconsistencies, 0 zero-volume days, and 0 drops in `adj_factor` (a drop would mean a broken dividend adjustment). The large moves are the known events already listed in `universe_selection.md`, plus BA/CVX in March 2020, META on 2022-10-27 and NFLX on 2022-01-21.

**Spin-offs appear as fractional "splits".** Examples: DD 2019 and 2025, GE 2023 and 2024, T 2022 (WarnerMedia), EXC 2022 (Constellation), PFE 2020 (Viatris), CMCSA 2026. Returns across these dates are continuous. But the `split` column is a price-adjustment ratio, not a share count, so don't use it as one.

**Cross-check against FNSPID** (`price_vs_fnspid.csv`): for the 27 tickers with FNSPID bars over 2020-07-07 → 2023-12-28, daily returns match to a median of 0.001 bp, with 0 days differing by more than 50 bp. FNSPID's post-2020 segment was evidently itself taken from Yahoo. So the match shows that Yahoo's adjustments haven't drifted since FNSPID was built, but it is **not** an independent vendor check. An independent check needs a second vendor; see S03 in the plan.

## 3. News (`data/processed/news.parquet`)

One row per (article, canonical ticker), with `news_id = article_id:ticker`.

**Sources:**
- Alpaca/Benzinga is primary: 216,470 rows.
- FNSPID's Nasdaq file is secondary: 145,358 raw rows, of which 140,995 survive URL and title dedup and 129,894 fall in the window from 2015. `All_external.csv` is dropped because it duplicates the Nasdaq rows.
- Tickers are mapped through the aliases FB→META, PCLN→BKNG, DWDP→DD and GOOG→GOOGL.

**Eligibility:**
- **Exact timestamps** (all of Alpaca, 0.2% of FNSPID): the same session if published on a session day before 17:00 NY, otherwise the next session.
- **Date-only stamps:** the first session strictly after the reported date.
- 25 articles that would only become eligible after 2026-10-28 are dropped.
- Asserted for every row: `eligible_at_utc > published_utc`.
- Spot-checked:
  - Friday after 17:00 → Monday;
  - 16:59 → same day;
  - 4 July 2023 → 5 July;
  - DST weeks;
  - a FNSPID date-only Thursday → Friday.

**Fields added:**

| Field | Meaning |
|---|---|
| `relevance` | `headline`: issuer named in the headline. `lead`: named only in the summary or first 400 characters of the body. `context`: peer, sector or market piece |
| `n_symbols` | Number of tickers the article is tagged with (Alpaca only). Median 3; market wraps go up to 2,649 |
| `hindsight_title` | Headline describes a price move that already happened ("shares rose…") |
| `edited_after_1h` | Alpaca `updated_at` is more than 1 h after `created_at`. The stored body is the latest version, so for strict tests use only the headline on these rows |
| `dup_of`, `is_canonical` | A FNSPID row whose normalised title matches an Alpaca row for the same ticker within 3 days points to that row and is non-canonical (14,466 rows, 11% of FNSPID) |

| Source | Rows | Exact time | Body > 200 chars | Headline names issuer | Lead only | Context | Hindsight titles | Edited > 1 h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Alpaca | 216,470 | 100% | 68% | 53% | 8% | 39% | 3.4% | 2.1% |
| FNSPID | 129,894 | 0.2% | 76% | 37% | 7% | 56% | 5.1% | — |

### Leak test against SEC (`news_leak_test.csv`, `news_leak_review.csv`)

The test takes every canonical headline that names the issuer and reports results (excluding previews), then matches it to the nearest 8-K Item 2.02 within 10 days. It covers all 25 names and both sources, using the eligibility actually stored in the table.

- **2,948 headlines matched.** 2,892 become eligible after the filing.
- **56 become eligible before the filing.** Each was reviewed by hand:

| Review | Count | What it is |
|---|---:|---|
| Preview or analyst estimate | 26 | "Bernstein expects Apple to report…", "Netflix reports Q4 earnings Tuesday: what to watch" (the preview filter misses these phrasings) |
| Public before the 8-K | 16 | The press release hit the wire at 16:0x NY and the 8-K was filed 1–3 h later (AMZN, META, AAPL, GOOGL, GILD on 2019-10-24 and 2020-07-30), or the next day (FCX, BKNG). Intel's 2021-01-13 pre-announcement. The information was public at eligibility time |
| Other company, or not results | 11 | "Intel beats… deal with Apple", "Spotify… shade at Apple", Corning, Hess, "Melania documentary beats forecasts" |
| Schedule notice | 2 | "Citigroup reports Q4 earnings January 18" |
| Unverified | 1 | A FNSPID gurufocus copy titled "JPMorgan Chase Reports First-Quarter Financial Results", dated 2020-04-06, eight days before the release. It has no body; the same title reappears correctly on 2020-04-14 |

**Result: 0 confirmed leaks.** The one unverified row is a headline-only FNSPID record with no figures in it. It is left in and documented here.

### Coverage (`news_missing_share.csv`, `news_daily.parquet`)

`news_daily.parquet` has one row per ticker × session (74,325 rows) with eligible-article counts by source, the same counts for issuer-relevant articles (`n_own_*`), and `news_missing = (no Alpaca article)`. Share of sessions with no Alpaca article:

| Tier | Names | 2016–21 | 2022–26 |
|---|---|---|---|
| Dense | AAPL, AMZN, META, GOOGL, MSFT, NFLX, PFE, BA, WMT, INTC, JPM | 0–44% | 0–35% |
| Medium | XOM, BAC, C, CVX, GILD, GE | 24–67% | 12–50% |
| Sparse | BKNG, FCX, PG, SPG, AMT, DUK, DD, EXC | 61–92% | 41–86% |

These are the same tiers as in `universe_selection.md`. Ranges are per ticker-year.

### Source shift to handle in modelling

FNSPID ends in December 2023, so any feature built from it disappears in the external period. **Primary news features use Alpaca only.** FNSPID enters only as an ablation, or through the source-disagreement feature, on 2016–2023, never as a feature the external period depends on. Alpaca itself gets denser over time (MSFT has no article on 26% of sessions in 2018 and 0% from 2023). So news-intensity features should be normalised per ticker on a trailing window, not on raw counts.

## Not done yet

- Manual check of the 120 sampled FNSPID joins (`temporal_join_sample.csv`), still open from the audit. The automatic SEC test above covers the earnings subset.
- A second price vendor for an independent cross-check (plan risk S03).
- Reserve tickers' news (DIS, CMCSA) is pulled only if a reserve is ever activated.
