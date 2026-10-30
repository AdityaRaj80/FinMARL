# Data Source Decision — Can AEGIS use FNSPID?

**Date:** 2026-10-29 · **Builds on:** [data_audit.md](data_audit.md) · **Evidence script:** `scripts/validate_dates_vs_sec.py` → `reports/data_audit/sec_date_check.csv`, `ticker_tag_precision.csv`

> **Update (2026-10-29, evening): Alpaca was tested and becomes the primary news source.** See [§5](#5-alpaca-news-test-results). Short version: 169k Benzinga articles for 24 tickers, 2015 → 2026-10-28, with exact timestamps, 0 genuine leaks against SEC, and near-daily coverage of the original mega-cap universe across the whole span. FNSPID is kept as a second source for 2018–2023 (bodies, source-disagreement feature, denser coverage of mid-cap names).

## Answer (before the Alpaca test)

**Yes for news, no for prices.**

- **FNSPID news (Nasdaq file only) is usable** as the historical news source for **2018–2023**, under four conditions: a re-scoped universe, issuer-relevance filtering, deduplication, and non-commercial use.
- **FNSPID prices are not usable.** They are replaced by one consistently adjusted provider.
- **FNSPID cannot cover 2024–2026.** That needs a second news source. The best candidate, Alpaca's Benzinga feed, may even be good enough to replace FNSPID as the primary source, but it can't be confirmed until a free Alpaca account exists.

## 1. Evidence for and against FNSPID

| Test | Result | Verdict |
|---|---|---|
| **Date accuracy vs SEC ground truth.** 1,207 own-company earnings-result headlines (16 tickers, 2016–2023) compared with 617 SEC 8-K Item 2.02 acceptance timestamps | **0 genuine leaks** under the next-session rule. 883 dated the release day, 178 the day after. Every "dated before release" case was a preview, a results-date announcement, another company, or (Chevron, 2023-07-23) a correct date where the 8-K was filed the next morning. Checked by hand. | **Pass** |
| Timestamp precision | 99.7% date-only. Works for an after-close daily decision; no intraday use | Pass (daily only) |
| Ticker-tag precision | Only **25–50%** of tagged headlines name the tagged company (MSFT 25%, AAPL 30%, QCOM 50%). The rest are peer, sector or market articles | **Condition:** relevance scoring needed |
| Coverage | Dense 2018–2023 for many names (INTC, WFC, CVX, MRK, KO, GOOGL, XOM, AMD, WMT…); patchy for the planned mega-caps | **Condition:** re-scope universe |
| Duplicates | 13–41% duplicate URLs per ticker | **Condition:** dedup first |
| `All_external.csv` | Fully duplicates the Nasdaq rows for these tickers | Drop it |
| Article bodies | 75–100% available on nasdaq.com rows | Good for FinBERT/RAG |
| Prices | Spliced at 2020-07-06 (AAPL 364 → 93); GOOGL/FB/BAC end in 2020; no META | **Fail** |
| License | HF card: CC BY-NC-4.0. GitHub: a July 2025 note says "all the right for commercial use and research use are released", but the README disclaimer still prohibits commercial use without permission | **Condition:** non-commercial portfolio only; don't redistribute raw text |
| Time span | Ends 2023-12 (a few AMD rows into Jan 2024) | Needs a bridge source |

FNSPID's main risk, wrong dates causing look-ahead, did not show up when tested against SEC ground truth. Its remaining weaknesses are noise (tags, duplicates) and scope, and the pipeline can handle both.

## 2. Prices: replace with yfinance (verified)

- `yfinance` download for 2016-01-04 → 2026-10-28 works. The series is continuous and split-adjusted through the Aug-2020 AAPL split. It provides `Open, High, Low, Close, Adj Close, Volume`, so adjusted open = `Open × AdjClose / Close`.
- One source covers 2016–2026, so there's no stitching between FNSPID and the bridge period.
- **Caveat:** Yahoo's terms allow personal, non-commercial use only. That's fine for this prototype. A commercial deployment would need a licensed vendor (EODHD from about $20/mo, Tiingo, Massive/Polygon).
- Use FNSPID's post-2020-07-06 bars only as an independent cross-check.

## 3. News for 2024–2026 (and possibly instead of FNSPID)

| Source | History | Timestamps | Ticker tags | Body | Cost / access | Status this session |
|---|---|---|---|---|---|---|
| **Alpaca News API** (Benzinga feed) | **2015 → now**, ~130+ articles/day | Precise (created/updated time) | Yes, by Benzinga | Summary, plus content for some articles | Free account with API keys; free-plan news access and rate limits **not yet confirmed** | Endpoint live (returns 401 without keys). **Needs your free account to test** |
| **SEC EDGAR** (8-K, 10-Q/K, Item 2.02 etc.) | 1994 → now | Exact acceptance time | By CIK (exact) | Full filing | Free, ≤10 req/s, User-Agent with contact | **Verified working**. Already used as ground truth above |
| **GDELT 2.0 GKG** | Feb 2015 → now, 15-min files | 15-min | Organisation names (fuzzy, e.g. "apple inc", "wells fargo company") | No (URL + page title + tone) | Free | **Verified**. A 2024-03-15 file has 2,089 docs, 36 naming target orgs. Noisy (stock-holding spam sites); needs domain allow-list |
| HF `Brianferrell787/financial-news-multisource` | Mixed. `sp500_daily_headlines` to 2024, `yahoo_finance_articles` 2025, NYT 2000–2025 | Mixed day/minute | Some subsets | Some | Free, gated, non-commercial | Card reviewed only. Uneven, but useful as a cross-check |
| Tiingo News | Individual plans: **3 months** history; up to 15 years via sales | Precise | Yes | Some | $30/mo, "internal use" | Rejected: not enough history without a sales contract |
| Finnhub / Marketaux / Alpha Vantage / Massive news | Unclear or short documented history | Precise | Yes | Mostly no | Free tiers to ~$50/mo | Good for **live** ingestion; not verified for 2024–2025 history |

### Recommendation

1. **Create a free Alpaca account** (only you can do this) and set `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` as environment variables. I'll then run the same coverage and SEC-date tests on it. If Alpaca returns dense, precisely timestamped, ticker-tagged news for 2018–2026 on the free plan, it should become the **primary news source**:
   - One source covers the whole span, so there's no 2023/2024 source-shift break.
   - Minute timestamps allow sharper eligibility than FNSPID's next-day rule.
   - The same API doubles as the live feed.

   FNSPID would then add article bodies and serve as an independent second source, which also feeds the plan's source-disagreement feature.
2. **If Alpaca's free plan doesn't include historical news:** keep the FNSPID + bridge design.
   - **2018–2023:** FNSPID Nasdaq file.
   - **2024–2026:** SEC EDGAR (authoritative events) plus GDELT GKG (broad coverage, filtered to a finance-domain allow-list), with an explicit `source_regime` feature.
   - **Live:** Finnhub or Alpaca.
3. **In both cases:** prices come from yfinance; SEC EDGAR provides the earnings/event calendar and serves as the leakage ground truth; the universe is chosen by the coverage rule *of whichever news source wins*.

## 4. Conditions if FNSPID is used

- Use `nasdaq_exteral_data.csv` only; drop `All_external.csv` for these tickers.
- Add an `issuer_relevance` score (issuer named in title or body, or the only tagged ticker). Train primary features on relevant rows and keep peer/market rows as a separate "context" channel.
- Deduplicate by normalized URL and title before computing counts or intensities.
- Use the next-session eligibility rule (validated). Flag hindsight headlines (5.6%).
- Non-commercial, private use. Publish only code and aggregate metrics, never raw article text.

## 5. Alpaca news test results

**Reproduce:** `python scripts/check_alpaca_news.py probe | pull | report` (keys in `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`). Raw data: `data/external/alpaca/news_<TICKER>_<YEAR>.parquet` (local only). Tables: `reports/data_audit/alpaca_ticker_year.csv`, `alpaca_quality.csv`, `alpaca_sec_timing.csv`, `alpaca_zero_news_sessions.csv`.

**Access.** The free plan with paper-trading keys returns historical news from 2015-01-01 to the previous day. Pulling 24 tickers × 12 years (with article bodies) took about 3 hours with 3 parallel workers, using well under the 200 requests/minute limit. Meta's pre-2022 news is tagged `FB`, so `FB` is merged into `META`.

### Verdict by test

| Test | Alpaca result | FNSPID result (for comparison) |
|---|---|---|
| Span | 2015-01 → 2026-10, one source | 2009 → 2023-12 for our names, patchy |
| Timestamp precision | Exact to the second (`created_at`); only 3.5% fall on minute :00 | 99.7% date-only |
| Leakage vs SEC 8-K Item 2.02 | 2,079 result headlines matched. Automated "Q_ EPS $X Beats $Y" headlines: 761 of 911 within ±15 min of the 8-K, median 1.8 min *before* it (the press release hits the wire just before the filing). **0 genuine leaks.** The 3 cases more than 24 h early: an analyst estimate, a wrong automated GS headline (published 4 days early with a wrong number; its ID confirms the timestamp), and AT&T's Saturday Q3-2016 release (8-K filed Monday) | 0 genuine leaks at day level |
| Backdating check | Article ID vs `created_at`: Spearman 0.9998; no article is stamped earlier than its ID neighbours; 0.1% are stamped later (conservative) | n/a |
| Coverage, original 12 names | Every name has ≥145 articles in every year 2015–2026. AAPL/AMZN/META/GOOGL have news on 85–100% of sessions in every year; MSFT 69–100%; NVDA from 2021 | META/JPM/BAC/JNJ end mid-2020; AAPL/MSFT/AMZN empty 2019–2021 |
| Article bodies (>200 chars) | 53–77% by ticker | 75–100% on nasdaq.com rows |
| Headline names the tagged issuer | 40–66% | 25–50% |
| Duplicates | None (unique article IDs) | 13–41% duplicate URLs |
| Edited more than 1 h after publication | 1–3% (`updated_at`) | n/a |

### Zero-news sessions (share of weekdays with no article, 17:00 NY cutoff)

| Ticker | 2015 | 2017 | 2019 | 2021 | 2023 | 2025 |
|---|---:|---:|---:|---:|---:|---:|
| AAPL | 0.01 | 0.02 | 0.03 | 0.02 | 0.00 | 0.00 |
| MSFT | 0.13 | 0.22 | 0.27 | 0.03 | 0.00 | 0.00 |
| AMZN | 0.07 | 0.04 | 0.02 | 0.02 | 0.01 | 0.00 |
| GOOGL | 0.11 | 0.09 | 0.15 | 0.03 | 0.00 | 0.00 |
| META | 0.08 | 0.10 | 0.09 | 0.03 | 0.01 | 0.00 |
| NVDA | 0.68 | 0.30 | 0.46 | 0.15 | 0.01 | 0.00 |
| JPM | 0.44 | 0.37 | 0.43 | 0.30 | 0.14 | 0.12 |
| XOM | 0.38 | 0.50 | 0.56 | 0.46 | 0.21 | 0.34 |

Mid-cap and non-tech names are thinner (KO, WFC, CVX, MS, SBUX: news on 30–70% of sessions). For those, FNSPID 2018–2023 is often denser and is worth merging. Weekdays include ~9 exchange holidays a year, so these shares are slightly overstated.

### Caveats

- **Tags are still loose.** About half the tagged headlines don't name the company (e.g. an Amazon options article tagged INTC). The `issuer_relevance` split from §4 applies to Alpaca too.
- **Occasional wrong automated headlines.** Found one with a wrong revenue figure. Rare, but numbers parsed from headlines must not be trusted as fundamentals.
- **`source` is empty for 100k older rows.** By URL and format they are Benzinga content.
- **Terms.** Alpaca's news comes from Benzinga under Alpaca's market-data terms. Treat it like FNSPID: local use for a non-commercial prototype, never redistribute article text, publish only code and aggregate metrics. Check Alpaca's data agreement before any commercial use.

### What changes

1. **Primary news source: Alpaca (Benzinga), 2015 → today, with exact timestamps.** The eligibility rule becomes precise: an article counts for session D if `created_at` < 17:00 NY on D. No next-day roll is needed.
2. **Superseded the same day by a point-in-time 25-name universe, see [universe_selection.md](universe_selection.md).** (Original note: the original universe is viable again.) The coverage problem was FNSPID's, not the universe's. AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM and XOM all have usable coverage; `configs/universe.yaml` can stay as it is, with `FB → META` already aliased. NVDA before 2021 and JPM/XOM will have many zero-news days, so the `news_missing` flag matters.
3. **Splits can be longer, with no source break:** train 2016–2021, validation 2022, frozen test 2023, external test 2024–2026 from the same source. The 2024–2026 period no longer needs GDELT or a `source_regime` feature.
4. **FNSPID's role:** a second source for 2018–2023. It contributes article bodies (more complete than Alpaca's), denser coverage for mid-cap names, and an independent source for the disagreement feature.
5. **Live feed:** the same Alpaca API (REST, or its news WebSocket).
