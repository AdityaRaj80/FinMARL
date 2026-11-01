# AEGIS 3.0: Project Status and Architecture

**Status date:** 2026-10-30

AEGIS (Adaptive Evidence-Gated Intelligence System) is an applied AI/ML portfolio project, not a live trading system. Once built, it will:
- forecast the 5-day return and volatility of 25 US stocks, giving an expected Sharpe estimate;
- use point-in-time news through stock-aware FinBERT and retrieval (RAG);
- use trained multi-agent RL (MAPPO) to decide what evidence to gather, when to refresh a forecast and how to manage risk;
- retrain itself through gated continual learning.

All trading is paper trading. There are no broker connections, leverage or short positions.

**Where we are:** the data foundation, the price-only baselines and the TFT vs PatchTST comparison are finished. Next-week returns were not reliably predictable from prices, volatility was, and no active strategy clearly beat equal weight. TFT won the comparison but didn't clear the full bar, so LightGBM and pooled HAR stay the default and TFT is the challenger for the news stage. Calibration and the ex-ante Sharpe are the remaining forecaster work.

`plan.md` remains the reference design, but it isn't binding. Section 2.9 lists where the project has deliberately moved away from it.

---

## 1. Status at a glance

| Stage | Plan gate | Status | Evidence |
|---|---|---|---|
| Data audit | F0 | **Passed, 1 item open** (120 manual date checks) | [data_audit.md](../reports/data_audit.md) |
| News source after 2023 | G0 | **Passed for a non-commercial prototype** | [source_decision.md](../reports/source_decision.md) |
| Universe | Universe gate | **Done** | [universe_selection.md](../reports/universe_selection.md) |
| Splits | — | **Done** | [splits.yaml](../configs/splits.yaml) |
| Canonical tables | — | **Done** | [data_tables.md](../reports/data_tables.md) |
| Price-only baselines | B0 | **Done** | [baselines.md](../reports/baselines.md) |
| Main forecaster | B0 | **Compared, bar not cleared**; TFT passes volatility and distribution, fails return and strategy. Calibration open | [forecasters.md](../reports/forecasters.md) |
| News value | N0 | Not started | — |
| RAG reliability | R0 | Not started | — |
| MARL value | M0 | Not started | — |
| Continual learning | A0 | Not started | — |
| Public release | D0 | Not started | — |

---

## 2. What has been done

### 2.1 FNSPID audit

Tested the local FNSPID copy before trusting any of it.

**Prices failed:**
- The series are stitched at 2020-07-06, and the older segment is unadjusted (AAPL jumps from 364 to 93).
- There is no META file, FB ends 2020-07-02 and GOOGL ends 2020-04-01.
- The price data ends 2023-12-28, and there is no adjusted open.

**News passed, with conditions:**
- 99.7% of timestamps are date-only, so a date-only article can only be used on the session after its date.
- Only 25–50% of tagged headlines name the tagged company.
- 13–41% of URLs are duplicates, depending on the ticker.
- `All_external.csv` duplicates the Nasdaq rows for these tickers, so it is dropped.
- About 5.6% of headlines describe a price move that had already happened.

### 2.2 Date check against SEC filings

Compared 1,207 own-company earnings headlines from 16 tickers (2016–2023) with 617 SEC 8-K Item 2.02 acceptance times. Under the next-session rule there were **0 genuine leaks**, so FNSPID dates can be trusted at day level.

### 2.3 Alpaca news test

Tested Alpaca's Benzinga feed with free paper-trading keys. The first test covered 169k articles for 24 tickers over 2015–2026.

| Test | Result |
|---|---|
| Timestamps | Exact to the second; only 3.5% fall on minute :00 |
| Backdating | Article ID against timestamp: Spearman 0.9998; nothing stamped earlier than its ID neighbours |
| SEC timing | 2,079 result headlines matched. 761 of 911 automated EPS headlines fall within ±15 min of the 8-K, with a median of 1.8 min before it |
| Leaks | 0 genuine leaks. The three cases over 24 h early were an analyst estimate, one wrong automated headline, and a weekend release |
| Bodies | Full text on 53–77% of articles, by ticker |
| Tag quality | Headline names the issuer on 40–66% of articles |
| Duplicates | None by article ID |
| Edits | 1–3% edited more than 1 h after publication |

**Decision:** Alpaca is the main news source for the whole period, FNSPID is the second source, yfinance supplies prices, and SEC EDGAR provides ground truth for event times.

### 2.4 Universe

The original 12 hand-picked stocks were replaced, because a list chosen in 2026 contains known winners. AMD wasn't in the S&P 500 at the end of 2015, and NVDA ranked only 166th by 2015 liquidity.

**Rule**, implemented in [build_universe.py](../scripts/build_universe.py):
1. Rebuild the S&P 500 as of 2015-12-31 from a pinned Wikipedia revision by undoing 230 later changes. That gives 509 members.
2. Rank them by 2015 average daily dollar volume, using one share class per company.
3. Take the top 2 in each of the 11 sectors, then fill to 25 by liquidity.
4. The next 5 become reserves, used in order only if a stock fails a data check.

| Sector | Stocks |
|---|---|
| Communication Services | META, NFLX, GOOGL |
| Consumer Discretionary | AMZN, BKNG |
| Consumer Staples | PG, WMT |
| Energy | XOM, CVX |
| Financials | BAC, JPM, C |
| Health Care | GILD, PFE |
| Industrials | GE, BA |
| Information Technology | AAPL, MSFT, INTC |
| Materials | DD, FCX |
| Real Estate | SPG, AMT |
| Utilities | DUK, EXC |

**Reserves, in order:** T, DIS, WFC, CMCSA, JNJ. **Benchmarks:** SPY, QQQ, IWM, TLT, GLD. **News aliases:** FB→META, PCLN→BKNG, DWDP→DD, GOOG→GOOGL.

**Remaining bias:** 111 of the 509 members have no free Yahoo history, most because they were later acquired or delisted. So the universe is much fairer than before but not bias-free. The equal-weight portfolio of these 25 stocks is therefore the main benchmark, not SPY alone. Names are never swapped because of results.

### 2.5 Splits

Defined in [splits.yaml](../configs/splits.yaml):

| Period | Dates | Use |
|---|---|---|
| Warm-up | 2015 | Lookback history only; nothing scored |
| Train | 2016-01-01 to 2021-12-31 | Fitting |
| Validation | 2022 | Model selection |
| Test (frozen) | 2023 | Run once per finished model |
| External | 2024-01-01 to 2026-10-28 | Later check on the same sources |
| Forward shadow | From 2026-10-29 | Paper only, live data |

Development uses expanding folds that validate on 2019, 2020, 2021 and finally 2022. Six sessions are removed at each boundary (H + 1), so a 5-day label can't cross into the next period.

### 2.6 Price table

[build_prices.py](../scripts/build_prices.py) writes `data/processed/prices.parquet`:
- 104,055 rows: 35 tickers × 2,973 sessions, 2015-01-02 to 2026-10-28.
- Adjusted open, high and low are derived as `price × adj_close / close`.
- It includes log returns (`ret_cc`, `ret_oo`), dollar volume, dividends, split factors and quality flags.
- 1,435 dividend and split events are stored in `corporate_actions.parquet`.
- The raw Yahoo download is frozen in `data/external/yfinance/`, because Yahoo revises adjustments.

**Checks:** every ticker has 0 missing sessions, 0 duplicate sessions, 0 bad prices, 0 inconsistent bars, 0 zero-volume days and 0 broken dividend adjustments. Every large move lines up with a known event.

**Two warnings for later work:**
- Adjusted price *levels* contain dividends paid later, so features must use returns and ratios only.
- Yahoo records spin-offs as fractional splits, so the `split` column isn't a share count.

### 2.7 News table

[build_news.py](../scripts/build_news.py) writes `data/processed/news.parquet` and `news_daily.parquet`:
- **346,364 rows**, one per article and ticker: 216,470 from Alpaca and 129,894 from FNSPID.
- **278,754 unique source articles** (162,961 Alpaca, 115,793 FNSPID). After removing FNSPID copies of Alpaca articles, 265,556 remain.
- FNSPID's last eligible session is 2023-12-18; Alpaca runs to 2026-10-28.
- `news_daily.parquet` has 74,325 rows: one per ticker and session, with article counts and a `news_missing` flag.

**Each news row carries:**
- `eligible_session` and `eligible_at_utc`: the first decision that may use it;
- `relevance`: `headline`, `lead` or `context`;
- `hindsight_title`: the headline describes a move that already happened;
- `edited_after_1h`: the stored body may contain later edits;
- `dup_of` and `is_canonical`: cross-source duplicate tracking.

**Eligibility rule:** an exact timestamp counts for session D if published before 17:00 New York time on a session day, otherwise for the next session. A date-only article counts from the first session after its date.

**Spot checks passed:** Friday evening goes to Monday, 16:59 stays on the same day, holidays roll forward, and daylight-saving weeks behave correctly.

**News coverage** (share of sessions with no Alpaca article, by ticker and year):

| Tier | Stocks | 2016–21 | 2022–26 |
|---|---|---|---|
| Dense | AAPL, AMZN, META, GOOGL, MSFT, NFLX, PFE, BA, WMT, INTC, JPM | 0–44% | 0–35% |
| Medium | XOM, BAC, C, CVX, GILD, GE | 24–67% | 12–50% |
| Sparse | BKNG, FCX, PG, SPG, AMT, DUK, DD, EXC | 61–92% | 41–86% |

### 2.8 Leak test on the final table

Every canonical earnings-result headline that names its issuer was matched to the nearest 8-K Item 2.02 within 10 days, using the eligibility stored in the table. Of the 2,948 matches, 56 became eligible before the filing. Each was reviewed by hand and the review is saved in [news_leak_review.csv](../reports/data_tables/news_leak_review.csv).

| Review | Count |
|---|---:|
| Preview or analyst estimate | 26 |
| Press release public before the 8-K | 16 |
| Another company, or not results | 11 |
| Earnings-date notice | 2 |
| Unverified | 1 |

**Result: 0 confirmed leaks.** The unverified row is a FNSPID JPMorgan headline dated eight days before the release. It has no body, so it can't be checked either way.

### 2.9 Where the project differs from `plan.md`

| Area | `plan.md` | Now | Why |
|---|---|---|---|
| Universe | 8 + 4 chosen by hand | 25 + 5 reserves, chosen by a fixed 2015 rule | Removes most selection hindsight |
| News | FNSPID, then a separate source after 2023 | Alpaca for the whole period; FNSPID second | One source avoids a break between periods |
| Prices | Licensed vendor | Frozen yfinance snapshot | Free, continuous and fine for a non-commercial prototype |
| Splits | Train to 2020, test 2022–23 | Train to 2021, val 2022, test 2023, external 2024–26 | Alpaca allows more history and a longer later check |
| Main benchmark | One of several | Equal-weight universe first | Judges allocation skill, not stock selection |
| GDELT and paid news APIs | Candidates | Not needed so far | Alpaca covers both history and live use |
| GPU | Rented 24 GB | Local RTX 5060 (8 GB) first; rent only if a measured need appears | Cuts cost |

No paid data or compute has been bought so far.

### 2.10 Price-only baselines

[run_baselines.py](../scripts/run_baselines.py) tested simple forecasters and strategies on the 2019–2022 folds. Nothing from 2023 onward was read, and the data loader now raises an error if asked for it.

| Question | Result |
|---|---|
| Can prices predict next-week returns? | Not reliably. Best IC 0.018 (t = 1.19), and a shuffled-target control scored 0.013. No model beat zero on R² |
| Can prices predict volatility? | Yes. LightGBM (QLIKE 0.502) and pooled HAR (0.510) are effectively tied |
| Is a Gaussian with that volatility calibrated? | Yes. Its 90% interval covered 90.1% of outcomes |
| Did any active strategy beat equal weight (Sharpe 0.62)? | Not clearly. The best, LightGBM cross-sectional at 0.68, has a gap interval of −0.30 to +0.39 and loses its lead at 10 bps |

The report fixes [the bar for later models](../reports/baselines.md#the-bar-for-later-models) before they run. The checks behind this work are 50 automated tests, a real-data P&L reference and a deliberate bug that the tests caught.

### 2.11 Files produced

| Location | Contents |
|---|---|
| `configs/` | `universe.yaml`, `splits.yaml` |
| `scripts/` | Audit, SEC check, Alpaca pull, universe, price, news and risk-free builders, baseline runner |
| `src/aegis/` | Data access, features, splits, backtest, metrics, baseline models |
| `tests/` | 50 automated tests |
| `reports/` | `data_audit.md`, `source_decision.md`, `universe_selection.md`, `data_tables.md`, `baselines.md`, plus check tables in `reports/data_audit/`, `reports/data_tables/` and `reports/baselines/` |
| `data/external/` | Alpaca files (480), SEC filings cache, universe inputs, frozen Yahoo snapshot |
| `data/interim/fnspid/` | FNSPID subset for the current universe |
| `data/processed/` | `prices`, `corporate_actions`, `news`, `news_daily`, `riskfree`, `baseline_oos` |
| `FNSPID_data/` | Raw FNSPID files, now moved here from the project root |
| `data.zip` | Archive of `data/`. It contains article text, so it must stay private |

---

## 3. Detailed architecture

Sections 3.1–3.3 describe rules and data that already exist. Everything after that is the **planned design**. Agent details, thresholds and hyperparameters are starting points and may change based on the gates.

### 3.1 Design rules

1. **Point-in-time everywhere.** Nothing may use information that wasn't available at the decision time.
2. **Numbers come from models, not language models.** An LLM may explain, but it never produces a forecast or a trade.
3. **Each layer must earn its place.** Each one is compared with something simpler, and a layer that loses is dropped or kept only as an explanation.
4. **Hard limits sit outside learning.** RL can't change risk limits, data rights, evidence rules or which model is promoted.
5. **Missing is not neutral.** No news stays explicitly missing and is never coded as neutral sentiment.
6. **Report honestly.** A failed gate gets reported as it is. Model-implied Sharpe is never presented as realised performance.

### 3.2 System overview

```text
+--------------------------------------------------------------------+
| SOURCES                                                            |
| Alpaca/Benzinga news | FNSPID news | Yahoo prices | SEC EDGAR      |
+--------------------------------------------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------+
| 1. DATA LAYER                                              [BUILT] |
| aliases, dedup, relevance, eligibility, trading calendar, checks   |
| prices.parquet | news.parquet | news_daily.parquet                 |
+--------------------------------------------------------------------+
            |                                        |
            v                                        v
+------------------------------+   +---------------------------------+
| 2. PRICE FEATURES    [PLAN]  |   | 3. NEWS INTELLIGENCE     [PLAN] |
| returns, volatility, volume, |   | FinBERT sentiment, stock-aware  |
| market context               |   | impact head, session features   |
+------------------------------+   +---------------------------------+
            |                                        |
            |                                        v
            |                      +---------------------------------+
            |                      | 4. POINT-IN-TIME RAG     [PLAN] |
            |                      | hybrid search, evidence ledger, |
            |                      | citations, conflict detection   |
            |                      +---------------------------------+
            |                                        |
            v                                        v
+--------------------------------------------------------------------+
| 5. FORECASTER                                               [PLAN] |
| 5-day return distribution + volatility -> expected Sharpe          |
+--------------------------------------------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------+
| 6. MARL DECISION LAYER                                      [PLAN] |
| NewsScout | ForecastManager | EvidenceAuditor | RiskAllocator      |
+--------------------------------------------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------+
| 7. RISK RULES AND PAPER PORTFOLIO                           [PLAN] |
| hard limits, next-open fills, costs, immutable decision log        |
+--------------------------------------------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------+
| 8. CONTINUAL LEARNING                                       [PLAN] |
| matured labels -> drift checks -> challenger -> gated promotion    |
+--------------------------------------------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------+
| 9. SERVING                                                  [PLAN] |
| API, dashboard, replay demo, agent traces, version history         |
+--------------------------------------------------------------------+
```

### 3.3 Decision timing

```text
Session t                                  Session t+1         Session t+6
09:30   16:00 close   17:00 decision       09:30 open          09:30 open
                      uses data known      ENTER               EXIT
                      by this time         adj_open[t+1]       adj_open[t+6]
                                           |<--- 5 intervals --------->|
```

- **Return label:** `Y_return = log(adj_open[t+6] / adj_open[t+1])`.
- **Variance proxy:** the sum of squared open-to-open log returns over the same 5 intervals.
- **Off-by-one warning:** `ret_oo` on date k is backward-looking, `log(adj_open[k] / adj_open[k-1])`. So the 5-day label equals the sum of `ret_oo` from t+2 to t+6.
- **Secondary horizon:** H = 1, from `adj_open[t+1]` to `adj_open[t+2]`.

### 3.4 Price features

Planned inputs, all computed from past data only:

| Group | Examples |
|---|---|
| Returns | 1, 5, 21 and 63-day returns; overnight gap; intraday move |
| Risk | 5, 21 and 63-day realised volatility; high-low range |
| Liquidity | Dollar volume and volume relative to its own recent history |
| Market context | Lagged SPY, QQQ, IWM, TLT and GLD returns; stock return relative to market and sector; beta estimated on past data |
| Calendar | Weekday, days since the last earnings filing |

Scalers and weights are fitted on training folds only. The first version is built and tested in `src/aegis/features.py`, except for days since the last earnings filing. The risk-free series comes from FRED's 3-month T-bill.

### 3.5 Forecasting

| Rung | Models | Question it answers |
|---|---|---|
| B0 | Cash, equal weight, SPY, 20-day momentum, inverse volatility, HAR(1,5,22) | What do simple methods achieve? |
| B1 | Ridge or ElasticNet, LightGBM | Does basic ML beat them? |
| B2 | One deep model: PatchTST or TFT | Is deep learning worth it? |
| B3 | B2 + frozen FinBERT features | Does language sentiment help? |
| B4 | B3 + stock-aware impact and verified events | Does context help? |
| B5 | B4 + MARL | Do learned agents beat fixed rules? |
| B6 | B5 + continual learning | Does adapting beat a frozen model? |

**Main model:** a shared encoder across all stocks with two output heads.
- A return head gives the mean and scale of a Student-t distribution over the 5-day return.
- A variance head predicts the realised-variance proxy.

**Starting settings:** 60-day lookback (126 as a test), width 64, up to 30 epochs, AdamW at `1e-3`, early stopping after 5 epochs without improvement. Batch size starts at 64 and drops if 8 GB of GPU memory isn't enough.

**Scoring:**
- Return: NLL or CRPS, pinball loss and calibration by stock, year and regime. Directional accuracy is only a secondary number.
- Volatility: QLIKE compared with HAR.
- Strategy: Sharpe from daily net P&L, with block-bootstrap intervals, drawdown, turnover and sensitivity to costs.

**Expected Sharpe:**

```text
SR_hat = (mu_hat - rf_5d - expected_round_trip_cost) / sigma_hat * sqrt(252 / 5)
```

Here `sigma_hat` is the standard deviation of the 5-day return. This is a model estimate used for ranking. Realised Sharpe is calculated separately from daily portfolio P&L, never from overlapping forecast windows.

### 3.6 News intelligence

```text
eligible article (Alpaca for primary features)
   -> FinBERT, cached by text hash and model version
   -> p_pos, p_neg, p_neu; sentiment = p_pos - p_neg
   -> impact head: text embedding + ticker + event type + novelty + regime
   -> expected stock-specific reaction and its uncertainty
   -> one feature row per ticker and session
```

- **Language and market reaction are separate targets.** Good results can still disappoint the market.
- **Impact labels** are returns relative to a past-estimated market beta, starting from the first tradable price after the article is available. They are observational, not causal.
- **Session features:** mean sentiment, positive and negative shares, article and source counts, novelty, event counts, contradictions, impact mean and uncertainty, and missingness.
- **Relevance:** `headline` and `lead` rows form the own-company channel. `context` rows form a separate market and peer channel.
- **Scaling:** news counts are normalised per stock on a trailing window, because coverage grows over time.
- **Source shift:** primary features use Alpaca only, because FNSPID disappears after 2023.
- **Strict variants:** test results excluding `hindsight_title` rows, and using headlines only for `edited_after_1h` rows.

**Tests for gate N0:**
- generic FinBERT against the stock-aware head, against no news;
- shuffled news across stocks on the same day;
- shuffled timestamps within a fixed window;
- random embeddings;
- the dense, medium and sparse tiers. If the news signal is real, the gain should be largest in dense stocks and near zero in sparse ones.

### 3.7 Point-in-time RAG

```text
query (ticker, event, decision time)
   -> hard filter: available time <= decision time, data rights allowed
   -> BM25 + finance embeddings
   -> reranker
   -> three views: company facts | same-event context | older analogues
   -> structured evidence: claim, source, URL, times, text span, confidence, conflicts
```

The filter runs before scoring, so a future document can't enter the results at all. The LLM never decides this. Article text is treated as untrusted data, so an instruction inside a headline can't trigger any action.

**Gate R0 requires:**
- no future document returned when one is deliberately inserted;
- valid citations;
- duplicate stories handled;
- contradictions detected;
- prompt-injection tests passed.

### 3.8 MARL decision layer

The agents are small trained policies, not chatbots talking to each other. FinBERT and the forecaster stay frozen during MARL training. The market is replayed history, so the multi-agent problem is how AEGIS's own specialists share a limited budget. It doesn't model other traders.

| Agent | Sees | Actions | Cost |
|---|---|---|---|
| `NewsScout` | New-story count, novelty, age, sentiment, contradictions, budget | Skip, retrieve local, retrieve external, request verification | Retrieval, tokens, latency |
| `ForecastManager` | Previous forecast, drift, news vector, matured errors, model age | Hold, refresh, request more evidence | Inference, latency |
| `EvidenceAuditor` | Citation scores, source disagreement, request history, budget | Accept, cross-check, reject unsupported, abstain | Verification |
| `RiskAllocator` | Validated return and risk, uncertainty, exposure, drawdown, costs | Hold, increase small, decrease small, go flat | Turnover, exposure |

**Setup:**
- Each after-close decision allows up to 3 internal steps.
- Actors see only local observations. The critic may see global state during training (centralised training, decentralised execution).
- Actions are masked if they are invalid or not permitted.
- Policies are shared across stocks at first (a proposal to test), so the design doesn't grow with each ticker.

**Reward:**

```text
net_pnl
  - lambda_drawdown * incremental_tail_risk
  - lambda_turnover * transaction_cost
  - lambda_tool     * tool_cost
  - lambda_false    * unsupported_or_false_alert_penalty
  + forecast quality, credited only after the label matures
```

**Comparison ladder, all under the same data, budget and cost rules:**
1. Fixed pipeline
2. Heuristic thresholds
3. Contextual bandit
4. Single PPO coordinator
5. IPPO
6. MAPPO

**Starting MAPPO settings:** 8 parallel replays, 64–128 steps per rollout, gamma 0.99, GAE lambda 0.95, clip 0.2, learning rates between `1e-4` and `5e-4`, 4 update epochs, MLP layers `[128, 128]`.

**Training order:**
1. Toy market with hand-checked rewards.
2. Two stocks, then 8, then all 25, starting with one seed and moving to at least 3.
3. Replay on held-out periods through the real restricted retrieval API.

**Attribution gate:** the auditor may escalate only for independently backed source disagreement, missing support for a material claim, a highly uncertain new event, or a confirmed drift alarm. Repeated requests caused by one agent's own failures are throttled and penalised. Urgent outside evidence can't be blocked.

**Exit rule for gate M0:** MAPPO must improve quality at a fixed cost, or match quality at a lower cost, against both the fixed pipeline and the heuristic or bandit, with confidence intervals. If it loses, the simpler policy ships as the default and the MAPPO result is reported honestly. If four learned agents prove too costly, the design shrinks to two.

### 3.9 Risk rules and backtest

**Hard limits, never learned:**
- long or flat only, with no leverage or shorting;
- at most 15% in one stock and 40% in one sector;
- gross exposure of at most 100%;
- daily turnover of at most 20% of NAV;
- paper NAV of $100,000, with cash earning the risk-free rate.

**Costs:** the base case is 5 bps per side, with sensitivity at 10, 20 and 50 bps.

**Fail-safe:** if data is stale, a model is missing, evidence is invalid or a risk limit is breached, the system opens no new positions and logs why.

**Signal:** a position requires a positive lower-confidence estimate of net excess return. The threshold is chosen on validation data and never on the test period.

**Decision record:** model versions, evidence IDs, actions, weights, costs, entry bar, exit and delayed outcome.

### 3.10 Continual learning

| Clock | When | What changes |
|---|---|---|
| Event | Each new article or filing | Index, features, live inference. Model weights do not change |
| Label maturity | After the 5-day outcome is known | The outcome joins the record; scoring and drift are updated |
| Training | Weekly or monthly, proposed | A challenger is trained on past data only and compared with the current model |

**Lifecycle:**
1. Every prediction records its data, features, model, calibration and policy versions.
2. Drift is monitored on residuals, calibration, sentiment mix, missingness and source delay.
3. A challenger trains on recent data plus a replay reservoir of older regimes.
4. It is tested in shadow and walk-forward against the current model and a simple periodic retrain.
5. Promotion needs a real gain and no damage to calibration, tail risk or point-in-time integrity.
6. Deployment is atomic with automatic rollback. Failed challengers stay in the record.

MARL can never promote a model. The final holdout is never used for promotion.

### 3.11 Serving

| Part | Plan |
|---|---|
| Backend | FastAPI |
| Storage | Postgres with pgvector and full-text search |
| Jobs | Worker for polling, maturity labels and drift; Redis if needed |
| Live feeds | Alpaca news and SEC filings, polled every 5–15 minutes. Described as near-real-time |
| GPU | Used only for batch work and training, never left running |
| Dashboard | Stock view, return and volatility distribution, Sharpe estimate, sources with timestamps, agent trace, fixed-vs-MAPPO replay toggle, version history, paper NAV, cost |
| API | `/health`, `/symbols`, `/news`, `/forecast`, `/signals`, `/agent-traces/{id}`, `/replay/{id}`, `/model-versions`, `/metrics` |
| Fallback | A clearly labelled replay if live data is down. Mock data is never shown as current |

### 3.12 Target code layout

```text
aegis/
  configs/      universe, splits, sources, policies, budget
  src/
    ingest/     news, SEC and price adapters
    data/       calendars, aliases, eligibility, quality checks
    features/   price features, FinBERT, event impact, session aggregation
    retrieval/  index, hybrid search, evidence ledger
    models/     baselines, forecast heads, training, calibration
    agents/     environment, observations, actions, rewards, gates
    train_rl/   PPO, IPPO, MAPPO
    adapt/      mature labels, drift, challengers, registry
    backtest/   engine, execution, risk, metrics
    api/        FastAPI app and schemas
  ui/           dashboard
  tests/        contracts, point-in-time, leakage, incentives, backtest, API
  scripts/      one-off builders and audits
  docs/         architecture, assumptions, data rights, model cards
```

Today `src/aegis/` is a flat package with data access, features, splits, backtest, metrics and baseline models. It grows into this layout as needed.

### 3.13 Testing

| Type | Examples |
|---|---|
| Point-in-time | Retrieval never returns a document later than the cutoff; a label can't enter training before it matures |
| Market calendar | Holidays, half-days, daylight saving, weekend news |
| Data | One bar per session; no duplicate stories in counts; no news stays missing |
| Backtest | No signal earns only cash; zero-fee P&L matches an independent reference; a stock split creates no wealth |
| Sharpe | Synthetic series with a known answer; no overlapping-window errors |
| Agents | Budgets enforced; reward-gaming checks; repeatable results from fixed seeds |
| Security | Prompt injection can't reach tools; no secrets in logs |

---

## 4. What remains

### Step 0: Housekeeping before baselines

| Task | Why |
|---|---|
| **Done:** local git, `.gitignore`, pinned `requirements.txt`, risk-free series | Remote set to the GitHub repo; nothing committed or pushed yet |
| Point scripts to `FNSPID_data/` | `build_fnspid_subset.py` and `validate_dates_vs_sec.py` still expect the old root paths and would fail |
| Update the default Alpaca ticker list | The default `pull` still has the old list; the 15 added tickers and aliases were passed by hand |
| Restore FNSPID file hashes in the build manifest | The rebuild skipped hashing; the old hashes are in `build_manifest_v1_old_universe.json` |
| Fix the wording in `universe_selection.md` | It says seven of the original twelve survive; the correct count is nine |
| Do the 120 manual FNSPID date checks | The last open F0 item |
| Write a data-rights matrix | Alpaca, FNSPID, Yahoo and FinBERT weights all need recorded limits |
| Move key checks into automated tests | Today's eligibility checks were run by hand or as build assertions |

### Step 1: Price-only baselines (gate B0): done

Results and the bar for later models are in [baselines.md](../reports/baselines.md).

### Step 2: Main forecaster (gate B0): compared, calibration open

1. Done: build the dual-head model and test it on 2 stocks, then all 25.
2. Done: choose PatchTST or TFT on validation data, and stop using the other. TFT won and PatchTST is dropped.
3. Open: calibrate it and compute the expected Sharpe. Coverage fell to 84.5% in 2022, so this matters.

**Done when:** it is scored against [the bar](../reports/baselines.md#the-bar-for-later-models). If it doesn't clear it, the simpler model stays the default. **Result:** TFT didn't clear the full bar, so LightGBM and pooled HAR stay the default. Details are in [forecasters.md](../reports/forecasters.md).

### Step 3: News features (gate N0)

1. Run frozen FinBERT once over deduplicated text and cache it.
2. Build session features and test B3.
3. Label a small set of event types by hand, then build the impact head for B4.
4. Run the negative controls and the news-tier comparison.

**Done when:** news helps on a pre-chosen main metric without hurting the others. If it doesn't, news stays as explanation only and no claim of signal is made.

### Step 4: RAG (gate R0)

1. Build the evidence ledger and hybrid index.
2. Add structured citations and conflict detection.
3. Run the future-document, duplicate and prompt-injection tests.

**Done when:** all reliability tests pass.

### Step 5: Risk rules and paper ledger

1. Add the hard limits, fail-safes and immutable decision log.
2. Run a full historical replay from news through forecast to paper trade.

**Done when:** the same inputs always give the same log, and limits can't be bypassed.

### Step 6: MARL (gate M0)

1. Build the replay environment and test it with scripted policies.
2. Train the bandit and single PPO first, then IPPO and MAPPO.
3. Add the attribution gate and reward-gaming tests.
4. Run held-out replay with several seeds.

**Done when:** the cost-matched comparison is complete and at least 100 replay fixtures reproduce exactly. MAPPO stays only if it wins.

### Step 7: Continual learning (gate A0)

1. Build label maturity, drift checks, a model registry and challenger training.
2. Compare a frozen model, monthly retraining and drift-triggered retraining.
3. Test rollback.

**Done when:** adaptation is shown to help, or a frozen model with scheduled retraining is kept.

### Step 8: Product and release (gate D0)

1. Build the API, the dashboard and live ingestion.
2. Build the four red-team demos: stale headline, conflicting earnings numbers, an agent overusing the auditor, and model drift.
3. Run the frozen test once for each finished model, then the external period.
4. Write the model card, scorecard, README and a 90-second video.
5. Run a rights and privacy check.

**Done when:** a working demo and a sanitised public repository exist, with honest results. The public repository contains code and aggregate metrics only.

---

## 5. Main risks and honest limits

| Risk | Status |
|---|---|
| Hindsight in the stock list | Reduced by the fixed rule, not removed; 111 members have no free data |
| Future information in the data | No confirmed leaks; one row unverified; `edited_after_1h` bodies need care |
| Prices alone show no reliable return signal | Measured in the baselines; later layers must beat a shuffled control |
| News may add nothing | Open; this is what N0 tests |
| MAPPO may not beat simple rules | Open; a loss is an accepted outcome |
| Sparse news on 8 stocks | Handled with explicit missing flags and a price-only fallback |
| Single price vendor | FNSPID's post-2020 prices match Yahoo because they appear to come from Yahoo, so no independent check exists yet |
| Yahoo revisions | Controlled by the frozen snapshot, but a later refresh may differ |
| Data rights | Fine for a private, non-commercial prototype. Alpaca, FNSPID and Yahoo text or data must not be redistributed |
| FinBERT weights | Licence must be checked before serving or releasing them publicly |
| 8 GB GPU | Not yet measured; a small amount of rented compute may be needed |
| Reused test data | Controlled by the frozen 2023 test and the separate external period |

---

## 6. Decisions still needed from you

| Decision | Default if not changed |
|---|---|
| Add a second price vendor for an independent check? | No for now; revisit before public release |
| Budget for rented GPU time? | Use the local GPU until a measured need appears |
| Public demo or a recorded private demo? | Decide at gate D0, after the rights check |
| Do the 120 FNSPID date checks yourself or have me prepare them for review? | I prepare the sheet; you confirm the final calls |
