"""Gate F0 audit of the filtered FNSPID subset.

Reads data/interim/fnspid/{news_raw,prices_raw}.parquet and writes tables to reports/data_audit/:
  news_by_ticker_year.csv, timestamp_profile.csv, hour_profile.csv, news_quality.csv,
  price_coverage.csv, price_anomalies.csv, zero_news_sessions.csv,
  temporal_join_sample.csv, summary.json

The trading calendar is taken from SPY's bar dates (FNSPID covers 1993-2023-12-28).
"""

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "interim" / "fnspid"
OUT = ROOT / "reports" / "data_audit"
NY = "America/New_York"
DECISION_HOUR_NY = 17  # canonical decision time: 17:00 New York after the close

# Headlines that describe a price move after the fact; unsafe as same-day predictors.
HINDSIGHT = re.compile(
    r"\b(?:stocks? moving|biggest movers|movers|shares? (?:are )?(?:up|down|higher|lower)|"
    r"(?:rose|fell|jumped|surged|plunged|soared|tumbled|slid|rallied|climbed|dropped|gained)\b|"
    r"hit 52-week|trading (?:higher|lower)|mid-day|midday|after-hours|pre-market|premarket|"
    r"why .* (?:stock|shares) (?:is|are)? ?(?:up|down|moving|trading)|stock market (?:today|news))",
    re.IGNORECASE,
)


def trading_days(prices):
    spy = prices.loc[prices.original_symbol == "SPY", "date"]
    return pd.DatetimeIndex(sorted(pd.to_datetime(spy).unique()))


def classify_precision(ts_utc):
    """date_only if the stamp sits exactly on a whole hour with zero minutes/seconds."""
    t = ts_utc.dt
    whole_hour = (t.minute == 0) & (t.second == 0)
    return np.select([whole_hour & (t.hour == 0), whole_hour], ["midnight_utc", "whole_hour"], "sub_hour")


def eligible_session(news, days):
    """First trading session whose 17:00 NY decision may use the record.

    precise stamps: the same session if published on a trading day by 17:00 NY, else the next session.
    date-only stamps: the first session strictly after the reported calendar date (the article could
    have appeared at any time on that date, including after the decision).
    """
    days_np = days.values.astype("datetime64[ns]")
    ny = news["published_utc"].dt.tz_convert(NY)
    ny_date = ny.dt.tz_localize(None).dt.normalize()
    after_cut = ny.dt.hour >= DECISION_HOUR_NY  # 17:00:00 itself is treated as too late
    # precise: smallest session >= date (or > date if after the cut-off)
    idx_same = np.searchsorted(days_np, ny_date.values.astype("datetime64[ns]"), side="left")
    idx_next = np.searchsorted(days_np, ny_date.values.astype("datetime64[ns]"), side="right")
    precise_idx = np.where(after_cut, idx_next, idx_same)
    # date-only: use the reported UTC calendar date, strictly after. A whole-hour stamp late in the
    # UTC day (e.g. 23:00) may be local midnight of the *next* date, so roll it forward a day.
    utc = news["published_utc"].dt.tz_localize(None)
    rep_date = utc.dt.normalize() + pd.to_timedelta((utc.dt.hour >= 12).astype(int), unit="D")
    dateonly_idx = np.searchsorted(days_np, rep_date.values.astype("datetime64[ns]"), side="right")
    idx = np.where(news["precision"] == "sub_hour", precise_idx, dateonly_idx)
    out = pd.Series(pd.NaT, index=news.index, dtype="datetime64[ns]")
    ok = idx < len(days_np)
    out[ok] = days_np[idx[ok]]
    return out


def audit_news(con, days):
    news = con.sql(f"""
        SELECT *, row_number() OVER () AS news_row
        FROM read_parquet('{(IN / 'news_raw.parquet').as_posix()}')
    """).df()
    news["published_utc"] = pd.to_datetime(news["published_raw"].str.replace(" UTC", "", regex=False),
                                           errors="coerce", utc=True)
    news["year"] = news["published_utc"].dt.year
    news["precision"] = classify_precision(news["published_utc"])
    news["has_body"] = news["body"].fillna("").str.strip().str.len() > 0
    news["url_norm"] = (news["url"].fillna("").str.lower().str.replace(r"^https?://(www\.)?", "", regex=True)
                        .str.rstrip("/").str.split("?").str[0])
    news["title_norm"] = news["title"].fillna("").str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True) \
                                      .str.split().str.join(" ")
    news["hindsight_title"] = news["title"].fillna("").str.contains(HINDSIGHT)
    news["eligible_session"] = eligible_session(news, days)
    news["eligible_lag_days"] = (news["eligible_session"] - news["published_utc"].dt.tz_localize(None)
                                 .dt.normalize()).dt.days

    # Benzinga URLs embed /news/YY/MM/ - a free consistency check on the timestamp's month.
    m = news["url"].fillna("").str.extract(r"benzinga\.com/(?:[a-z-]+/)*?(\d{2})/(\d{2})/")
    has_m = m[0].notna()
    url_ym = (2000 + m.loc[has_m, 0].astype(int)) * 100 + m.loc[has_m, 1].astype(int)
    ts = news.loc[has_m, "published_utc"]
    ts_ym = ts.dt.year * 100 + ts.dt.month
    news["url_month_mismatch"] = pd.NA
    news.loc[has_m, "url_month_mismatch"] = (url_ym != ts_ym).values

    by_ty = (news.groupby(["canonical_ticker", "year", "source_file"]).size()
             .unstack("source_file", fill_value=0).reset_index())
    by_ty.to_csv(OUT / "news_by_ticker_year.csv", index=False)

    ts_prof = (news.groupby(["source_file", "precision"]).size().rename("rows").reset_index())
    ts_prof["share"] = ts_prof["rows"] / ts_prof.groupby("source_file")["rows"].transform("sum")
    ts_prof.to_csv(OUT / "timestamp_profile.csv", index=False)

    hours = (news.assign(hour_utc=news["published_utc"].dt.hour)
             .groupby(["source_file", "precision", "hour_utc"]).size().rename("rows").reset_index())
    hours.to_csv(OUT / "hour_profile.csv", index=False)

    dup_url_within = news.duplicated(["source_file", "url_norm"], keep="first") & (news["url_norm"] != "")
    news["pub_date"] = news["published_utc"].dt.date
    dup_title_ticker = news.duplicated(["canonical_ticker", "title_norm", "pub_date"], keep="first")
    url_sets = news[news.url_norm != ""].groupby("url_norm")["source_file"].nunique()
    news["url_in_both_sources"] = news["url_norm"].map(url_sets).fillna(0) > 1
    q = news.assign(dup_url_within=dup_url_within, dup_title_same_day=dup_title_ticker,
                    bad_timestamp=news["published_utc"].isna())
    quality = q.groupby(["source_file", "canonical_ticker"]).agg(
        rows=("news_row", "size"),
        first=("published_utc", "min"), last=("published_utc", "max"),
        body_share=("has_body", "mean"),
        lsa_share=("has_lsa_summary", "mean"),
        dup_url_share=("dup_url_within", "mean"),
        dup_title_same_day_share=("dup_title_same_day", "mean"),
        url_in_both_sources_share=("url_in_both_sources", "mean"),
        hindsight_title_share=("hindsight_title", "mean"),
        bad_timestamp=("bad_timestamp", "sum"),
    ).reset_index()
    quality.to_csv(OUT / "news_quality.csv", index=False)
    return news


def audit_prices(prices, days):
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    rows, anomalies = [], []
    for sym, g in prices.groupby("original_symbol"):
        g = g.sort_values("date")
        window = days[(days >= g.date.min()) & (days <= g.date.max())]
        missing = window.difference(pd.DatetimeIndex(g.date))
        r_close = np.log(g.close).diff()
        r_open = np.log(g.open).diff()
        factor = g.adj_close / g.close
        rows.append({
            "symbol": sym, "canonical_ticker": g.canonical_ticker.iloc[0],
            "first": g.date.min().date(), "last": g.date.max().date(), "rows": len(g),
            "duplicate_dates": int(g.date.duplicated().sum()),
            "missing_sessions_in_span": len(missing),
            "zero_volume_days": int((g.volume == 0).sum()),
            "nonpositive_price_rows": int((g[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
            "ohlc_inconsistent_rows": int(((g.high < g[["open", "close"]].max(axis=1) - 1e-6) |
                                           (g.low > g[["open", "close"]].min(axis=1) + 1e-6)).sum()),
            "adj_factor_min": factor.min(), "adj_factor_max": factor.max(),
            "adj_factor_decreasing_steps": int((factor.diff() < -1e-4).sum()),
            "rows_2016_2023": int(((g.date >= "2016-01-01") & (g.date <= "2023-12-31")).sum()),
            "sessions_2016_2023": int(((days >= "2016-01-01") & (days <= "2023-12-31")).sum()),
        })
        big = g.assign(r_close=r_close, r_open=r_open)
        big = big[(big.r_close.abs() > np.log(1.25)) | (big.r_open.abs() > np.log(1.25))]
        for _, b in big.iterrows():
            anomalies.append({"symbol": sym, "date": b.date.date(), "close": b.close,
                              "log_ret_close": round(b.r_close, 4), "log_ret_open": round(b.r_open, 4)})
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT / "price_coverage.csv", index=False)
    pd.DataFrame(anomalies).to_csv(OUT / "price_anomalies.csv", index=False)
    return cov


def zero_news_sessions(news, days):
    s = news.dropna(subset=["eligible_session"])
    out = []
    for (t, src), g in s.groupby(["canonical_ticker", "source_file"]):
        have = set(g.eligible_session)
        for y in range(2010, 2024):
            yd = days[days.year == y]
            if len(yd) == 0:
                continue
            n_zero = sum(d not in have for d in yd)
            out.append({"canonical_ticker": t, "source_file": src, "year": y,
                        "sessions": len(yd), "zero_news_sessions": n_zero,
                        "zero_share": round(n_zero / len(yd), 3)})
    pd.DataFrame(out).to_csv(OUT / "zero_news_sessions.csv", index=False)


def temporal_sample(news, n=120, seed=7):
    """Stratified sample for manual inspection of timestamp -> eligible-session joins."""
    s = news[news.year.between(2016, 2023)]
    keys = ["source_file", "precision"]
    per = max(1, n // max(1, s.groupby(keys).ngroups))
    sample = s.sample(frac=1, random_state=seed).groupby(keys).head(per)
    cols = ["source_file", "canonical_ticker", "original_symbol", "published_raw", "precision",
            "eligible_session", "eligible_lag_days", "hindsight_title", "title", "url", "publisher"]
    sample[cols].sort_values(["source_file", "published_raw"]).to_csv(OUT / "temporal_join_sample.csv",
                                                                     index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    prices = pd.read_parquet(IN / "prices_raw.parquet")
    days = trading_days(prices)
    news = audit_news(con, days)
    cov = audit_prices(prices, days)
    zero_news_sessions(news, days)
    temporal_sample(news)

    summary = {
        "news_rows": len(news),
        "news_rows_by_source": news.source_file.value_counts().to_dict(),
        "unparseable_timestamps": int(news.published_utc.isna().sum()),
        "precision_by_source": news.groupby("source_file")["precision"].value_counts(normalize=True)
                                   .round(4).unstack().to_dict(orient="index"),
        "body_share_by_source": news.groupby("source_file")["has_body"].mean().round(4).to_dict(),
        "hindsight_title_share": round(float(news.hindsight_title.mean()), 4),
        "url_month_mismatch_share_benzinga": round(float(news.url_month_mismatch.dropna().astype(bool).mean()), 4)
        if news.url_month_mismatch.notna().any() else None,
        "eligible_after_price_end": int(news.eligible_session.isna().sum()),
        "price_symbols": cov[["symbol", "first", "last"]].astype(str).to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
