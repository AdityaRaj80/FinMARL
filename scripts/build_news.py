"""Point-in-time news table for the universe: Alpaca (Benzinga) primary + FNSPID Nasdaq second source.

  python scripts/build_news.py        (run build_prices.py first: the trading calendar comes from it)

Outputs (local only, never redistributed):
  data/processed/news.parquet        one row per (article, canonical ticker); news_id = article_id:ticker
  data/processed/news_daily.parquet  ticker x session counts of eligible articles, with news_missing
  reports/data_tables/news_*.csv     coverage, relevance and leak-test tables

Eligibility (decision at 17:00 NY, see configs/splits.yaml):
  exact stamps (Alpaca, rare FNSPID)  same session if published on a session day before 17:00 NY, else next session
  date-only stamps (99.7% of FNSPID)  first session strictly after the reported date (audit_fnspid.eligible_session)
Rows eligible after data_end are dropped.

relevance: "headline" = issuer named in the headline; "lead" = named only in the summary / first 400 characters
of the body; "context" = neither (peer, sector or market pieces tagged with the ticker).
Cross-source duplicates (same ticker, same normalised title, within 3 days) keep the Alpaca row;
the FNSPID copy gets dup_of = that row's news_id and is_canonical = False.
"""

import hashlib
import html
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_fnspid import HINDSIGHT, classify_precision, eligible_session  # noqa: E402
from check_alpaca_news import ISSUER  # noqa: E402
from validate_dates_vs_sec import CACHE, PREVIEW, RESULT, earnings_filings, get_json  # noqa: E402

ALPACA = ROOT / "data" / "external" / "alpaca"
FNSPID = ROOT / "data" / "interim" / "fnspid" / "news_raw.parquet"
OUT = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "data_tables"
NY = "America/New_York"
START = "2015-01-01"
LEAD_CHARS = 400
DUP_WINDOW_DAYS = 3
TAG = re.compile(r"<[^>]+>")


def load_config():
    uni = yaml.safe_load((ROOT / "configs" / "universe.yaml").read_text())
    spl = yaml.safe_load((ROOT / "configs" / "splits.yaml").read_text())
    tickers = [t for names in uni["universe"].values() for t in names]
    aliases = {a: c for a, c in uni["aliases"].items() if c in tickers}
    return tickers, aliases, str(spl["data_end"])


def plain(s):
    return re.sub(r"\s+", " ", html.unescape(TAG.sub(" ", s))).strip() if isinstance(s, str) and s else ""


def norm_title(s):
    return s.fillna("").str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.split().str.join(" ")


def norm_url(s):
    return (s.fillna("").str.lower().str.replace(r"^https?://(www\.)?", "", regex=True)
            .str.split("?").str[0].str.rstrip("/"))


def load_alpaca(tickers, aliases):
    want = set(tickers) | set(aliases)
    files = [p for p in sorted(ALPACA.glob("news_*.parquet")) if p.stem.split("_")[1] in want]
    df = pd.concat([f for f in (pd.read_parquet(p) for p in files) if not f.empty], ignore_index=True)
    df = df.rename(columns={"ticker": "raw_symbol", "headline": "headline", "source": "publisher"})
    df["ticker"] = df.raw_symbol.replace(aliases)
    df = df.drop_duplicates(["ticker", "id"])
    canon = df.symbols.str.split(",").map(lambda s: sorted({aliases.get(x, x) for x in s}))
    out = pd.DataFrame({
        "article_id": "alp:" + df.id.astype(str), "source": "alpaca", "ticker": df.ticker, "raw_symbol": df.raw_symbol,
        "published_utc": pd.to_datetime(df.created_at, utc=True), "updated_utc": pd.to_datetime(df.updated_at, utc=True),
        "precision": "sub_hour", "headline": df.headline.fillna(""), "summary": df.summary.map(plain),
        "body": df.content.map(plain), "url": df.url.fillna(""),
        "publisher": df.publisher.replace("", np.nan).fillna("benzinga"), "author": df.author.fillna(""),
        "symbols": canon.str.join(","), "n_symbols": canon.str.len(),
    })
    return out


def load_fnspid(tickers):
    df = pd.read_parquet(FNSPID)
    df = df[(df.source_file == "nasdaq") & df.canonical_ticker.isin(tickers)].copy()  # All_external duplicates it
    df["published_utc"] = pd.to_datetime(df.published_raw.str.replace(" UTC", "", regex=False), errors="coerce", utc=True)
    df = df.dropna(subset=["published_utc"])
    df["url_norm"] = norm_url(df.url)
    df["title_norm"] = norm_title(df.title)
    df["day"] = df.published_utc.dt.floor("D")
    n0 = len(df)
    df = df.drop_duplicates(["canonical_ticker", "url_norm"]).drop_duplicates(["canonical_ticker", "title_norm", "day"])
    print(f"FNSPID: {n0:,} rows -> {len(df):,} after URL/title dedup")
    # FNSPID has one raw row per (article, symbol); only tags inside our universe are visible here.
    tags = df.groupby("url_norm").canonical_ticker.agg(lambda s: ",".join(sorted(set(s))))
    key = (df.url_norm.where(df.url_norm != "", df.title_norm + df.day.astype(str)))
    out = pd.DataFrame({
        "article_id": "fns:" + key.map(lambda k: hashlib.sha1(k.encode()).hexdigest()[:16]),
        "source": "fnspid", "ticker": df.canonical_ticker, "raw_symbol": df.original_symbol,
        "published_utc": df.published_utc, "updated_utc": pd.Series(pd.NaT, index=df.index, dtype=df.published_utc.dtype),
        "precision": classify_precision(df.published_utc),
        "headline": df.title.fillna(""), "summary": "", "body": df.body.fillna("").str.strip(), "url": df.url.fillna(""),
        "publisher": df.publisher.fillna("nasdaq.com"), "author": df.author.fillna(""),
        "symbols": df.url_norm.map(tags), "n_symbols": pd.NA,
    })
    return out


def annotate(news, sessions):
    warnings.filterwarnings("ignore", "This pattern is interpreted as a regular expression")
    news["eligible_session"] = eligible_session(news, sessions)
    news["eligible_at_utc"] = (news.eligible_session + pd.Timedelta(hours=17)).dt.tz_localize(NY).dt.tz_convert("UTC")
    news["ts_precision"] = np.where(news.precision == "sub_hour", "exact", "date_only")
    lead = (news.summary + " " + news.body.str[:LEAD_CHARS])
    pat = {t: re.compile(rf"\b({ISSUER.get(t, t)})", re.IGNORECASE) for t in news.ticker.unique()}
    in_head = np.array([bool(pat[t].search(h)) for t, h in zip(news.ticker, news.headline)])
    in_lead = np.array([bool(pat[t].search(s)) for t, s in zip(news.ticker, lead)])
    news["relevance"] = np.select([in_head, in_lead], ["headline", "lead"], "context")
    news["hindsight_title"] = news.headline.str.contains(HINDSIGHT)
    news["edited_after_1h"] = (news.updated_utc - news.published_utc) > pd.Timedelta(hours=1)
    news["has_body"] = news.body.str.len() > 200
    return news


def cross_source_dups(news):
    news["title_norm"] = norm_title(news.headline)
    a = news.loc[news.source == "alpaca", ["news_id", "ticker", "title_norm", "published_utc"]]
    f = news.loc[news.source == "fnspid", ["news_id", "ticker", "title_norm", "published_utc"]].reset_index()
    m = f.merge(a, on=["ticker", "title_norm"], suffixes=("", "_alp"))
    m = m[(m.title_norm.str.len() > 20)
          & ((m.published_utc - m.published_utc_alp).abs() <= pd.Timedelta(days=DUP_WINDOW_DAYS))]
    m = m.sort_values("published_utc_alp").drop_duplicates("index")
    news["dup_of"] = pd.Series(m.news_id_alp.values, index=m["index"].values).reindex(news.index)
    news["is_canonical"] = news.dup_of.isna()
    return news.drop(columns="title_norm")


def daily_panel(news, sessions, tickers):
    s = sessions[sessions >= START]
    idx = pd.MultiIndex.from_product([tickers, s], names=["ticker", "session"])
    c = news[news.is_canonical].assign(own=lambda d: d.relevance != "context")
    g = c.groupby(["ticker", "eligible_session", "source"])
    counts = pd.concat({"n": g.size(), "n_own": g.own.sum()}, axis=1).unstack("source", fill_value=0)
    counts.columns = [f"{k}_{src}" for k, src in counts.columns]
    counts.index = counts.index.set_names(["ticker", "session"])
    panel = counts.reindex(idx, fill_value=0).astype("int32").reset_index()
    panel["news_missing"] = panel.n_alpaca == 0
    return panel


def leak_test(news):
    """Own-company result headlines must not become eligible before the matching 8-K Item 2.02 is accepted."""
    tj = get_json("https://www.sec.gov/files/company_tickers.json", CACHE / "company_tickers.json")
    cik = {v["ticker"]: int(v["cik_str"]) for v in tj.values()}
    res = news[(news.relevance == "headline") & news.headline.str.contains(RESULT)
               & ~news.headline.str.contains(PREVIEW) & news.is_canonical]
    rows = []
    for t, g in res.groupby("ticker"):
        f = earnings_filings(t, cik[t]).sort_values("accepted_utc")
        acc = f.accepted_utc.reset_index(drop=True)
        i = np.searchsorted(acc.values, g.published_utc.values)
        for (_, a), j in zip(g.iterrows(), i):
            cands = [acc[k] for k in (j - 1, j) if 0 <= k < len(acc)]
            if not cands:
                continue
            r = min(cands, key=lambda x: abs(x - a.published_utc))
            if abs(a.published_utc - r) <= pd.Timedelta(days=10):
                rows.append({"ticker": t, "source": a.source, "ts_precision": a.ts_precision,
                             "published_utc": a.published_utc, "eligible_at_utc": a.eligible_at_utc,
                             "release_accepted_utc": r,
                             "eligible_minus_release_h": round((a.eligible_at_utc - r).total_seconds() / 3600, 2),
                             "headline": a.headline})
    out = pd.DataFrame(rows)
    out["eligible_before_release"] = out.eligible_minus_release_h < 0
    return out


def main():
    tickers, aliases, data_end = load_config()
    px = pd.read_parquet(ROOT / "data" / "processed" / "prices.parquet", columns=["date", "ticker"])
    sessions = pd.DatetimeIndex(sorted(px.loc[px.ticker == "SPY", "date"].unique()))

    news = pd.concat([load_alpaca(tickers, aliases), load_fnspid(tickers)], ignore_index=True)
    news["news_id"] = news.article_id + ":" + news.ticker
    news = news[news.published_utc >= START]
    news = annotate(news, sessions)
    late = news.eligible_session.isna() | (news.eligible_session > data_end)
    print(f"dropped {late.sum():,} rows eligible after {data_end}")
    news = cross_source_dups(news[~late].reset_index(drop=True))
    assert news.news_id.is_unique, "duplicate news_id"
    assert (news.eligible_at_utc > news.published_utc).all(), "article eligible before it was published"
    assert news.eligible_session.isin(sessions).all()

    cols = ["news_id", "article_id", "source", "ticker", "raw_symbol", "published_utc", "updated_utc", "ts_precision",
            "eligible_session", "eligible_at_utc", "headline", "summary", "body", "url", "publisher", "author",
            "symbols", "n_symbols", "relevance", "hindsight_title", "edited_after_1h", "has_body", "dup_of",
            "is_canonical"]
    news = news[cols].sort_values(["ticker", "published_utc", "news_id"]).reset_index(drop=True)
    news["n_symbols"] = news.n_symbols.astype("Int16")
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    news.to_parquet(OUT / "news.parquet", compression="zstd", index=False)
    panel = daily_panel(news, sessions, tickers)
    panel.to_parquet(OUT / "news_daily.parquet", index=False)

    pd.set_option("display.width", 250)
    news["year"] = news.eligible_session.dt.year
    cov = (news[news.is_canonical].groupby(["ticker", "source", "year"]).size()
           .unstack("year", fill_value=0))
    cov.to_csv(TABLES / "news_ticker_year.csv")
    q = news.groupby("source").agg(rows=("news_id", "size"), canonical=("is_canonical", "mean"),
                                   exact_ts=("ts_precision", lambda s: (s == "exact").mean()),
                                   body=("has_body", "mean"), hindsight=("hindsight_title", "mean"),
                                   edited_after_1h=("edited_after_1h", "mean")).round(3)
    rel = pd.crosstab(news.source, news.relevance, normalize="index").round(3)
    q = q.join(rel)
    q.to_csv(TABLES / "news_quality.csv")
    p = panel.assign(year=panel.session.dt.year)
    miss = p.pivot_table(index="ticker", columns="year", values="news_missing", aggfunc="mean").round(2)
    miss.to_csv(TABLES / "news_missing_share.csv")
    leaks = leak_test(news)
    leaks.to_csv(TABLES / "news_leak_test.csv", index=False)

    print(f"\n{len(news):,} rows ({news.is_canonical.sum():,} canonical), {news.ticker.nunique()} tickers, "
          f"eligible {news.eligible_session.min().date()} -> {news.eligible_session.max().date()}")
    print(q.to_string())
    print("\nshare of sessions with no Alpaca article:\n" + miss.to_string())
    print(f"\nleak test: {len(leaks)} result headlines matched to an 8-K 2.02; "
          f"{leaks.eligible_before_release.sum()} eligible before the filing")
    print(leaks.groupby(["source", pd.cut(leaks.eligible_minus_release_h, [-1e4, -24, -1, 0, 1, 24, 1e4])],
                        observed=True).size().unstack(0, fill_value=0).to_string())
    print(leaks[leaks.eligible_before_release].sort_values("eligible_minus_release_h")
          .head(25)[["ticker", "source", "published_utc", "release_accepted_utc", "eligible_minus_release_h", "headline"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
