"""Check FNSPID news dates against SEC 8-K Item 2.02 (earnings release) acceptance times.

Question answered: under the date-only rule (article dated D is usable at the 17:00 NY decision of
the first session after D), could any earnings-*result* article be used before the results were public?

  leak  <=>  earnings release accepted after 17:00 NY on the article's eligible session

Writes reports/data_audit/sec_date_check.csv (one row per matched article) and prints a summary.
"""

import json
import re
import time
import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "external" / "sec"
OUT = ROOT / "reports" / "data_audit"
UA = "AEGIS-research-prototype/0.1 (non-commercial student project)"
NY = "America/New_York"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "XOM", "AMD", "WMT",
           "INTC", "WFC", "CVX", "MRK", "KO", "GS", "SBUX", "QCOM"]
NEWS_SYMBOLS = {t: t for t in TICKERS} | {"GOOG": "GOOGL"}

RESULT = re.compile(
    r"(\b(beats?|tops?|misses?|surpass(es)?|exceeds?|lags?|trails?|falls? short of|in line with)\b.*"
    r"\b(estimates?|expectations?|views?|consensus|forecasts?)\b)"
    r"|(\b(reports?|posts?|announces?|delivers?)\b.*\b(Q[1-4]|quarter|quarterly)\b.*\b(results|earnings|profit|loss|EPS|revenue)s?\b)"
    r"|(\bearnings call transcript\b)"
    r"|(\bQ[1-4]\b.*\b(earnings|results|profit|EPS|revenue)s?\b.*\b(beat|top|miss|rise|rose|fall|fell|jump|surge|decline|drop|grow|grew)s?\b)",
    re.IGNORECASE,
)
PREVIEW = re.compile(
    r"\b(preview|ahead of|what to expect|expected to|to report|will report|set to|on deck|"
    r"earnings (date|due)|could|should|likely|estimates? (rise|fall)|before (the )?earnings|"
    r"week ahead|this week|options|implied|to post|in store|in the cards|things to watch|"
    r"reports? .* results (on )?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*)\b|\?",
    re.IGNORECASE,
)
# Headline must name the tagged company itself; Nasdaq tags many articles about peers/competitors.
ISSUER = {
    "AAPL": r"Apple|AAPL", "MSFT": r"Microsoft|MSFT", "NVDA": r"Nvidia|NVDA", "AMZN": r"Amazon|AMZN",
    "GOOGL": r"Alphabet|Google|GOOGL?", "XOM": r"Exxon|XOM", "AMD": r"AMD|Advanced Micro",
    "WMT": r"Walmart|Wal-Mart|WMT", "INTC": r"Intel|INTC", "WFC": r"Wells Fargo|WFC",
    "CVX": r"Chevron|CVX", "MRK": r"Merck|MRK", "KO": r"Coca-Cola|Coke|\bKO\b", "GS": r"Goldman|\bGS\b",
    "SBUX": r"Starbucks|SBUX", "QCOM": r"Qualcomm|QCOM",
}


def get_json(url, path):
    if path.exists():
        return json.loads(path.read_text())
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    time.sleep(0.15)  # stay well under SEC's 10 requests/second
    return data


def earnings_filings(ticker, cik):
    sub = get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", CACHE / f"{ticker}.json")
    pages = [sub["filings"]["recent"]]
    for i, f in enumerate(sub["filings"].get("files", [])):
        pages.append(get_json(f"https://data.sec.gov/submissions/{f['name']}", CACHE / f"{ticker}_{i}.json"))
    df = pd.concat([pd.DataFrame({k: p[k] for k in ("form", "acceptanceDateTime", "items")}) for p in pages])
    df = df[df.form.isin(["8-K", "8-K/A"]) & df["items"].fillna("").str.contains(r"\b2\.02\b")]
    df["accepted_utc"] = pd.to_datetime(df.acceptanceDateTime, utc=True)
    return df.assign(ticker=ticker)[["ticker", "form", "accepted_utc"]]


def load_news():
    mapping = ", ".join(f"('{r}', '{c}')" for r, c in NEWS_SYMBOLS.items())
    con = duckdb.connect()
    con.sql(f"CREATE TABLE alias(raw VARCHAR, ticker VARCHAR); INSERT INTO alias VALUES {mapping}")
    path = (ROOT / "Stock_news" / "nasdaq_exteral_data.csv").as_posix()
    return con.sql(f"""
        SELECT a.ticker, r.Date AS published_raw, r.Article_title AS title, r.Url AS url
        FROM read_csv('{path}', all_varchar=true, header=true, ignore_errors=true) r
        JOIN alias a ON upper(trim(r.Stock_symbol)) = a.raw
        WHERE r.Date >= '2016' AND r.Date < '2024'
    """).df()


def main():
    tickers = get_json("https://www.sec.gov/files/company_tickers.json", CACHE / "company_tickers.json")
    cik = {v["ticker"]: int(v["cik_str"]) for v in tickers.values()}
    filings = pd.concat([earnings_filings(t, cik[t]) for t in TICKERS])
    filings = filings[filings.accepted_utc.dt.year.between(2015, 2024)]
    filings["accepted_ny"] = filings.accepted_utc.dt.tz_convert(NY)
    filings["release_date"] = filings.accepted_ny.dt.tz_localize(None).dt.normalize()
    print(f"8-K Item 2.02 filings 2015-2024: {len(filings)}")
    print(filings.groupby("ticker").size().to_dict())
    print("release hour (NY) distribution:", filings.accepted_ny.dt.hour.value_counts().sort_index().to_dict())

    prices = pd.read_parquet(ROOT / "data" / "interim" / "fnspid" / "prices_raw.parquet")
    days = np.array(sorted(pd.to_datetime(prices.loc[prices.original_symbol == "SPY", "date"]).unique()),
                    dtype="datetime64[ns]")

    news = load_news()
    news["title"] = news.title.fillna("")
    news["names_issuer"] = [bool(re.search(rf"\b({ISSUER[t]})", s, re.IGNORECASE)) for t, s in
                            zip(news.ticker, news.title)]
    tag_share = news.groupby("ticker").names_issuer.mean().round(3)
    tag_share.rename("share_of_tagged_headlines_naming_issuer").to_csv(OUT / "ticker_tag_precision.csv")
    print("\nshare of tagged headlines that name the issuer:", tag_share.to_dict())
    news = news[news.names_issuer & news.title.str.contains(RESULT) & ~news.title.str.contains(PREVIEW)].copy()
    utc = pd.to_datetime(news.published_raw.str.replace(" UTC", "", regex=False), utc=True).dt.tz_localize(None)
    news["reported_date"] = utc.dt.normalize() + pd.to_timedelta((utc.dt.hour >= 12).astype(int), unit="D")
    idx = np.searchsorted(days, news.reported_date.values.astype("datetime64[ns]"), side="right")
    news = news[idx < len(days)]
    news["eligible_session"] = days[idx[idx < len(days)]]
    news["decision_ny"] = (news.eligible_session + pd.Timedelta(hours=17)).dt.tz_localize(NY)

    # Pair each result article with the latest earnings release accepted on or before reported_date + 3
    # days; result articles should never precede the release they describe.
    rows = []
    for t, g in news.groupby("ticker"):
        f = filings[filings.ticker == t].sort_values("accepted_utc")
        if f.empty:
            continue
        rel = f.release_date.values.astype("datetime64[ns]")
        for _, a in g.iterrows():
            j = np.searchsorted(rel, (a.reported_date + pd.Timedelta(days=3)).to_datetime64(), side="right") - 1
            if j < 0:
                continue
            r = f.iloc[j]
            gap = (a.reported_date - r.release_date).days
            if gap > 10:  # article not near any release (e.g. regex false positive)
                continue
            rows.append({"ticker": t, "published_raw": a.published_raw, "title": a.title, "url": a.url,
                         "release_accepted_ny": r.accepted_ny, "reported_minus_release_days": gap,
                         "eligible_session": a.eligible_session.date(),
                         "leak": r.accepted_ny > a.decision_ny})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "sec_date_check.csv", index=False)

    print(f"\nresult-type articles matched to a release (within 10 days): {len(res)}")
    print("reported date minus release date (days):")
    print(res.reported_minus_release_days.value_counts().sort_index().to_string())
    print(f"\nLEAKS under the date-only rule: {int(res.leak.sum())} of {len(res)} "
          f"({res.leak.mean():.2%})")
    print(res.groupby("ticker").agg(n=("leak", "size"), leaks=("leak", "sum"),
                                    dated_before_release=("reported_minus_release_days",
                                                          lambda s: int((s < 0).sum()))).to_string())
    if res.leak.any():
        print("\nleaking examples:")
        print(res[res.leak].head(15)[["ticker", "published_raw", "release_accepted_ny", "title"]].to_string())
    early = res[res.reported_minus_release_days < 0]
    if len(early):
        print("\ndated-before-release examples:")
        print(early.head(15)[["ticker", "published_raw", "release_accepted_ny", "title"]].to_string())


if __name__ == "__main__":
    main()
