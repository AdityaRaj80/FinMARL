"""Probe Alpaca's historical news API (Benzinga feed) as a candidate primary news source.

  python scripts/check_alpaca_news.py probe      # access check + earliest/latest article per ticker
  python scripts/check_alpaca_news.py pull       # page through 2015-2026 per ticker -> data/external/alpaca/
  python scripts/check_alpaca_news.py report     # coverage by ticker-year + timing check vs SEC 8-K Item 2.02

Keys come from APCA_API_KEY_ID / APCA_API_SECRET_KEY (process env, or the Windows user env set via setx).
Article text is kept local only (non-redistributable).
"""

import http.client
import json
import os
import re
import sys
import time
import warnings
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_dates_vs_sec import CACHE, ISSUER, PREVIEW, RESULT, earnings_filings, get_json  # noqa: E402

URL = "https://data.alpaca.markets/v1beta1/news"
OUT = ROOT / "data" / "external" / "alpaca"
REPORT = ROOT / "reports" / "data_audit"
NY = "America/New_York"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "AMD", "BAC", "JNJ", "WMT",
           "INTC", "WFC", "CVX", "MRK", "KO", "GS", "SBUX", "QCOM", "T", "GE", "MS", "CRM", "FB"]
ALIASES = {"FB": "META", "PCLN": "BKNG", "DWDP": "DD"}  # pre-rename tickers (Meta 2022, Booking 2018, DowDuPont 2017-19)
ISSUER = ISSUER | {
    "META": r"Meta\b|Facebook|\bFB\b", "JPM": r"JPMorgan|JP Morgan|\bJPM\b", "BAC": r"Bank of America|BofA|\bBAC\b",
    "JNJ": r"Johnson & Johnson|J&J|\bJNJ\b", "T": r"AT&T|\bAT ?& ?T\b", "GE": r"General Electric|\bGE\b",
    "MS": r"Morgan Stanley", "CRM": r"Salesforce|\bCRM\b",
    "NFLX": r"Netflix|NFLX", "BKNG": r"Booking|Priceline|BKNG|PCLN", "PG": r"Procter|P&G|\bPG\b",
    "PFE": r"Pfizer|PFE", "GILD": r"Gilead|GILD", "BA": r"Boeing|\bBA\b", "DD": r"DuPont|DowDuPont|\bDD\b",
    "FCX": r"Freeport|FCX", "SPG": r"Simon Property|\bSPG\b", "AMT": r"American Tower|\bAMT\b",
    "DUK": r"Duke Energy|\bDUK\b", "EXC": r"Exelon|\bEXC\b", "C": r"Citigroup|Citi\b|Citibank",
}
START, END = "2015-01-01T00:00:00Z", "2026-10-29T00:00:00Z"


def key(name):
    v = os.environ.get(name)
    if not v and sys.platform == "win32":  # picks up `setx` without restarting the app
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                v = winreg.QueryValueEx(k, name)[0]
        except OSError:
            pass
    if not v:
        sys.exit(f"{name} is not set")
    return v


HEADERS = None


def fetch(params, retries=5):
    global HEADERS
    if HEADERS is None:
        HEADERS = {"APCA-API-KEY-ID": key("APCA_API_KEY_ID"), "APCA-API-SECRET-KEY": key("APCA_API_SECRET_KEY")}
    req = urllib.request.Request(f"{URL}?{urllib.parse.urlencode(params)}", headers=HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                time.sleep(0.31)  # free plan: 200 requests/minute
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(2 ** attempt * 3)
                continue
            sys.exit(f"HTTP {e.code}: {e.read().decode()[:300]}")
        except (http.client.HTTPException, urllib.error.URLError, OSError, json.JSONDecodeError):
            time.sleep(2 ** attempt * 3)  # dropped connection / truncated chunked body
    sys.exit("gave up after repeated errors")


def probe():
    for t in ["AAPL", "INTC", "META"]:
        first = fetch({"symbols": t, "start": START, "end": END, "sort": "asc", "limit": 1})["news"]
        last = fetch({"symbols": t, "start": START, "end": END, "sort": "desc", "limit": 1})["news"]
        for label, n in (("first", first), ("last", last)):
            if n:
                a = n[0]
                print(f"{t} {label}: {a['created_at']}  [{a['source']}]  {a['headline'][:90]}")
            else:
                print(f"{t} {label}: none")
    sample = fetch({"symbols": "AAPL", "start": "2019-01-28T00:00:00Z", "end": "2019-02-01T00:00:00Z",
                    "limit": 50, "include_content": "true"})["news"]
    print(f"\nAAPL 2019-01-28..31: {len(sample)} articles; with content: {sum(bool(a.get('content')) for a in sample)}")
    print("fields:", sorted(sample[0]) if sample else "-")


def pull(tickers):
    OUT.mkdir(parents=True, exist_ok=True)
    years = range(int(START[:4]), int(END[:4]) + 1)
    for t in tickers:
        for y in years:  # one file per ticker-year so an interrupted run resumes cheaply
            path = OUT / f"news_{t}_{y}.parquet"
            if path.exists():
                continue
            start, end = f"{y}-01-01T00:00:00Z", min(f"{y + 1}-01-01T00:00:00Z", END)
            rows, token = [], None
            while True:
                p = {"symbols": t, "start": start, "end": end, "sort": "asc", "limit": 50, "include_content": "true"}
                if token:
                    p["page_token"] = token
                d = fetch(p)
                rows += d["news"]
                token = d.get("next_page_token")
                if not token:
                    break
            df = pd.DataFrame(rows)
            if len(df):
                df["symbols"] = df.symbols.map(",".join)
                df = df.drop(columns=["images"], errors="ignore").assign(ticker=t)
            df.to_parquet(path, compression="zstd")
            print(f"{t} {y}: {len(df)} articles", flush=True)


def report():
    warnings.filterwarnings("ignore", "This pattern is interpreted as a regular expression")
    news = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("news_*.parquet"))], ignore_index=True)
    news["ticker"] = news.ticker.replace(ALIASES)
    news = news.drop_duplicates(["ticker", "id"])
    news["created_at"] = pd.to_datetime(news.created_at, utc=True)
    news["updated_at"] = pd.to_datetime(news.updated_at, utc=True)
    news["year"] = news.created_at.dt.year
    cov = news.groupby(["ticker", "year"]).size().unstack(fill_value=0)
    cov["min_18_23"] = cov.loc[:, 2018:2023].min(axis=1)
    cov = cov.sort_values("min_18_23", ascending=False)
    cov.to_csv(REPORT / "alpaca_ticker_year.csv")
    print(cov.to_string())

    news["n_symbols"] = news.symbols.str.count(",") + 1
    news["names_issuer"] = [bool(re.search(rf"\b({ISSUER.get(t, t)})", h or "", re.IGNORECASE))
                            for t, h in zip(news.ticker, news.headline)]
    q = news.groupby("ticker").agg(
        articles=("id", "size"), unique_ids=("id", "nunique"),
        body_share=("content", lambda s: round((s.fillna("").str.len() > 200).mean(), 3)),
        names_issuer=("names_issuer", "mean"), single_symbol=("n_symbols", lambda s: round((s == 1).mean(), 3)))
    q["edited_after_1h"] = news.assign(e=(news.updated_at - news.created_at) > pd.Timedelta(hours=1)) \
        .groupby("ticker").e.mean().round(3)
    q["names_issuer"] = q.names_issuer.round(3)
    print("\n", q.to_string())
    print("\nsources:", news.source.value_counts().head(5).to_dict())
    print("minute-of-hour == 0 share (backfill smell):", round((news.created_at.dt.minute == 0).mean(), 3))
    q.to_csv(REPORT / "alpaca_quality.csv")

    # Timing vs SEC: own-company result headlines should appear at or shortly after the earnings release.
    tickers_json = get_json("https://www.sec.gov/files/company_tickers.json", CACHE / "company_tickers.json")
    cik = {v["ticker"]: int(v["cik_str"]) for v in tickers_json.values()}
    sec_t = [t for t in news.ticker.unique() if t in ISSUER and t in cik]
    filings = pd.concat([earnings_filings(t, cik[t]) for t in sec_t])
    res = news[news.ticker.isin(sec_t) & news.names_issuer & news.headline.str.contains(RESULT)
               & ~news.headline.str.contains(PREVIEW)].copy()
    rows = []
    for t, g in res.groupby("ticker"):
        f = filings[filings.ticker == t].sort_values("accepted_utc")
        acc = f.accepted_utc.values
        for _, a in g.iterrows():
            # nearest release within +/- 10 days
            i = acc.searchsorted(a.created_at.to_datetime64())
            cands = [f.iloc[j].accepted_utc for j in (i - 1, i) if 0 <= j < len(f)]
            if not cands:
                continue
            r = min(cands, key=lambda x: abs(x - a.created_at))
            gap_h = (a.created_at - r).total_seconds() / 3600
            if abs(gap_h) <= 240:
                rows.append({"ticker": t, "created_at": a.created_at, "release_accepted": r,
                             "article_minus_release_h": round(gap_h, 2), "headline": a.headline})
    chk = pd.DataFrame(rows)
    chk.to_csv(REPORT / "alpaca_sec_timing.csv", index=False)
    h = chk.article_minus_release_h
    bins = [-240, -24, -1, -0.25, 0.25, 1, 24, 240]
    print(f"\nresult headlines matched to an 8-K 2.02 release: {len(chk)}")
    print(pd.cut(h, bins).value_counts().sort_index().to_string())
    print("\nmore than 1h BEFORE the 8-K (inspect):")
    print(chk[h < -1].sort_values("article_minus_release_h").head(20).to_string())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    {"probe": probe, "pull": lambda: pull(sys.argv[2:] or TICKERS), "report": report}[mode]()
