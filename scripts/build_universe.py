"""Point-in-time universe selection, decided before any return is looked at.

Rule (pre-registered 2026-10-29):
  1. Candidates = S&P 500 members on 2015-12-31 (the day before training starts), reconstructed from
     the pinned Wikipedia revision's constituent list by undoing every change dated after 2015-12-31.
  2. Liquidity = mean daily dollar volume (close x volume) over 2015, from Yahoo. One share class per issuer.
  3. Pick the 2 most liquid names in each of the 11 GICS sectors, then fill to 25 with the next most
     liquid names regardless of sector. The next 5 are registered reserves, used in order if a pick
     fails the price gate.
  4. Price gate: continuous adjusted history 2015-01-02 -> today, and every |daily return| > 25% must be
     a dated market event (checked by hand; listed in reports/universe_selection.md).

No news counts and no returns are used to choose names. Survivorship caveat: members that were later
acquired or delisted have no free Yahoo history and drop out at step 2; they are listed in the output.
"""

import io
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "universe"
REVID = 1355685534  # List_of_S&P_500_companies, 2026-05-23 (last revision found with the changes table)
AS_OF = "2015-12-31"
N_PER_SECTOR, N_TOTAL, N_RESERVE = 2, 25, 5
SECOND_CLASS = {"GOOG", "FOX", "NWS", "DISCK", "UA"}  # keep GOOGL, FOXA, NWSA, DISCA, UAA
RENAMED = {"ACE": "CB", "GPS": "GAP"}  # same issuer, new Yahoo ticker
UA = {"User-Agent": "AEGIS-research-prototype/0.1 (non-commercial student project)"}


def members_as_of():
    path = OUT / "sp500_wiki_rev.html"
    if not path.exists():
        url = f"https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid={REVID}"
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode())
    cur, ch = pd.read_html(io.StringIO(path.read_text(encoding="utf-8")))[:2]
    ch.columns = ["date", "add_t", "add_n", "rem_t", "rem_n", "reason"]
    ch["date"] = pd.to_datetime(ch.date, format="mixed")
    after = ch[ch.date > AS_OF].sort_values("date", ascending=False)
    members = set(cur.Symbol)
    for _, r in after.iterrows():  # undo newest change first
        members.discard(r.add_t)
        if isinstance(r.rem_t, str):
            members.add(r.rem_t)
    names = dict(zip(cur.Symbol, cur.Security)) | dict(zip(after.rem_t, after.rem_n))
    df = pd.DataFrame({"ticker": sorted(members)})
    df["name"] = df.ticker.map(names)
    df["sector"] = df.ticker.map(dict(zip(cur.Symbol, cur["GICS Sector"])))
    df["still_member"] = df.ticker.isin(cur.Symbol)
    return df


def liquidity(df):
    yt = {t: RENAMED.get(t, t).replace(".", "-") for t in df.ticker}
    px = yf.download(list(yt.values()), start="2015-01-01", end="2016-01-01", auto_adjust=False, progress=False)
    dv, n = (px["Close"] * px["Volume"]).mean(), px["Close"].notna().sum()
    df["adv_2015_musd"] = df.ticker.map(lambda t: dv.get(yt[t])) / 1e6
    df["days_2015"] = df.ticker.map(lambda t: n.get(yt[t], 0))
    for t in df.loc[df.days_2015 < 200, "ticker"]:  # batch downloads drop tickers; retry one by one
        d = yf.Ticker(yt[t]).history(start="2015-01-01", end="2016-01-01", auto_adjust=False)
        time.sleep(0.3)
        if len(d) >= 200:
            df.loc[df.ticker == t, ["adv_2015_musd", "days_2015"]] = [(d.Close * d.Volume).mean() / 1e6, len(d)]
    return df


def select(df):
    ok = df[(df.days_2015 >= 200) & df.sector.notna() & ~df.ticker.isin(SECOND_CLASS)]
    ok = ok.sort_values("adv_2015_musd", ascending=False).copy()
    ok["rank_in_sector"] = ok.groupby("sector").cumcount() + 1
    picks = ok[ok.rank_in_sector <= N_PER_SECTOR].assign(reason="sector top-2")
    rest = ok.drop(picks.index)
    fill = rest.head(N_TOTAL - len(picks)).assign(reason="liquidity fill")
    reserve = rest.iloc[len(fill):len(fill) + N_RESERVE].assign(reason="reserve")
    return pd.concat([picks, fill, reserve])


def price_gate(tickers):
    px = yf.download(tickers, start="2015-01-01", auto_adjust=False, progress=False)["Adj Close"]
    r = px.pct_change(fill_method=None)
    return pd.DataFrame({
        "first_bar": px.apply(lambda s: s.first_valid_index().date()),
        "missing_bars": px.loc["2015-01-02":].isna().sum(),
        "big_moves": r.apply(lambda s: ", ".join(f"{d.date()}:{v:+.0%}" for d, v in s[s.abs() > 0.25].items())),
    })


def main():
    df = liquidity(members_as_of())
    df.to_csv(OUT / f"sp500_{AS_OF}.csv", index=False)
    sel = select(df)
    gate = price_gate(sel.ticker.tolist())
    sel = sel.join(gate, on="ticker")
    cols = ["ticker", "name", "sector", "reason", "rank_in_sector", "adv_2015_musd", "first_bar", "missing_bars", "big_moves"]
    sel[cols].round(1).to_csv(ROOT / "reports" / "universe_selection.csv", index=False)
    print(sel[cols].round(0).to_string(index=False))
    gone = df[df.days_2015 < 200]
    print(f"\n{len(df)} members on {AS_OF}; {len(gone)} without free 2015 history (survivorship hole):")
    print(", ".join(gone.ticker))
    (OUT / "selection_meta.json").write_text(json.dumps(
        {"wiki_revid": REVID, "as_of": AS_OF, "members": len(df), "no_history": gone.ticker.tolist()}, indent=1))


if __name__ == "__main__":
    main()
