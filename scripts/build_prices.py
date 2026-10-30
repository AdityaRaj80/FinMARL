"""Canonical daily price table for the universe, reserves and benchmarks (yfinance, frozen snapshot).

  python scripts/build_prices.py

Outputs:
  data/external/yfinance/raw_<data_end>.parquet  Yahoo download as received (frozen; re-used if present,
                                                 because Yahoo silently revises adjustments)
  data/processed/prices.parquet                  one row per (ticker, session), long format
  data/processed/corporate_actions.parquet       dividends and splits by ex-date
  reports/data_tables/price_checks.csv           per-ticker gate results

Yahoo's Open/High/Low/Close/Volume are split-adjusted but not dividend-adjusted; Adj Close is split- and
dividend-adjusted. adj_factor = Adj Close / Close, and adj_open/high/low apply the same factor, so
adjusted open-to-open returns are total returns. Adjusted *levels* embed later dividends and must not be
used as features; returns and ratios are point-in-time safe.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "yfinance"
OUT = ROOT / "data" / "processed"
CHECKS = ROOT / "reports" / "data_tables"
NY = "America/New_York"
BIG_MOVE = 0.25


def load_config():
    uni = yaml.safe_load((ROOT / "configs" / "universe.yaml").read_text())
    spl = yaml.safe_load((ROOT / "configs" / "splits.yaml").read_text())
    role = {t: "universe" for names in uni["universe"].values() for t in names}
    role |= {t: "reserve" for t in uni["reserves"]} | {t: "benchmark" for t in uni["benchmarks"]}
    sector = {t: s for s, names in uni["universe"].items() for t in names}
    return role, sector, str(spl["data_start"]), str(spl["data_end"])


def download(tickers, start, end):
    path = RAW / f"raw_{end}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    RAW.mkdir(parents=True, exist_ok=True)
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    px = yf.download(tickers, start=start, end=end_excl, auto_adjust=False, actions=True,
                     progress=False, group_by="column")
    raw = px.stack(level="Ticker", future_stack=True).reset_index()
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
    raw = raw.rename(columns={"stock_splits": "split"})
    missing = sorted(set(tickers) - set(raw.dropna(subset=["close"]).ticker))
    if missing:
        sys.exit(f"Yahoo returned no bars for {missing}; re-run before freezing a snapshot")
    raw["downloaded_at_utc"] = pd.Timestamp.now(tz="UTC")
    raw.to_parquet(path, compression="zstd", index=False)
    return raw


def build(raw, role, sector, data_end):
    df = raw.rename(columns={"date": "date"}).copy()
    df["date"] = pd.to_datetime(df.date).dt.tz_localize(None).dt.normalize()
    df = df[df.date <= data_end]
    sessions = pd.DatetimeIndex(sorted(df.loc[df.ticker == "SPY", "date"].unique()))
    df = df[df.date.isin(sessions)].dropna(subset=["close"]).sort_values(["ticker", "date"])

    df["adj_factor"] = df.adj_close / df.close
    for c in ("open", "high", "low"):
        df[f"adj_{c}"] = df[c] * df.adj_factor
    g = df.groupby("ticker")
    df["ret_cc"] = np.log(df.adj_close / g.adj_close.shift())
    df["ret_oo"] = np.log(df.adj_open / g.adj_open.shift())
    df["dollar_volume"] = df.close * df.volume
    # Close of session D is known at 16:00 NY (13:00 on half-days, so this is conservative).
    df["available_at_utc"] = (df.date + pd.Timedelta(hours=16)).dt.tz_localize(NY).dt.tz_convert("UTC")
    df["role"] = df.ticker.map(role)
    df["sector"] = df.ticker.map(sector)

    flags = pd.DataFrame({
        "ohlc_inconsistent": (df.low > df[["open", "close"]].min(axis=1) * 1.0001)
                             | (df.high < df[["open", "close"]].max(axis=1) * 0.9999),
        "zero_volume": df.volume <= 0,
        "big_move": df.ret_cc.abs() > np.log1p(BIG_MOVE),
    }, index=df.index)
    df["quality_flags"] = flags.apply(lambda r: ",".join(flags.columns[r.values]), axis=1)
    df["vendor"] = "yahoo"
    cols = ["date", "ticker", "role", "sector", "open", "high", "low", "close", "volume", "adj_close",
            "adj_factor", "adj_open", "adj_high", "adj_low", "ret_cc", "ret_oo", "dollar_volume",
            "dividends", "split", "available_at_utc", "quality_flags", "vendor", "downloaded_at_utc"]
    return df[cols].reset_index(drop=True), sessions


def checks(px, sessions, data_start):
    span = sessions[sessions >= data_start]
    rows = []
    for t, g in px.groupby("ticker"):
        f = g.adj_factor.values
        rows.append({
            "ticker": t, "role": g.role.iat[0], "first": g.date.min().date(), "last": g.date.max().date(),
            "bars": len(g), "missing_sessions": len(span.difference(g.date)), "dup_sessions": g.date.duplicated().sum(),
            "nonpositive_prices": int((g[["open", "high", "low", "close", "adj_close"]] <= 0).any(axis=1).sum()),
            "ohlc_inconsistent": int(g.quality_flags.str.contains("ohlc").sum()),
            "zero_volume": int(g.quality_flags.str.contains("zero_volume").sum()),
            # Adj factor must be flat or rise over time (each ex-dividend date lowers all earlier factors).
            "adj_factor_drops": int((np.diff(f) < -1e-4 * f[1:]).sum()),
            "big_moves": ", ".join(f"{d.date()}:{np.expm1(r):+.0%}" for d, r in zip(g.date, g.ret_cc)
                                   if abs(r) > np.log1p(BIG_MOVE)),
            "dividends": int((g.dividends > 0).sum()), "splits": ", ".join(
                f"{d.date()}:{s:g}" for d, s in zip(g.date, g.split) if s > 0),
        })
    return pd.DataFrame(rows)


def cross_check_fnspid(px):
    """Compare split-adjusted daily close returns with FNSPID's post-2020-07-06 segment."""
    path = ROOT / "data" / "interim" / "fnspid" / "prices_raw.parquet"
    if not path.exists():
        return pd.DataFrame()
    f = pd.read_parquet(path)
    f = f[f.original_symbol == f.canonical_ticker].assign(date=lambda d: pd.to_datetime(d.date))
    f = f[(f.date >= "2020-07-07") & (f.date <= "2023-12-28")]
    m = px.merge(f[["date", "canonical_ticker", "adj_close"]].rename(
        columns={"canonical_ticker": "ticker", "adj_close": "fn_adj"}), on=["date", "ticker"])
    m = m.sort_values(["ticker", "date"])
    m["r_yf"] = m.groupby("ticker").adj_close.pct_change()
    m["r_fn"] = m.groupby("ticker").fn_adj.pct_change()
    d = (m.r_yf - m.r_fn).abs()
    return (m.assign(d=d).dropna(subset=["d"]).groupby("ticker")
            .agg(days=("d", "size"), corr=("r_yf", lambda s: s.corr(m.loc[s.index, "r_fn"])),
                 median_abs_diff_bp=("d", lambda s: s.median() * 1e4),
                 days_diff_over_50bp=("d", lambda s: int((s > 0.005).sum())))
            .round(4).reset_index())


def main():
    role, sector, start, end = load_config()
    raw = download(sorted(role), start, end)
    px, sessions = build(raw, role, sector, end)
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)
    px.to_parquet(OUT / "prices.parquet", compression="zstd", index=False)
    acts = px[(px.dividends > 0) | (px.split > 0)][["date", "ticker", "dividends", "split"]]
    acts.to_parquet(OUT / "corporate_actions.parquet", index=False)

    chk = checks(px, sessions, start)
    chk.to_csv(CHECKS / "price_checks.csv", index=False)
    xc = cross_check_fnspid(px)
    xc.to_csv(CHECKS / "price_vs_fnspid.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"{len(px):,} rows, {px.ticker.nunique()} tickers, {len(sessions)} sessions "
          f"{sessions[0].date()} -> {sessions[-1].date()}")
    print(chk.drop(columns=["big_moves"]).to_string(index=False))
    print("\nbig moves:\n" + chk.loc[chk.big_moves != "", ["ticker", "big_moves"]].to_string(index=False))
    print("\nvs FNSPID (2020-07-07 -> 2023-12-28):\n" + xc.to_string(index=False))
    bad = chk[(chk.missing_sessions > 0) | (chk.dup_sessions > 0) | (chk.nonpositive_prices > 0)]
    if len(bad):
        sys.exit(f"price gate FAILED for {bad.ticker.tolist()}")


if __name__ == "__main__":
    main()
