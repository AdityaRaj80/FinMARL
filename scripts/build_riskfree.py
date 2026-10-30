"""Risk-free series from FRED's 3-month T-bill discount rate (DTB3), aligned to trading sessions.

  python scripts/build_riskfree.py

Outputs: data/external/fred/DTB3.csv (frozen download, re-used if present) and data/processed/riskfree.parquet.
rf_period is the cash earned from one session to the next. Quotes are published after the close, so anything
used as a model input at time t must take the previous session's value; the backtest may accrue the same day's.
"""

import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aegis.data import PROCESSED, riskfree_from_tbill, splits  # noqa: E402

RAW = ROOT / "data" / "external" / "fred" / "DTB3.csv"
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd=2014-12-01&coed={end}"


def main():
    end = str(splits()["data_end"])
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL.format(end=end), timeout=60) as r:
            RAW.write_bytes(r.read())
    raw = pd.read_csv(RAW, parse_dates=["observation_date"], index_col="observation_date")
    tbill = pd.to_numeric(raw.DTB3, errors="coerce")
    px = pd.read_parquet(PROCESSED / "prices.parquet", columns=["date", "ticker"])
    sessions = pd.DatetimeIndex(sorted(px.loc[px.ticker == "SPY", "date"].unique()))
    rf = riskfree_from_tbill(tbill, sessions)
    missing = rf.rf_annual.isna().sum()
    if missing:
        sys.exit(f"{missing} sessions have no risk-free value")
    rf.to_parquet(PROCESSED / "riskfree.parquet")
    by_year = rf.rf_annual.groupby(rf.index.year).mean().round(4)
    print(f"{len(rf)} sessions {rf.index[0].date()} -> {rf.index[-1].date()}\n{by_year.to_string()}")


if __name__ == "__main__":
    main()
