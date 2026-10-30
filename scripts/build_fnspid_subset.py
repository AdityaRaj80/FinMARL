"""Stream-filter the raw FNSPID files down to the fixed AEGIS universe.

Outputs (under data/interim/fnspid/):
  news_raw.parquet    one row per raw news record for universe symbols (incl. aliases)
  prices_raw.parquet  daily bars for universe + benchmark symbols
  build_manifest.json source file sizes/hashes, row counts, rejected-row counts

Raw text fields are kept as strings; nothing is parsed or deduplicated here.
"""

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
NEWS_FILES = {
    "nasdaq": ROOT / "Stock_news" / "nasdaq_exteral_data.csv",
    "external": ROOT / "Stock_news" / "All_external.csv",
}
PRICE_ZIP = ROOT / "Stock_price" / "full_history.zip"
OUT = ROOT / "data" / "interim" / "fnspid"


def load_universe():
    cfg = yaml.safe_load((ROOT / "configs" / "universe.yaml").read_text())
    tradeable = [t for names in cfg["universe"].values() for t in names] + cfg["reserves"]
    news_symbols = {s: s for s in tradeable} | cfg["aliases"]
    # Price files only for the old-ticker aliases that were real listings (GOOG is a different share class).
    price_symbols = {s: s for s in tradeable + cfg["benchmarks"]} | {a: c for a, c in cfg["aliases"].items() if a != "GOOG"}
    return cfg, news_symbols, price_symbols


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_news(con, news_symbols, manifest):
    mapping = ", ".join(f"('{raw}', '{canon}')" for raw, canon in news_symbols.items())
    con.sql(f"CREATE OR REPLACE TABLE alias(raw_symbol VARCHAR, canonical_ticker VARCHAR); "
            f"INSERT INTO alias VALUES {mapping}")
    parts = []
    for source, path in NEWS_FILES.items():
        t0 = time.time()
        rejects = f"rejects_{source}"
        con.sql(f"""
            CREATE OR REPLACE TABLE news_{source} AS
            SELECT '{source}' AS source_file,
                   r.Date AS published_raw,
                   r.Article_title AS title,
                   r.Stock_symbol AS original_symbol,
                   a.canonical_ticker,
                   r.Url AS url,
                   r.Publisher AS publisher,
                   r.Author AS author,
                   r.Article AS body,
                   r.Lsa_summary IS NOT NULL AS has_lsa_summary
            FROM read_csv('{path.as_posix()}', all_varchar=true, header=true,
                          store_rejects=true, rejects_table='{rejects}',
                          rejects_scan='{rejects}_scan') r
            JOIN alias a ON upper(trim(r.Stock_symbol)) = a.raw_symbol
        """)
        n = con.sql(f"SELECT count(*) FROM news_{source}").fetchone()[0]
        n_rej = con.sql(f"SELECT count(*) FROM {rejects}").fetchone()[0]
        manifest["news"][source] = {"kept_rows": n, "rejected_rows": n_rej,
                                    "seconds": round(time.time() - t0, 1)}
        print(f"[news] {source}: kept {n:,} rows, {n_rej:,} rejected, {time.time() - t0:.0f}s")
        parts.append(f"SELECT * FROM news_{source}")
    con.sql(f"COPY ({' UNION ALL '.join(parts)}) TO '{(OUT / 'news_raw.parquet').as_posix()}' "
            "(FORMAT parquet, COMPRESSION zstd)")


def build_prices(price_symbols, manifest):
    frames = []
    with zipfile.ZipFile(PRICE_ZIP) as zf:
        # Match case-insensitively (e.g. nvda.csv) and skip macOS resource forks.
        members = {Path(n).stem.upper(): n for n in zf.namelist()
                   if n.startswith("full_history/") and n.lower().endswith(".csv")}
        for raw, canon in price_symbols.items():
            if raw not in members:
                manifest["prices"]["missing"].append(raw)
                continue
            with zf.open(members[raw]) as f:
                df = pd.read_csv(f, dtype={"date": str})
            df = df.rename(columns={"adj close": "adj_close"})
            df.insert(0, "original_symbol", raw)
            df.insert(1, "canonical_ticker", canon)
            frames.append(df)
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices.sort_values(["original_symbol", "date"]).reset_index(drop=True)
    prices.to_parquet(OUT / "prices_raw.parquet", index=False)
    manifest["prices"]["rows"] = len(prices)
    print(f"[prices] {len(prices):,} rows, missing symbols: {manifest['prices']['missing']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hash", action="store_true", help="skip SHA256 of raw files")
    ap.add_argument("--only", choices=["news", "prices"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg, news_symbols, price_symbols = load_universe()
    manifest = {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "universe": cfg,
                "sources": {}, "news": {}, "prices": {"missing": []}}
    for path in [*NEWS_FILES.values(), PRICE_ZIP]:
        entry = {"bytes": path.stat().st_size}
        if not args.skip_hash:
            entry["sha256"] = sha256(path)
        manifest["sources"][path.relative_to(ROOT).as_posix()] = entry

    if args.only != "news":
        build_prices(price_symbols, manifest)
    if args.only != "prices":
        con = duckdb.connect(str(OUT / "_build.duckdb"))
        build_news(con, news_symbols, manifest)
        con.close()
        (OUT / "_build.duckdb").unlink()

    (OUT / "build_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
