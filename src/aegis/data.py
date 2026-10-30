"""Access to the canonical tables. Development code cannot read past the end of validation."""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


class DevelopmentOnlyError(Exception):
    """Raised when development code asks for data from the frozen test or external periods."""


def splits():
    return yaml.safe_load((ROOT / "configs" / "splits.yaml").read_text())


def universe():
    return yaml.safe_load((ROOT / "configs" / "universe.yaml").read_text())


def dev_end():
    return pd.Timestamp(splits()["periods"]["val"][1])


def check_dev_cutoff(until):
    if pd.Timestamp(until) > dev_end():
        raise DevelopmentOnlyError(f"{until} is after the end of validation ({dev_end().date()})")


def load_prices(until=None):
    until = dev_end() if until is None else pd.Timestamp(until)
    check_dev_cutoff(until)
    return pd.read_parquet(PROCESSED / "prices.parquet", filters=[("date", "<=", until)])


def load_riskfree(until=None):
    until = dev_end() if until is None else pd.Timestamp(until)
    check_dev_cutoff(until)
    return pd.read_parquet(PROCESSED / "riskfree.parquet").loc[:until]


def riskfree_from_tbill(tbill_pct, sessions):
    """Bond-equivalent yield from a 3-month T-bill discount rate, plus cash earned to the next session."""
    d = tbill_pct.sort_index().ffill(limit=5).reindex(sessions) / 100
    annual = 365 * d / (360 - 91 * d)
    days = pd.Series(sessions, index=sessions).diff().shift(-1).dt.days.fillna(1)
    return pd.DataFrame({"rf_annual": annual, "rf_period": annual * days / 365}, index=sessions)
