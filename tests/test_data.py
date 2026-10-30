"""The frozen periods must be unreachable from development code, and risk-free conversion must be right."""

import numpy as np
import pandas as pd
import pytest

from aegis.data import DevelopmentOnlyError, check_dev_cutoff, riskfree_from_tbill


def test_cutoff_inside_validation_is_allowed():
    check_dev_cutoff("2022-12-30")


def test_cutoff_inside_frozen_test_raises():
    with pytest.raises(DevelopmentOnlyError):
        check_dev_cutoff("2023-01-03")


def test_tbill_discount_rate_becomes_a_bond_equivalent_yield():
    # 5.00% discount on a 91-day bill: price 0.987361, yield 365/91 * (1/price - 1) = 5.1343%
    sessions = pd.DatetimeIndex(["2022-01-03"])
    rf = riskfree_from_tbill(pd.Series([5.0], index=sessions), sessions)
    assert rf.rf_annual.iloc[0] == pytest.approx(0.051343, abs=1e-5)


def test_cash_accrues_over_calendar_days_including_weekends():
    sessions = pd.DatetimeIndex(["2022-01-07", "2022-01-10"])  # Friday, Monday
    rf = riskfree_from_tbill(pd.Series([3.65, 3.65], index=sessions), sessions)
    friday_to_monday = rf.rf_period.iloc[0]
    assert friday_to_monday == pytest.approx(rf.rf_annual.iloc[0] * 3 / 365)


def test_bond_market_holiday_uses_the_previous_quote():
    sessions = pd.DatetimeIndex(["2022-10-07", "2022-10-10", "2022-10-11"])  # Columbus Day: stocks open, no T-bill
    tbill = pd.Series([3.0, np.nan, 3.2], index=sessions)
    rf = riskfree_from_tbill(tbill, sessions)
    assert rf.rf_annual.iloc[1] == rf.rf_annual.iloc[0]
