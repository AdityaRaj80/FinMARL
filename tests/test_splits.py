"""No training label may be finished at or after the start of its validation year."""

import pandas as pd

from aegis.splits import fold_frames


def panel():
    dates = pd.bdate_range("2016-01-01", "2020-12-31")
    return pd.DataFrame({
        "date": dates, "ticker": "AAA",
        "label_end_5": pd.Series(dates).shift(-6).values,
    })


FOLD = {"train": ["2016-01-01", "2018-12-31"], "val": ["2019-01-01", "2019-12-31"]}


def test_training_labels_finish_before_validation_starts():
    train, _ = fold_frames(panel(), FOLD, label_end="label_end_5")
    assert train.label_end_5.max() < pd.Timestamp("2019-01-01")


def test_validation_scoring_rows_stay_inside_the_year():
    _, val = fold_frames(panel(), FOLD, label_end="label_end_5")
    assert val.date.min() >= pd.Timestamp("2019-01-01")
    assert val.label_end_5.max() <= pd.Timestamp("2019-12-31")


def test_decisions_without_a_finished_label_are_not_trained_on():
    p = panel()
    p.loc[p.date >= "2018-06-01", "label_end_5"] = pd.NaT
    train, _ = fold_frames(p, FOLD, label_end="label_end_5")
    assert train.date.max() < pd.Timestamp("2018-06-01")
