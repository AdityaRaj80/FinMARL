"""Fold selection with purging: a training label must finish before its validation year starts."""

import pandas as pd


def fold_frames(panel, fold, label_end):
    """Rows usable for fitting and for scoring in one fold (see the embargo note in configs/splits.yaml)."""
    tr_start, tr_end = map(pd.Timestamp, fold["train"])
    va_start, va_end = map(pd.Timestamp, fold["val"])
    end = panel[label_end]
    train = panel[panel.date.between(tr_start, tr_end) & end.notna() & (end < va_start)]
    val = panel[panel.date.between(va_start, va_end) & end.notna() & (end <= va_end)]
    return train, val
