"""Duplicate row detection and removal."""
from __future__ import annotations

from typing import List, Optional, Union

import pandas as pd


def get_duplicate_rows(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return all rows that belong to a duplicate group (both occurrences shown).
    Results are sorted so duplicates appear adjacent.
    """
    mask = df.duplicated(subset=subset, keep=False)
    sort_cols = subset if subset else df.columns.tolist()
    return df[mask].sort_values(by=sort_cols)


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: Union[str, bool] = "first",
) -> pd.DataFrame:
    """
    Drop duplicate rows.

    Args:
        subset: Columns to consider for duplication; None = all columns.
        keep:   "first", "last", or False (drop all occurrences).
    """
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
