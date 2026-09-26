"""Datetime parsing and feature extraction."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import pandas as pd


# Maps feature name → extractor lambda that operates on a DatetimeIndex accessor
_FEATURE_EXTRACTORS: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "Year":    lambda s: s.dt.year,
    "Month":   lambda s: s.dt.month,
    "Week":    lambda s: s.dt.isocalendar().week.astype("Int64"),
    "Day":     lambda s: s.dt.day,
    "Weekday": lambda s: s.dt.weekday,
    "Quarter": lambda s: s.dt.quarter,
    "Hour":    lambda s: s.dt.hour,
    "Minute":  lambda s: s.dt.minute,
}


def parse_datetime(
    df: pd.DataFrame,
    column: str,
    fmt: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse an object column as datetime in-place.

    Args:
        fmt: strftime format string (e.g. "%Y-%m-%d").  If None, pandas infers.
    """
    df = df.copy()
    df[column] = pd.to_datetime(
        df[column],
        format=fmt if fmt else "mixed",
        errors="coerce",
    )
    return df


def extract_features(
    df: pd.DataFrame,
    column: str,
    features: List[str],
) -> pd.DataFrame:
    """
    Extract datetime sub-components from a datetime column.
    If the column is not yet datetime, it is parsed first.

    New columns are named <column>_<feature_lowercase>.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(
            df[column], format="mixed", errors="coerce"
        )

    for feat in features:
        extractor = _FEATURE_EXTRACTORS.get(feat)
        if extractor is None:
            continue
        df[f"{column}_{feat.lower()}"] = extractor(df[column])

    return df
