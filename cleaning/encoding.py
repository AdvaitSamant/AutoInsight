"""Categorical variable encoding strategies."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def label_encode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Encode each selected column with integer labels (0 … N-1).
    A fresh LabelEncoder is fitted per column.
    """
    df = df.copy()
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def one_hot_encode(
    df: pd.DataFrame,
    columns: List[str],
    drop_first: bool = False,
) -> pd.DataFrame:
    """
    Expand each selected column into binary indicator columns.
    New columns are appended as integers (0/1).
    """
    return pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype=int)


def ordinal_encode(
    df: pd.DataFrame,
    column: str,
    order: List[str],
) -> pd.DataFrame:
    """
    Map category values to integers according to a caller-defined order.
    Values not in *order* are mapped to NaN.

    Args:
        order: Category values in ascending ordinal order; first = 0.
    """
    df = df.copy()
    mapping: Dict[str, int] = {val: i for i, val in enumerate(order)}
    df[column] = df[column].astype(str).map(mapping)
    return df
