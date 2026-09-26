"""Missing value handling strategies.

Each function is pure: it accepts a DataFrame and returns a new DataFrame.
None of these functions mutate their input or touch Streamlit state.
"""
from __future__ import annotations

from typing import List, Union

import pandas as pd


def drop_rows(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drop any row that has a missing value in at least one of the given columns."""
    return df.dropna(subset=columns).reset_index(drop=True)


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Remove the specified columns entirely."""
    return df.drop(columns=columns)


def fill_mean(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Fill missing values with the column mean (numeric columns only)."""
    df = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
    return df


def fill_median(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Fill missing values with the column median (numeric columns only)."""
    df = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
    return df


def fill_mode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Fill missing values with the most frequent value in each column."""
    df = df.copy()
    for col in columns:
        mode_vals = df[col].mode()
        if not mode_vals.empty:
            df[col] = df[col].fillna(mode_vals.iloc[0])
    return df


def fill_constant(
    df: pd.DataFrame,
    columns: List[str],
    value: Union[str, float, int],
) -> pd.DataFrame:
    """Fill missing values with a user-supplied constant."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna(value)
    return df


def forward_fill(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Propagate the last valid observation forward (ffill)."""
    df = df.copy()
    df[columns] = df[columns].ffill()
    return df


def backward_fill(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Propagate the next valid observation backward (bfill)."""
    df = df.copy()
    df[columns] = df[columns].bfill()
    return df
