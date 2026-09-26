"""Column-level transformations: structural and text operations.

All functions are pure — they accept a DataFrame and return a new DataFrame.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def _is_string_col(series: pd.Series) -> bool:
    """Return True for string-like columns (object, category, or pandas StringDtype)."""
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or str(series.dtype) in ("string", "str")
    )


# ── Structural operations ──────────────────────────────────────────────────────

def rename_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=mapping)


def delete_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df.drop(columns=columns)


def reorder_columns(df: pd.DataFrame, new_order: List[str]) -> pd.DataFrame:
    """Place columns in new_order first; any unmentioned columns follow."""
    extras = [c for c in df.columns if c not in new_order]
    return df[new_order + extras]


def merge_columns(
    df: pd.DataFrame,
    columns: List[str],
    new_name: str,
    separator: str = " ",
    drop_originals: bool = True,
) -> pd.DataFrame:
    """Concatenate multiple columns into one string column."""
    df = df.copy()
    df[new_name] = df[columns].astype(str).agg(separator.join, axis=1)
    if drop_originals:
        cols_to_drop = [c for c in columns if c != new_name]
        df = df.drop(columns=cols_to_drop)
    return df


def split_column(
    df: pd.DataFrame,
    column: str,
    delimiter: str,
    new_prefix: str,
    max_splits: int = -1,
    drop_original: bool = True,
) -> pd.DataFrame:
    """Split a string column on a delimiter into multiple new columns."""
    df = df.copy()
    n = max_splits if max_splits > 0 else -1
    split_df = df[column].astype(str).str.split(delimiter, n=n, expand=True)
    for i in range(split_df.shape[1]):
        df[f"{new_prefix}_{i + 1}"] = split_df[i]
    if drop_original:
        df = df.drop(columns=[column])
    return df


def copy_column(df: pd.DataFrame, column: str, new_name: str) -> pd.DataFrame:
    df = df.copy()
    df[new_name] = df[column]
    return df


# ── Text transformations ───────────────────────────────────────────────────────

def strip_whitespace(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if _is_string_col(df[col]):
            df[col] = df[col].str.strip()
    return df


def to_uppercase(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if _is_string_col(df[col]):
            df[col] = df[col].str.upper()
    return df


def to_lowercase(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if _is_string_col(df[col]):
            df[col] = df[col].str.lower()
    return df


def to_titlecase(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if _is_string_col(df[col]):
            df[col] = df[col].str.title()
    return df


def find_replace(
    df: pd.DataFrame,
    column: str,
    find: str,
    replace: str,
    case_sensitive: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].astype(str).str.replace(
        find, replace, case=case_sensitive, regex=False
    )
    return df


def regex_replace(
    df: pd.DataFrame,
    column: str,
    pattern: str,
    replacement: str,
) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].astype(str).str.replace(pattern, replacement, regex=True)
    return df


def extract_substring(
    df: pd.DataFrame,
    column: str,
    new_name: str,
    start: int,
    end: Optional[int] = None,
) -> pd.DataFrame:
    df = df.copy()
    df[new_name] = df[column].astype(str).str[start:end]
    return df


# ── Sorting / filtering ────────────────────────────────────────────────────────

def sort_dataframe(
    df: pd.DataFrame,
    column: str,
    ascending: bool = True,
) -> pd.DataFrame:
    return df.sort_values(column, ascending=ascending).reset_index(drop=True)


def filter_rows(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: str,
) -> pd.DataFrame:
    """
    Apply a single filter condition.

    Numeric operators: ==, !=, >, <, >=, <=
    String operators:  ==, !=, contains, not contains, starts with, ends with,
                       is null, not null
    """
    col_series = df[column]

    if pd.api.types.is_numeric_dtype(col_series):
        try:
            num_val = float(value)
        except ValueError:
            raise ValueError(f"'{value}' is not a valid number for column '{column}'.")
        ops = {
            "==": col_series == num_val,
            "!=": col_series != num_val,
            ">": col_series > num_val,
            "<": col_series < num_val,
            ">=": col_series >= num_val,
            "<=": col_series <= num_val,
        }
        mask = ops.get(operator)
        if mask is None:
            raise ValueError(f"Unsupported operator '{operator}' for numeric column.")
        return df[mask].reset_index(drop=True)

    str_series = col_series.astype(str)
    ops = {
        "==": str_series == value,
        "!=": str_series != value,
        "contains": str_series.str.contains(value, na=False, case=False),
        "not contains": ~str_series.str.contains(value, na=False, case=False),
        "starts with": str_series.str.startswith(value, na=False),
        "ends with": str_series.str.endswith(value, na=False),
        "is null": col_series.isnull(),
        "not null": col_series.notnull(),
    }
    mask = ops.get(operator)
    if mask is None:
        raise ValueError(f"Unsupported operator '{operator}'.")
    return df[mask].reset_index(drop=True)
