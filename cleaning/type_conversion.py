"""Safe data type conversion with pre-flight validation.

validate_conversion() checks feasibility before any data is changed.
convert_column() performs the actual cast after the user has confirmed.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def validate_conversion(series: pd.Series, target: str) -> Tuple[bool, str]:
    """
    Determine whether a Series can be safely cast to the target dtype.

    Returns:
        (True, "")              — conversion is safe
        (False, reason_string)  — conversion would fail or lose data
    """
    non_null = series.dropna()
    if non_null.empty:
        return True, ""

    try:
        if target == "int64":
            converted = pd.to_numeric(non_null, errors="coerce")
            bad = int(converted.isna().sum())
            if bad:
                return False, f"{bad} value(s) cannot be parsed as integers."
            if (converted != converted.round()).any():
                return False, "Column contains fractional values; converting would truncate them."
        elif target == "float64":
            pd.to_numeric(non_null, errors="raise")
        elif target == "datetime64":
            pd.to_datetime(non_null, format="mixed", errors="raise")
        elif target == "bool":
            valid = {"true", "false", "1", "0", "yes", "no"}
            bad_vals = [
                str(v) for v in non_null
                if str(v).lower() not in valid
            ]
            if bad_vals:
                preview = bad_vals[:5]
                return False, f"Non-boolean values present: {preview}"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def convert_column(df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    """
    Cast *column* to *target* dtype and return a new DataFrame.

    Supported targets: "int64", "float64", "str", "bool", "datetime64", "category"
    """
    df = df.copy()

    if target == "int64":
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    elif target == "float64":
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")

    elif target == "str":
        # Preserve pandas NA for null cells rather than the string "nan"
        df[column] = df[column].where(df[column].isna(), df[column].astype(str))

    elif target == "bool":
        mapping = {
            "true": True, "1": True, "yes": True,
            "false": False, "0": False, "no": False,
        }
        lower = df[column].astype(str).str.lower()
        df[column] = lower.map(mapping)
        df[column] = df[column].astype("boolean")

    elif target == "datetime64":
        df[column] = pd.to_datetime(
            df[column], format="mixed", errors="coerce"
        )

    elif target == "category":
        df[column] = df[column].astype("category")

    else:
        raise ValueError(f"Unsupported target dtype: '{target}'")

    return df
