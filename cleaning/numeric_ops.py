"""Numeric transformations: scaling, normalization, and outlier removal.

All functions are pure — they return a new DataFrame and never mutate the input.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)


# ── Scaling / normalization ────────────────────────────────────────────────────

def min_max_scale(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Scale each column to the [0, 1] range."""
    df = df.copy()
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns].astype(float))
    return df


def standard_scale(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Standardize columns to zero mean and unit variance (z-score)."""
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns].astype(float))
    return df


def robust_scale(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Scale using median and IQR — more resistant to outliers than standard scaling."""
    df = df.copy()
    scaler = RobustScaler()
    df[columns] = scaler.fit_transform(df[columns].astype(float))
    return df


def l2_normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Normalize rows so that each selected column set has unit L2 norm."""
    df = df.copy()
    normalizer = Normalizer(norm="l2")
    df[columns] = normalizer.fit_transform(df[columns].astype(float))
    return df


def log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply log1p (log(1 + x)) to each column.
    Raises ValueError if any column contains negative values.
    """
    df = df.copy()
    for col in columns:
        if (df[col].dropna() < 0).any():
            raise ValueError(
                f"Column '{col}' contains negative values. "
                "Log transform requires non-negative data."
            )
        df[col] = np.log1p(df[col])
    return df


def winsorize(
    df: pd.DataFrame,
    columns: List[str],
    lower: float = 0.05,
    upper: float = 0.05,
) -> pd.DataFrame:
    """
    Clip values at the specified lower and upper percentiles.

    Args:
        lower: Fraction of data to clip at the bottom (e.g. 0.05 = 5th percentile).
        upper: Fraction of data to clip at the top (e.g. 0.05 = 95th percentile).
    """
    df = df.copy()
    for col in columns:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(1.0 - upper)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ── Outlier removal ────────────────────────────────────────────────────────────

def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Remove rows where any selected column falls outside
    [Q1 − multiplier×IQR, Q3 + multiplier×IQR].
    """
    df = df.copy()
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[
            (df[col] >= q1 - multiplier * iqr)
            & (df[col] <= q3 + multiplier * iqr)
        ]
    return df.reset_index(drop=True)


def remove_outliers_zscore(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Remove rows where |z-score| ≥ threshold for any of the selected columns."""
    df = df.copy()
    for col in columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        z = np.abs(scipy_stats.zscore(non_null))
        valid_idx = non_null.index
        keep_mask = pd.Series(True, index=df.index)
        keep_mask[valid_idx] = z < threshold
        df = df[keep_mask]
    return df.reset_index(drop=True)
