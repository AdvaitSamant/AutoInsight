"""Descriptive statistics and correlation analysis.

Functions are decorated with @st.cache_data so Streamlit caches results
per unique DataFrame hash, avoiding redundant computation on re-renders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as scipy_stats


@st.cache_data
def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute extended descriptive statistics for all numeric columns.

    Returns a DataFrame indexed by column name with columns:
    Count, Mean, Median, Std Dev, Variance, Min, 25%, 75%, Max,
    Skewness, Kurtosis.
    """
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()

    records: dict = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if s.empty:
            continue
        records[col] = {
            "Count": int(len(s)),
            "Mean": s.mean(),
            "Median": s.median(),
            "Std Dev": s.std(),
            "Variance": s.var(),
            "Min": s.min(),
            "25%": s.quantile(0.25),
            "75%": s.quantile(0.75),
            "Max": s.max(),
            "Skewness": float(scipy_stats.skew(s)),
            "Kurtosis": float(scipy_stats.kurtosis(s)),
        }

    return pd.DataFrame(records).T.round(4)


@st.cache_data
def compute_correlation(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Compute pairwise correlation for all numeric columns.

    Args:
        method: "pearson", "spearman", or "kendall".

    Returns an empty DataFrame if fewer than 2 numeric columns exist.
    """
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr(method=method)
