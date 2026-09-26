"""Automated dataset quality analysis and insight generation.

generate_insights() produces a list of Insight objects that are rendered
in both the Dashboard and the Exploratory Analysis page.  Each insight has
a title, a plain-English description, and a severity level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Insight:
    title: str
    description: str
    severity: str  # "info" | "warning" | "error"


def generate_insights(df: pd.DataFrame) -> List[Insight]:
    """
    Analyse a DataFrame and return a list of quality/structure insights.
    The list is never empty — at minimum an "all clear" insight is returned.
    """
    insights: List[Insight] = []
    n_rows, _ = df.shape

    if n_rows == 0:
        insights.append(Insight("Empty Dataset", "The dataset contains no rows.", "error"))
        return insights

    _check_missing(df, n_rows, insights)
    _check_constants(df, insights)
    _check_high_cardinality(df, n_rows, insights)
    _check_duplicates(df, n_rows, insights)
    _check_correlations(df, insights)
    _check_low_variance(df, insights)
    _check_outliers(df, n_rows, insights)
    _check_potential_dates(df, insights)

    if not insights:
        insights.append(Insight(
            "Dataset Looks Clean",
            "No significant quality issues were detected.",
            "info",
        ))
    return insights


# ── Private checkers ───────────────────────────────────────────────────────────

def _check_missing(df: pd.DataFrame, n_rows: int, out: List[Insight]) -> None:
    pct = df.isnull().mean()
    high = pct[pct > 0.30]
    moderate = pct[(pct > 0.05) & (pct <= 0.30)]
    if not high.empty:
        cols = ", ".join(f"'{c}' ({v:.0%})" for c, v in high.items())
        out.append(Insight(
            "High Missing Rate",
            f"Columns with >30% missing values: {cols}.",
            "warning",
        ))
    if not moderate.empty:
        cols = ", ".join(f"'{c}'" for c in moderate.index[:5])
        suffix = f" (and {len(moderate) - 5} more)" if len(moderate) > 5 else ""
        out.append(Insight(
            "Moderate Missing Values",
            f"Columns with 5–30% missing: {cols}{suffix}.",
            "info",
        ))


def _check_constants(df: pd.DataFrame, out: List[Insight]) -> None:
    constants = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constants:
        out.append(Insight(
            "Constant Columns",
            (
                f"{len(constants)} column(s) have only one unique value and carry "
                f"no predictive signal: {', '.join(constants[:5])}."
            ),
            "warning",
        ))


def _check_high_cardinality(
    df: pd.DataFrame, n_rows: int, out: List[Insight]
) -> None:
    for col in df.select_dtypes(include=["object", "category"]).columns:
        ratio = df[col].nunique() / n_rows
        if ratio > 0.90 and n_rows > 50:
            out.append(Insight(
                f"High Cardinality: '{col}'",
                (
                    f"'{col}' has {df[col].nunique():,} unique values ({ratio:.0%} of rows). "
                    "This may be an ID or free-text column — unlikely to be useful as a category."
                ),
                "info",
            ))


def _check_duplicates(
    df: pd.DataFrame, n_rows: int, out: List[Insight]
) -> None:
    n_dups = int(df.duplicated().sum())
    if n_dups > 0:
        pct = n_dups / n_rows
        out.append(Insight(
            "Duplicate Rows",
            f"{n_dups:,} exact duplicate rows detected ({pct:.1%} of the dataset).",
            "warning" if pct > 0.05 else "info",
        ))


def _check_correlations(df: pd.DataFrame, out: List[Insight]) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return
    corr = numeric.corr().abs()
    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )
    pairs = [
        (c, r, upper.loc[r, c])
        for r in upper.columns
        for c in upper.index
        if pd.notna(upper.loc[r, c]) and upper.loc[r, c] > 0.90
    ]
    if pairs:
        pair_str = ", ".join(f"'{a}'/'{b}' ({v:.2f})" for a, b, v in pairs[:3])
        out.append(Insight(
            "Highly Correlated Features",
            f"Pairs with |r| > 0.90: {pair_str}. One member of each pair may be redundant.",
            "info",
        ))


def _check_low_variance(df: pd.DataFrame, out: List[Insight]) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return
    std = numeric.std()
    near_zero = std[std < 1e-6].index.tolist()
    if near_zero:
        out.append(Insight(
            "Near-Zero Variance",
            (
                f"Features with essentially zero standard deviation: "
                f"{', '.join(near_zero[:5])}. They will not contribute to most models."
            ),
            "info",
        ))


def _check_outliers(
    df: pd.DataFrame, n_rows: int, out: List[Insight]
) -> None:
    outlier_summaries = []
    for col in df.select_dtypes(include="number").columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = int(
            ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        )
        if n_out / n_rows > 0.05:
            outlier_summaries.append(f"'{col}' ({n_out:,})")
    if outlier_summaries:
        out.append(Insight(
            "Potential Outliers",
            (
                f"Columns where >5% of values fall outside 1.5×IQR: "
                f"{', '.join(outlier_summaries[:4])}."
            ),
            "warning",
        ))


def _check_potential_dates(df: pd.DataFrame, out: List[Insight]) -> None:
    from utils.dataframe_utils import infer_potential_datetimes
    candidates = infer_potential_datetimes(df)
    if candidates:
        out.append(Insight(
            "Possible Date Columns",
            (
                f"Object columns that appear to contain dates: "
                f"{', '.join(candidates)}. "
                "Consider converting them in Data Cleaning → Date Features."
            ),
            "info",
        ))
