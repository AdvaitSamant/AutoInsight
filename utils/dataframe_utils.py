"""DataFrame inspection and profiling utilities.

Pure functions — no Streamlit calls, no side effects.  All functions accept
a DataFrame and return derived data structures; they never mutate the input.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd


# ── Column classification ──────────────────────────────────────────────────────

def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group column names by their broad dtype category."""
    return {
        "numeric": df.select_dtypes(include="number").columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime": df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist(),
        "boolean": df.select_dtypes(include="bool").columns.tolist(),
    }


# ── Missing values ─────────────────────────────────────────────────────────────

def get_missing_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with per-column missing counts and percentages,
    restricted to columns that actually have missing values, sorted descending.
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    result = pd.DataFrame({"Missing Count": missing, "Missing %": pct})
    return result[result["Missing Count"] > 0].sort_values("Missing Count", ascending=False)


# ── Duplicates ─────────────────────────────────────────────────────────────────

def get_duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


# ── Memory ─────────────────────────────────────────────────────────────────────

def get_memory_usage(df: pd.DataFrame) -> str:
    """Return DataFrame memory footprint as a human-readable string."""
    total = float(df.memory_usage(deep=True).sum())
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024.0:
            return f"{total:.1f} {unit}"
        total /= 1024.0
    return f"{total:.1f} TB"


# ── Column info table ──────────────────────────────────────────────────────────

def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-column summary table containing dtype, null counts,
    unique value count, and a sample value.
    """
    rows = []
    for col in df.columns:
        non_null = df[col].dropna()
        rows.append({
            "Column": col,
            "Dtype": str(df[col].dtype),
            "Non-Null": int(df[col].notna().sum()),
            "Missing": int(df[col].isna().sum()),
            "Missing %": round(df[col].isna().mean() * 100, 2),
            "Unique": int(df[col].nunique()),
            "Sample": str(non_null.iloc[0]) if len(non_null) > 0 else "",
        })
    return pd.DataFrame(rows).set_index("Column")


# ── Datetime inference ─────────────────────────────────────────────────────────

def infer_potential_datetimes(df: pd.DataFrame) -> List[str]:
    """
    Heuristically identify object columns that likely contain date strings.
    A column is a candidate if ≥80% of its non-null sample parses as a date.
    """
    candidates: List[str] = []
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(50)
        if sample.empty:
            continue
        try:
            parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
            if parsed.notna().mean() >= 0.8:
                candidates.append(col)
        except Exception:
            pass
    return candidates
