"""Dataset Overview page — column info, missing values, statistics, duplicates."""
from __future__ import annotations

import plotly.express as px
import streamlit as st
import pandas as pd

from utils.dataframe_utils import (
    get_column_info,
    get_column_types,
    get_duplicate_count,
    get_missing_info,
)
from config.settings import COLOR_SEQUENCE, PREVIEW_DEFAULT_ROWS


def render() -> None:
    df: pd.DataFrame = st.session_state.df
    st.header("Dataset Overview")

    tabs = st.tabs(["Preview", "Column Info", "Missing Values", "Statistics", "Duplicates"])

    with tabs[0]:
        _tab_preview(df)
    with tabs[1]:
        _tab_column_info(df)
    with tabs[2]:
        _tab_missing(df)
    with tabs[3]:
        _tab_statistics(df)
    with tabs[4]:
        _tab_duplicates(df)


# ── Tab implementations ────────────────────────────────────────────────────────

def _tab_preview(df: pd.DataFrame) -> None:
    view = st.radio("Show", ["Head", "Tail", "Random Sample"], horizontal=True)
    n = st.slider("Rows", min_value=5, max_value=min(500, len(df)), value=PREVIEW_DEFAULT_ROWS)

    if view == "Head":
        display = df.head(n)
    elif view == "Tail":
        display = df.tail(n)
    else:
        display = df.sample(min(n, len(df)), random_state=42)

    st.dataframe(display, use_container_width=True)
    st.caption(f"Displaying {len(display):,} of {len(df):,} rows")


def _tab_column_info(df: pd.DataFrame) -> None:
    col_types = get_column_types(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Numeric", len(col_types["numeric"]))
    c2.metric("Categorical", len(col_types["categorical"]))
    c3.metric("Datetime", len(col_types["datetime"]))
    c4.metric("Boolean", len(col_types["boolean"]))

    st.write("")
    st.dataframe(get_column_info(df), use_container_width=True)


def _tab_missing(df: pd.DataFrame) -> None:
    missing_info = get_missing_info(df)

    if missing_info.empty:
        st.success("No missing values found in the dataset.")
        return

    total_missing = int(df.isnull().sum().sum())
    total_cells = df.size
    st.metric(
        "Total Missing Cells",
        f"{total_missing:,}",
        delta=f"{total_missing / total_cells:.2%} of all cells",
        delta_color="off",
    )

    st.dataframe(missing_info, use_container_width=True)

    fig = px.bar(
        missing_info.reset_index().rename(columns={"index": "Column"}),
        x="Column",
        y="Missing %",
        color="Missing %",
        color_continuous_scale="Oranges",
        title="Missing Value Rate by Column (%)",
        height=360,
        template="plotly_white",
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=48, b=40))
    st.plotly_chart(fig, use_container_width=True)


def _tab_statistics(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(include=["object", "category"])

    if not numeric.empty:
        st.subheader("Numeric Columns")
        st.dataframe(numeric.describe().T.round(4), use_container_width=True)

    if not categorical.empty:
        st.subheader("Categorical Columns")
        st.dataframe(categorical.describe(include="all").T, use_container_width=True)

    if numeric.empty and categorical.empty:
        st.info("No columns available for statistics.")


def _tab_duplicates(df: pd.DataFrame) -> None:
    n_dups = get_duplicate_count(df)
    pct = n_dups / len(df) * 100 if len(df) > 0 else 0.0

    st.metric("Duplicate Rows", f"{n_dups:,}", delta=f"{pct:.2f}% of dataset", delta_color="off")

    if n_dups == 0:
        st.success("No duplicate rows found.")
        return

    with st.expander(f"View duplicate rows ({n_dups:,} total)"):
        dup_df = df[df.duplicated(keep=False)]
        st.dataframe(dup_df.head(200), use_container_width=True)
        st.caption(f"Showing up to 200 of {n_dups:,} rows")
