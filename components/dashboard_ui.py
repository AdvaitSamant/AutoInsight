"""Dashboard page — high-level dataset metrics and automated insights."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from analysis.insights import generate_insights, Insight
from utils.dataframe_utils import get_column_types, get_duplicate_count, get_memory_usage
from utils.io_utils import format_bytes


def render() -> None:
    df: pd.DataFrame = st.session_state.df

    st.header("Dashboard")
    st.caption(st.session_state.file_name)

    _render_metrics(df)
    st.divider()
    _render_insights(df)
    st.divider()
    _render_preview(df)


def _render_metrics(df: pd.DataFrame) -> None:
    col_types = get_column_types(df)

    row1 = st.columns(4)
    row1[0].metric("Rows", f"{df.shape[0]:,}")
    row1[1].metric("Columns", f"{df.shape[1]:,}")
    row1[2].metric("Missing Values", f"{df.isnull().sum().sum():,}")
    row1[3].metric("Duplicate Rows", f"{get_duplicate_count(df):,}")

    row2 = st.columns(4)
    row2[0].metric("Numeric Features", len(col_types["numeric"]))
    row2[1].metric("Categorical Features", len(col_types["categorical"]))
    row2[2].metric("Memory Usage", get_memory_usage(df))
    row2[3].metric("File Size", format_bytes(st.session_state.file_size))


def _render_insights(df: pd.DataFrame) -> None:
    st.subheader("Automated Insights")
    for insight in generate_insights(df):
        _render_insight(insight)


def _render_insight(insight: Insight) -> None:
    msg = f"**{insight.title}** — {insight.description}"
    if insight.severity == "error":
        st.error(msg)
    elif insight.severity == "warning":
        st.warning(msg)
    else:
        st.info(msg)


def _render_preview(df: pd.DataFrame) -> None:
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
