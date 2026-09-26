"""Exploratory Data Analysis page — statistics, correlation, and insights."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.insights import generate_insights, Insight
from analysis.statistics import compute_correlation, compute_descriptive_stats
from config.settings import COLOR_SEQUENCE


def render() -> None:
    df: pd.DataFrame = st.session_state.df
    st.header("Exploratory Analysis")

    tabs = st.tabs(["Descriptive Statistics", "Correlation", "Automated Insights"])

    with tabs[0]:
        _tab_statistics(df)
    with tabs[1]:
        _tab_correlation(df)
    with tabs[2]:
        _tab_insights(df)


# ── Tab: Descriptive Statistics ────────────────────────────────────────────────

def _tab_statistics(df: pd.DataFrame) -> None:
    stats = compute_descriptive_stats(df)

    if stats.empty:
        st.info("No numeric columns available for descriptive statistics.")
        return

    st.subheader("Numeric Summary")
    st.dataframe(stats, use_container_width=True)

    st.subheader("Distribution Explorer")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    col = st.selectbox("Column", numeric_cols, key="eda_stat_col")

    if col:
        left, right = st.columns(2)
        with left:
            fig = px.histogram(
                df,
                x=col,
                nbins=40,
                color_discrete_sequence=COLOR_SEQUENCE[:1],
                title=f"Histogram — {col}",
                height=360,
                template="plotly_white",
            )
            fig.update_layout(showlegend=False, margin=dict(t=48, b=40))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig2 = px.box(
                df,
                y=col,
                color_discrete_sequence=COLOR_SEQUENCE[:1],
                title=f"Box Plot — {col}",
                height=360,
                template="plotly_white",
                points="outliers",
            )
            fig2.update_layout(showlegend=False, margin=dict(t=48, b=40))
            st.plotly_chart(fig2, use_container_width=True)


# ── Tab: Correlation ───────────────────────────────────────────────────────────

def _tab_correlation(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        st.info("At least 2 numeric columns are needed for correlation analysis.")
        return

    method = st.selectbox("Method", ["pearson", "spearman", "kendall"], key="corr_method")
    corr = compute_correlation(df, method)

    st.subheader("Correlation Matrix")
    st.dataframe(corr.round(3), use_container_width=True)

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
        title=f"{method.title()} Correlation Heatmap",
        height=560,
        template="plotly_white",
    )
    fig.update_layout(margin=dict(t=56, b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Top correlated pairs table
    st.subheader("Top Correlated Pairs")
    upper_mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    upper = corr.where(pd.DataFrame(upper_mask, index=corr.index, columns=corr.columns))

    pairs = []
    for r in upper.columns:
        for c in upper.index:
            v = upper.loc[c, r]
            if pd.notna(v):
                pairs.append({
                    "Column A": c,
                    "Column B": r,
                    "Correlation": round(float(v), 4),
                    "|r|": round(abs(float(v)), 4),
                })

    if pairs:
        pairs_df = (
            pd.DataFrame(pairs)
            .sort_values("|r|", ascending=False)
            .drop(columns="|r|")
            .head(20)
        )
        st.dataframe(pairs_df, use_container_width=True, hide_index=True)


# ── Tab: Automated Insights ────────────────────────────────────────────────────

def _tab_insights(df: pd.DataFrame) -> None:
    st.subheader("Automated Insights")
    st.caption("Analysis of dataset quality, structure, and potential issues.")

    insights = generate_insights(df)

    errors = [i for i in insights if i.severity == "error"]
    warnings = [i for i in insights if i.severity == "warning"]
    infos = [i for i in insights if i.severity == "info"]

    for i in errors:
        st.error(f"**{i.title}** — {i.description}")
    for i in warnings:
        st.warning(f"**{i.title}** — {i.description}")
    for i in infos:
        st.info(f"**{i.title}** — {i.description}")
