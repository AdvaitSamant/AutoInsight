"""Plotly chart factory functions.

Every public function:
  - Accepts a DataFrame plus chart-specific parameters.
  - Returns a plotly.graph_objects.Figure.
  - Never calls st.plotly_chart() — that is the caller's responsibility.
  - Applies a consistent base layout via _apply_layout().

This separation keeps chart functions testable and reusable outside
the Streamlit context.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from config.settings import CHART_HEIGHT, COLOR_SEQUENCE, PLOTLY_THEME


# ── Layout helper ──────────────────────────────────────────────────────────────

def _apply_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply a consistent, clean layout to any Plotly figure."""
    fig.update_layout(
        template=PLOTLY_THEME,
        height=CHART_HEIGHT,
        title=dict(text=title, font=dict(size=15)) if title else None,
        margin=dict(l=48, r=32, t=56 if title else 32, b=48),
        colorway=COLOR_SEQUENCE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Distribution charts ────────────────────────────────────────────────────────

def histogram(
    df: pd.DataFrame,
    col: str,
    bins: int = 30,
    color_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    fig = px.histogram(
        df,
        x=col,
        nbins=bins,
        color=color_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.85,
    )
    return _apply_layout(fig, title or f"Histogram — {col}")


def kde_plot(df: pd.DataFrame, col: str, title: str = "") -> go.Figure:
    data = df[col].dropna().tolist()
    if not data:
        raise ValueError(f"Column '{col}' has no non-null values.")
    fig = ff.create_distplot(
        [data],
        [col],
        show_hist=False,
        show_rug=False,
        colors=COLOR_SEQUENCE[:1],
    )
    fig.update_layout(showlegend=False)
    return _apply_layout(fig, title or f"KDE — {col}")


def distribution_plot(df: pd.DataFrame, col: str, title: str = "") -> go.Figure:
    """Combined histogram + KDE overlay."""
    data = df[col].dropna().tolist()
    if not data:
        raise ValueError(f"Column '{col}' has no non-null values.")
    fig = ff.create_distplot(
        [data],
        [col],
        show_rug=False,
        colors=COLOR_SEQUENCE[:1],
    )
    return _apply_layout(fig, title or f"Distribution — {col}")


def box_plot(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    color_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    fig = px.box(
        df,
        y=y_col,
        x=x_col,
        color=color_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        points="outliers",
    )
    return _apply_layout(fig, title or f"Box Plot — {y_col}")


def violin_plot(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    color_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    fig = px.violin(
        df,
        y=y_col,
        x=x_col,
        color=color_col,
        box=True,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _apply_layout(fig, title or f"Violin Plot — {y_col}")


def strip_plot(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    color_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    fig = px.strip(
        df,
        y=y_col,
        x=x_col,
        color=color_col,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _apply_layout(fig, title or f"Strip Plot — {y_col}")


# ── Trend charts ───────────────────────────────────────────────────────────────

def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    fig = px.line(
        df,
        x=x,
        y=y,
        color=color_col,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _apply_layout(fig, title or f"{y} over {x}")


def area_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    fig = px.area(
        df,
        x=x,
        y=y,
        color=color_col,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _apply_layout(fig, title or f"{y} over {x}")


# ── Comparison charts ──────────────────────────────────────────────────────────

def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    agg_func: str = "mean",
    color_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    agg_df = df.groupby(x, observed=True)[y].agg(agg_func).reset_index()
    fig = px.bar(
        agg_df,
        x=x,
        y=y,
        color=color_col if color_col and color_col in agg_df.columns else None,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _apply_layout(fig, title or f"{y} by {x} ({agg_func})")


def horizontal_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    agg_func: str = "mean",
    color_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    agg_df = df.groupby(y, observed=True)[x].agg(agg_func).reset_index()
    fig = px.bar(
        agg_df,
        x=x,
        y=y,
        orientation="h",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _apply_layout(fig, title or f"{x} by {y} ({agg_func})")


def count_plot(df: pd.DataFrame, col: str, title: str = "") -> go.Figure:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Count"]
    fig = px.bar(
        counts,
        x=col,
        y="Count",
        color=col,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(showlegend=False)
    return _apply_layout(fig, title or f"Count — {col}")


def grouped_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    agg_func: str = "mean",
    title: str = "",
) -> go.Figure:
    agg_df = df.groupby([x, group], observed=True)[y].agg(agg_func).reset_index()
    fig = px.bar(
        agg_df,
        x=x,
        y=y,
        color=group,
        barmode="group",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _apply_layout(fig, title or f"{y} by {x} grouped by {group}")


# ── Relationship charts ────────────────────────────────────────────────────────

def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color_col,
        size=size_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.70,
    )
    return _apply_layout(fig, title or f"{y} vs {x}")


def bubble_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: str,
    color_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size,
        color=color_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.70,
    )
    return _apply_layout(fig, title or f"Bubble: {y} vs {x} (size: {size})")


# ── Composition charts ─────────────────────────────────────────────────────────

def pie_chart(
    df: pd.DataFrame,
    names_col: str,
    values_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    if values_col:
        agg = df.groupby(names_col, observed=True)[values_col].sum().reset_index()
        fig = px.pie(
            agg,
            names=names_col,
            values=values_col,
            color_discrete_sequence=COLOR_SEQUENCE,
        )
    else:
        counts = df[names_col].value_counts().reset_index()
        counts.columns = [names_col, "Count"]
        fig = px.pie(
            counts,
            names=names_col,
            values="Count",
            color_discrete_sequence=COLOR_SEQUENCE,
        )
    return _apply_layout(fig, title or f"Distribution of {names_col}")


def treemap(
    df: pd.DataFrame,
    path: List[str],
    values: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    fig = px.treemap(
        df,
        path=path,
        values=values,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _apply_layout(fig, title or "Treemap")


def sunburst(
    df: pd.DataFrame,
    path: List[str],
    values: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    fig = px.sunburst(
        df,
        path=path,
        values=values,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return _apply_layout(fig, title or "Sunburst")


# ── Matrix charts ──────────────────────────────────────────────────────────────

def correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    title: str = "",
) -> go.Figure:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("At least 2 numeric columns are required for a correlation heatmap.")
    corr = numeric.corr(method=method)
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    return _apply_layout(fig, title or f"Correlation Heatmap ({method.title()})")


def heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    values: str,
    agg_func: str = "mean",
    title: str = "",
) -> go.Figure:
    pivot = df.pivot_table(index=y, columns=x, values=values, aggfunc=agg_func)
    fig = px.imshow(
        pivot,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
    )
    return _apply_layout(fig, title or f"Heatmap: {values} by {x} × {y}")


# ── 3D charts ──────────────────────────────────────────────────────────────────

def scatter_3d(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    color_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color=color_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.70,
    )
    return _apply_layout(fig, title or f"3D Scatter: {x} × {y} × {z}")
