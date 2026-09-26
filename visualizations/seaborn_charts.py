"""Seaborn / Matplotlib chart functions.

These cover chart types not well-supported in Plotly (pair plots, joint plots,
swarm plots, regression plots, hexbin plots).

Every function returns a matplotlib.figure.Figure so callers can render it
with st.pyplot(fig) and close it afterwards with plt.close(fig).

The non-interactive 'Agg' backend is set at import time to prevent
'cannot show a figure on a headless display' errors in Streamlit.
"""
from __future__ import annotations

from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


_STYLE = "whitegrid"
_PALETTE = "muted"
_FIG_SIZE = (10, 6)


def _setup() -> None:
    """Apply consistent seaborn style before each chart."""
    sns.set_style(_STYLE)
    sns.set_palette(_PALETTE)
    plt.rcParams["figure.dpi"] = 100


# ── Chart functions ────────────────────────────────────────────────────────────

def pair_plot(
    df: pd.DataFrame,
    columns: List[str],
    hue: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Seaborn pairplot for the specified numeric columns.
    If hue is provided it must be present in *df* but need not be in *columns*.
    """
    _setup()
    plot_cols = list(dict.fromkeys(columns + ([hue] if hue and hue not in columns else [])))
    g = sns.pairplot(
        df[plot_cols],
        hue=hue,
        diag_kind="kde",
        plot_kws={"alpha": 0.6},
    )
    return g.figure


def joint_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    kind: str = "scatter",
) -> matplotlib.figure.Figure:
    """Seaborn joint plot with marginal distributions."""
    _setup()
    g = sns.jointplot(
        data=df,
        x=x,
        y=y,
        kind=kind,
        color=sns.color_palette(_PALETTE)[0],
        marginal_kws={"fill": True},
    )
    return g.figure


def regression_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
) -> matplotlib.figure.Figure:
    """Scatter plot with an OLS regression line and 95% confidence band."""
    _setup()
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    sns.regplot(
        data=df,
        x=x,
        y=y,
        ax=ax,
        scatter_kws={"alpha": 0.5, "s": 20},
        line_kws={"linewidth": 1.8},
    )
    ax.set_title(f"Regression: {y} ~ {x}", fontsize=13)
    fig.tight_layout()
    return fig


def swarm_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Seaborn swarm plot.
    Automatically samples to ≤500 points to keep rendering fast.
    """
    _setup()
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    plot_df = df.sample(min(len(df), 500), random_state=42)
    sns.swarmplot(data=plot_df, x=x, y=y, hue=hue, ax=ax, size=3.5, dodge=True)
    ax.set_title(f"Swarm Plot — {y} by {x}", fontsize=13)
    fig.tight_layout()
    return fig


def hexbin_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
) -> matplotlib.figure.Figure:
    """Matplotlib hexbin density plot."""
    _setup()
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    valid = df[[x, y]].dropna()
    hb = ax.hexbin(valid[x], valid[y], gridsize=35, cmap="Blues", mincnt=1)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Hexbin — {x} × {y}", fontsize=13)
    fig.colorbar(hb, ax=ax, label="Count")
    fig.tight_layout()
    return fig
