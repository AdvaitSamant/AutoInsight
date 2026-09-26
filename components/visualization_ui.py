"""Visualization Studio — interactive chart builder.

Architecture:
  render()               — two-column layout: left = controls, right = chart
  _render_controls()     — per-chart-type widget tree; returns config dict
  _render_chart()        — dispatches to Plotly or Seaborn rendering
  _build_plotly_figure() — maps (chart_type, config) → go.Figure
  _render_seaborn_chart()— renders matplotlib figure via st.pyplot()
  _render_export_row()   — download buttons for HTML, PNG, SVG

Charts that return Plotly figures are handled by visualizations/plotly_charts.py.
Charts that need Seaborn are handled by visualizations/seaborn_charts.py.
Neither module calls any Streamlit function — that is done here.
"""
from __future__ import annotations

import io
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import visualizations.plotly_charts as pc
import visualizations.seaborn_charts as sc
from config.settings import AGGREGATION_FUNCTIONS, CHART_CATEGORIES, COLOR_SEQUENCE

# Chart types rendered with Plotly
_PLOTLY_CHARTS = {
    "Histogram", "KDE Plot", "Distribution Plot",
    "Box Plot", "Violin Plot", "Strip Plot",
    "Bar Chart", "Horizontal Bar", "Count Plot", "Grouped Bar",
    "Scatter Plot", "Bubble Chart",
    "Line Chart", "Area Chart",
    "Pie Chart", "Treemap", "Sunburst",
    "Correlation Heatmap", "Heatmap",
    "3D Scatter Plot",
}

# Chart types rendered with Seaborn / Matplotlib
_SEABORN_CHARTS = {
    "Pair Plot", "Joint Plot", "Regression Plot", "Swarm Plot", "Hexbin Plot",
}


# ── Entry point ────────────────────────────────────────────────────────────────

def render() -> None:
    df: pd.DataFrame = st.session_state.df
    st.header("Visualization Studio")

    col_types = {
        "numeric": df.select_dtypes(include="number").columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "all": df.columns.tolist(),
    }

    ctrl_col, chart_col = st.columns([1, 3], gap="large")

    with ctrl_col:
        category = st.selectbox("Category", list(CHART_CATEGORIES.keys()), key="viz_cat")
        chart_type = st.selectbox(
            "Chart Type", CHART_CATEGORIES[category], key="viz_type"
        )
        st.divider()
        config = _render_controls(df, chart_type, col_types)

    with chart_col:
        _render_chart(df, chart_type, config)


# ── Configuration widgets ──────────────────────────────────────────────────────

def _render_controls(df: pd.DataFrame, chart_type: str, col_types: dict) -> dict:
    """
    Render axis-selector and option widgets appropriate for the selected chart type.
    Returns a plain dict of configuration values consumed by _render_chart().
    """
    num = col_types["numeric"]
    cat = col_types["categorical"]
    all_cols = col_types["all"]
    config: dict = {}

    # --- Single-column distribution charts ---
    if chart_type in ("Histogram", "KDE Plot", "Distribution Plot"):
        config["col"] = st.selectbox("Column", num or all_cols, key="viz_col")
        if chart_type == "Histogram":
            config["bins"] = st.slider("Bins", 5, 100, 30, key="viz_bins")
            config["color_col"] = _opt_select("Color by", cat, "viz_color")
        return config

    if chart_type == "Count Plot":
        config["col"] = st.selectbox("Column", cat or all_cols, key="viz_col")
        return config

    # --- Box / Violin / Strip ---
    if chart_type in ("Box Plot", "Violin Plot", "Strip Plot"):
        config["y_col"] = st.selectbox("Y axis (numeric)", num or all_cols, key="viz_y")
        config["x_col"] = _opt_select("X axis (category)", cat, "viz_x")
        config["color_col"] = _opt_select("Color by", cat, "viz_color")
        return config

    # --- Trend ---
    if chart_type in ("Line Chart", "Area Chart"):
        config["x"] = st.selectbox("X axis", all_cols, key="viz_x")
        config["y"] = st.selectbox("Y axis (numeric)", num or all_cols, key="viz_y")
        config["color_col"] = _opt_select("Color by", cat, "viz_color")
        return config

    # --- Bar / Horizontal Bar ---
    if chart_type in ("Bar Chart", "Horizontal Bar"):
        config["x"] = st.selectbox("X axis (category)", cat or all_cols, key="viz_x")
        config["y"] = st.selectbox("Y axis (numeric)", num or all_cols, key="viz_y")
        config["agg_func"] = st.selectbox("Aggregation", list(AGGREGATION_FUNCTIONS.keys()), key="viz_agg")
        config["color_col"] = _opt_select("Color by", cat, "viz_color")
        return config

    if chart_type == "Grouped Bar":
        config["x"] = st.selectbox("X axis", cat or all_cols, key="viz_x")
        config["y"] = st.selectbox("Y axis (numeric)", num or all_cols, key="viz_y")
        config["group"] = st.selectbox("Group by", cat or all_cols, key="viz_grp")
        config["agg_func"] = st.selectbox("Aggregation", list(AGGREGATION_FUNCTIONS.keys()), key="viz_agg")
        return config

    # --- Scatter / Bubble ---
    if chart_type == "Scatter Plot":
        config["x"] = st.selectbox("X axis", num or all_cols, key="viz_x")
        config["y"] = st.selectbox("Y axis", num or all_cols, key="viz_y")
        config["color_col"] = _opt_select("Color by", cat + num, "viz_color")
        config["size_col"] = _opt_select("Size by", num, "viz_size")
        return config

    if chart_type == "Bubble Chart":
        config["x"] = st.selectbox("X axis", num or all_cols, key="viz_x")
        config["y"] = st.selectbox("Y axis", num or all_cols, key="viz_y")
        config["size"] = st.selectbox("Size", num or all_cols, key="viz_size")
        config["color_col"] = _opt_select("Color by", cat, "viz_color")
        return config

    # --- Composition ---
    if chart_type == "Pie Chart":
        config["names_col"] = st.selectbox("Labels", cat or all_cols, key="viz_names")
        config["values_col"] = _opt_select("Values (numeric)", num, "viz_values")
        return config

    if chart_type in ("Treemap", "Sunburst"):
        config["path"] = st.multiselect(
            "Hierarchy path (order matters)",
            cat or all_cols,
            default=(cat or all_cols)[:min(2, len(cat or all_cols))],
            key="viz_path",
        )
        config["values"] = _opt_select("Values (numeric)", num, "viz_values")
        return config

    # --- Matrix ---
    if chart_type == "Correlation Heatmap":
        config["method"] = st.selectbox("Method", ["pearson", "spearman", "kendall"], key="viz_method")
        return config

    if chart_type == "Heatmap":
        config["x"] = st.selectbox("X axis", cat or all_cols, key="viz_x")
        config["y"] = st.selectbox("Y axis", cat or all_cols, key="viz_y")
        config["values"] = st.selectbox("Values", num or all_cols, key="viz_values")
        config["agg_func"] = st.selectbox("Aggregation", list(AGGREGATION_FUNCTIONS.keys()), key="viz_agg")
        return config

    # --- 3D ---
    if chart_type == "3D Scatter Plot":
        config["x"] = st.selectbox("X axis", num or all_cols, key="viz_x")
        config["y"] = st.selectbox("Y axis", num or all_cols, key="viz_y")
        config["z"] = st.selectbox("Z axis", num or all_cols, key="viz_z")
        config["color_col"] = _opt_select("Color by", cat, "viz_color")
        return config

    # --- Seaborn charts ---
    if chart_type == "Pair Plot":
        config["columns"] = st.multiselect(
            "Columns",
            num,
            default=num[:min(4, len(num))],
            key="viz_cols",
        )
        config["hue"] = _opt_select("Color by", cat, "viz_hue")
        return config

    if chart_type == "Joint Plot":
        config["x"] = st.selectbox("X axis", num or all_cols, key="viz_x")
        config["y"] = st.selectbox("Y axis", num or all_cols, key="viz_y")
        config["kind"] = st.selectbox("Kind", ["scatter", "kde", "hex"], key="viz_kind")
        return config

    if chart_type in ("Regression Plot", "Hexbin Plot"):
        config["x"] = st.selectbox("X axis", num or all_cols, key="viz_x")
        config["y"] = st.selectbox("Y axis", num or all_cols, key="viz_y")
        return config

    if chart_type == "Swarm Plot":
        config["x"] = st.selectbox("X axis (category)", cat or all_cols, key="viz_x")
        config["y"] = st.selectbox("Y axis (numeric)", num or all_cols, key="viz_y")
        config["hue"] = _opt_select("Color by", cat, "viz_hue")
        return config

    return config


# ── Chart rendering ────────────────────────────────────────────────────────────

def _render_chart(df: pd.DataFrame, chart_type: str, config: dict) -> None:
    """Build and display the chart; show export buttons below."""
    try:
        if chart_type in _SEABORN_CHARTS:
            _render_seaborn_chart(df, chart_type, config)
        elif chart_type in _PLOTLY_CHARTS:
            fig = _build_plotly_figure(df, chart_type, config)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                _render_export_row(fig, chart_type)
        else:
            st.info("Select a chart type to get started.")
    except Exception as exc:
        st.error(f"Could not render chart: {exc}")


def _build_plotly_figure(
    df: pd.DataFrame, chart_type: str, config: dict
) -> Optional[go.Figure]:
    agg = AGGREGATION_FUNCTIONS.get(config.get("agg_func", "Mean"), "mean")

    if chart_type == "Histogram":
        return pc.histogram(df, config["col"], config.get("bins", 30), config.get("color_col"))
    if chart_type == "KDE Plot":
        return pc.kde_plot(df, config["col"])
    if chart_type == "Distribution Plot":
        return pc.distribution_plot(df, config["col"])
    if chart_type == "Count Plot":
        return pc.count_plot(df, config["col"])
    if chart_type == "Box Plot":
        return pc.box_plot(df, config["y_col"], config.get("x_col"), config.get("color_col"))
    if chart_type == "Violin Plot":
        return pc.violin_plot(df, config["y_col"], config.get("x_col"), config.get("color_col"))
    if chart_type == "Strip Plot":
        return pc.strip_plot(df, config["y_col"], config.get("x_col"), config.get("color_col"))
    if chart_type == "Line Chart":
        return pc.line_chart(df, config["x"], config["y"], config.get("color_col"))
    if chart_type == "Area Chart":
        return pc.area_chart(df, config["x"], config["y"], config.get("color_col"))
    if chart_type == "Bar Chart":
        return pc.bar_chart(df, config["x"], config["y"], agg, config.get("color_col"))
    if chart_type == "Horizontal Bar":
        return pc.horizontal_bar(df, config["x"], config["y"], agg, config.get("color_col"))
    if chart_type == "Grouped Bar":
        return pc.grouped_bar(df, config["x"], config["y"], config["group"], agg)
    if chart_type == "Scatter Plot":
        return pc.scatter_plot(df, config["x"], config["y"], config.get("color_col"), config.get("size_col"))
    if chart_type == "Bubble Chart":
        return pc.bubble_chart(df, config["x"], config["y"], config["size"], config.get("color_col"))
    if chart_type == "Pie Chart":
        return pc.pie_chart(df, config["names_col"], config.get("values_col"))
    if chart_type == "Treemap":
        path = config.get("path") or []
        if not path:
            raise ValueError("Select at least one level for the hierarchy path.")
        return pc.treemap(df, path, config.get("values"))
    if chart_type == "Sunburst":
        path = config.get("path") or []
        if not path:
            raise ValueError("Select at least one level for the hierarchy path.")
        return pc.sunburst(df, path, config.get("values"))
    if chart_type == "Correlation Heatmap":
        return pc.correlation_heatmap(df, config.get("method", "pearson"))
    if chart_type == "Heatmap":
        return pc.heatmap(df, config["x"], config["y"], config["values"], agg)
    if chart_type == "3D Scatter Plot":
        return pc.scatter_3d(df, config["x"], config["y"], config["z"], config.get("color_col"))
    return None


def _render_seaborn_chart(
    df: pd.DataFrame, chart_type: str, config: dict
) -> None:
    if chart_type == "Pair Plot":
        cols = config.get("columns")
        if not cols:
            st.warning("Select at least one column for the pair plot.")
            return
        fig = sc.pair_plot(df, cols, config.get("hue"))
    elif chart_type == "Joint Plot":
        fig = sc.joint_plot(df, config["x"], config["y"], config.get("kind", "scatter"))
    elif chart_type == "Regression Plot":
        fig = sc.regression_plot(df, config["x"], config["y"])
    elif chart_type == "Swarm Plot":
        fig = sc.swarm_plot(df, config["x"], config["y"], config.get("hue"))
    elif chart_type == "Hexbin Plot":
        fig = sc.hexbin_plot(df, config["x"], config["y"])
    else:
        st.info("Chart not implemented.")
        return

    st.pyplot(fig)
    plt.close(fig)

    # PNG export for matplotlib figures
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        "Download PNG",
        data=buf.getvalue(),
        file_name=f"{chart_type.lower().replace(' ', '_')}.png",
        mime="image/png",
    )


def _render_export_row(fig: go.Figure, chart_type: str) -> None:
    """Render export buttons below a Plotly chart."""
    name = chart_type.lower().replace(" ", "_")
    c1, c2, c3 = st.columns(3)

    # HTML export (always available)
    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
    c1.download_button(
        "Download HTML",
        data=html_bytes,
        file_name=f"{name}.html",
        mime="text/html",
        use_container_width=True,
    )

    # PNG export (requires kaleido)
    try:
        img_bytes = fig.to_image(format="png", scale=2)
        c2.download_button(
            "Download PNG",
            data=img_bytes,
            file_name=f"{name}.png",
            mime="image/png",
            use_container_width=True,
        )
    except Exception:
        c2.caption("PNG export requires `kaleido`.")

    # SVG export
    try:
        svg_bytes = fig.to_image(format="svg")
        c3.download_button(
            "Download SVG",
            data=svg_bytes,
            file_name=f"{name}.svg",
            mime="image/svg+xml",
            use_container_width=True,
        )
    except Exception:
        pass  # SVG export silently fails if kaleido is unavailable


# ── Helpers ────────────────────────────────────────────────────────────────────

def _opt_select(label: str, options: list, key: str) -> Optional[str]:
    """Selectbox that includes a 'None' option at the top."""
    if not options:
        return None
    choices = ["None"] + list(dict.fromkeys(options))  # deduplicate while preserving order
    selected = st.selectbox(label, choices, key=key)
    return None if selected == "None" else selected
