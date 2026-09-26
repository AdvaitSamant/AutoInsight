"""AutoInsight — Data Analysis & Cleaning Studio.

Entry point.  Responsibilities:
  - Page configuration and minimal CSS overrides
  - Session state initialization
  - Sidebar: file upload, navigation, undo/redo controls, session info
  - Page routing to the appropriate component
"""
from __future__ import annotations

import streamlit as st

from config.settings import APP_NAME, PAGES
from utils.io_utils import format_bytes, read_uploaded_file
from utils.session import init_session, load_dataset, redo, reset_to_original, undo

import components.cleaning_ui as cleaning
import components.dashboard_ui as dashboard
import components.eda_ui as eda
import components.export_ui as export_page
import components.overview as overview
import components.visualization_ui as viz


# ── Page configuration — must be the first Streamlit call ─────────────────────

st.set_page_config(
    page_title=APP_NAME,
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            f"**{APP_NAME}**  \n"
            "Data Analysis & Cleaning Studio  \n"
            "Upload a dataset to get started."
        )
    },
)

# ── Minimal CSS overrides ──────────────────────────────────────────────────────
# Keep overrides surgical — only fix things Streamlit's theme cannot handle.

st.markdown(
    """
    <style>
    /* Reduce default top padding so headers sit closer to the nav bar */
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2rem;
    }
    /* Slightly smaller metric labels to match the denser layout */
    [data-testid="stMetricLabel"] p {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.7;
    }
    /* Tighten sidebar section spacing */
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }
    /* Make the sidebar app title visually distinct */
    [data-testid="stSidebar"] h1 {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    /* Remove default anchor highlight on headings */
    h1 a, h2 a, h3 a { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session initialization ─────────────────────────────────────────────────────

init_session()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"# {APP_NAME}")
    st.caption("Data Analysis & Cleaning Studio")
    st.divider()

    # File upload
    uploaded = st.file_uploader(
        "Upload Dataset",
        type=["csv", "tsv", "xlsx"],
        help="Supports CSV, TSV, and Excel (.xlsx) files up to 200 MB.",
    )

    if uploaded is not None:
        # Only reload if it's a new file (avoids reloading on every interaction)
        is_new_file = (
            not st.session_state.data_loaded
            or uploaded.name != st.session_state.file_name
        )
        if is_new_file:
            with st.spinner("Parsing file…"):
                df_loaded, err = read_uploaded_file(uploaded)
            if err:
                st.error(err)
            else:
                load_dataset(df_loaded, uploaded.name, uploaded.size)
                st.success(
                    f"Loaded **{uploaded.name}** "
                    f"({format_bytes(uploaded.size)})."
                )

    # Navigation — only shown once a dataset is loaded
    page = PAGES[0]
    if st.session_state.data_loaded:
        st.divider()
        page = st.radio(
            "Navigation",
            PAGES,
            label_visibility="collapsed",
        )

        st.divider()

        # Undo / Redo
        undo_off = not bool(st.session_state.undo_stack)
        redo_off = not bool(st.session_state.redo_stack)
        uc, rc = st.columns(2)
        if uc.button("Undo", disabled=undo_off, use_container_width=True):
            if undo():
                st.rerun()
        if rc.button("Redo", disabled=redo_off, use_container_width=True):
            if redo():
                st.rerun()

        if st.button("Reset to Original", use_container_width=True):
            reset_to_original()
            st.rerun()

        # Session summary
        st.divider()
        df_cur = st.session_state.df
        n_ops = len(st.session_state.cleaning_log)
        st.caption(
            f"**{st.session_state.file_name}**  \n"
            f"{df_cur.shape[0]:,} rows × {df_cur.shape[1]:,} cols  \n"
            f"{n_ops} operation{'s' if n_ops != 1 else ''} applied"
        )

# ── Landing page ───────────────────────────────────────────────────────────────

def _render_landing() -> None:
    """Shown before any file is loaded."""
    st.markdown(f"# {APP_NAME}")
    st.markdown(
        "Upload a dataset using the sidebar to get started. "
        "Supports CSV, TSV, and Excel files."
    )

    st.write("")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Inspect")
        st.write(
            "Explore row counts, column types, missing values, "
            "duplicate rows, and summary statistics."
        )

    with c2:
        st.markdown("#### Clean")
        st.write(
            "Apply targeted transformations with full undo/redo support. "
            "Every operation is logged and exportable."
        )

    with c3:
        st.markdown("#### Visualize")
        st.write(
            "Build interactive charts across 20+ types. "
            "Export as PNG, SVG, or self-contained HTML."
        )


# ── Main content area ──────────────────────────────────────────────────────────

if not st.session_state.data_loaded:
    _render_landing()
else:
    _ROUTES = {
        "Dashboard":            dashboard.render,
        "Dataset Overview":     overview.render,
        "Data Cleaning":        cleaning.render,
        "Exploratory Analysis": eda.render,
        "Visualization Studio": viz.render,
        "Export":               export_page.render,
    }
    renderer = _ROUTES.get(page)
    if renderer:
        renderer()
