"""Session state initialization and lifecycle management.

All state mutations go through this module.  Components call these functions
rather than touching st.session_state keys directly, keeping state transitions
centralized and predictable.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import MAX_UNDO_STACK


# ── Initialization ─────────────────────────────────────────────────────────────

def init_session() -> None:
    """Set default values for every session key on first load."""
    defaults: dict = {
        "df": None,
        "original_df": None,
        "file_name": None,
        "file_size": 0,
        "undo_stack": [],   # list[dict] — each item: {"df": pd.DataFrame, "label": str}
        "redo_stack": [],
        "cleaning_log": [], # list[str] — human-readable operation descriptions
        "data_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_dataset(df: pd.DataFrame, file_name: str, file_size: int) -> None:
    """Load a new dataset and clear all derived state."""
    st.session_state.df = df.copy()
    st.session_state.original_df = df.copy()
    st.session_state.file_name = file_name
    st.session_state.file_size = file_size
    st.session_state.undo_stack = []
    st.session_state.redo_stack = []
    st.session_state.cleaning_log = []
    st.session_state.data_loaded = True


# ── Undo / Redo ────────────────────────────────────────────────────────────────

def push_undo(label: str) -> None:
    """Snapshot the current DataFrame before a mutation."""
    if st.session_state.df is None:
        return
    stack: list = st.session_state.undo_stack
    stack.append({"df": st.session_state.df.copy(), "label": label})
    if len(stack) > MAX_UNDO_STACK:
        stack.pop(0)
    st.session_state.undo_stack = stack
    st.session_state.redo_stack = []  # new operation clears redo history


def undo() -> bool:
    """Restore the previous DataFrame state. Returns True when successful."""
    if not st.session_state.undo_stack:
        return False
    st.session_state.redo_stack.append({
        "df": st.session_state.df.copy(),
        "label": "redo",
    })
    snapshot = st.session_state.undo_stack.pop()
    st.session_state.df = snapshot["df"]
    if st.session_state.cleaning_log:
        st.session_state.cleaning_log.pop()
    return True


def redo() -> bool:
    """Re-apply the most recently undone operation. Returns True when successful."""
    if not st.session_state.redo_stack:
        return False
    st.session_state.undo_stack.append({
        "df": st.session_state.df.copy(),
        "label": "undo",
    })
    snapshot = st.session_state.redo_stack.pop()
    st.session_state.df = snapshot["df"]
    return True


# ── Reset ──────────────────────────────────────────────────────────────────────

def reset_to_original() -> None:
    """Replace the working DataFrame with the original upload."""
    if st.session_state.original_df is None:
        return
    push_undo("Before full reset")
    st.session_state.df = st.session_state.original_df.copy()
    st.session_state.cleaning_log = []


# ── Apply operation ────────────────────────────────────────────────────────────

def apply_operation(new_df: pd.DataFrame, description: str) -> None:
    """
    Commit a cleaning operation: push undo snapshot, update state, append log.

    All cleaning components should call this function after producing a new
    DataFrame, rather than updating st.session_state.df directly.
    """
    push_undo(description)
    st.session_state.df = new_df
    st.session_state.cleaning_log.append(description)
