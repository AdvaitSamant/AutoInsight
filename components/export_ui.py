"""Export page — download the cleaned dataset and supporting reports."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from utils.io_utils import (
    build_analysis_summary,
    format_cleaning_log,
    to_csv_bytes,
    to_excel_bytes,
)


def render() -> None:
    df: pd.DataFrame = st.session_state.df
    log: list[str] = st.session_state.cleaning_log
    file_stem = (
        st.session_state.file_name.rsplit(".", 1)[0]
        if st.session_state.file_name
        else "dataset"
    )

    st.header("Export")

    # ── Cleaned dataset ────────────────────────────────────────────────────────
    st.subheader("Cleaned Dataset")
    st.caption(f"{df.shape[0]:,} rows × {df.shape[1]:,} columns")

    c1, c2 = st.columns(2)
    c1.download_button(
        label="Download CSV",
        data=to_csv_bytes(df),
        file_name=f"{file_stem}_cleaned.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c2.download_button(
        label="Download Excel (.xlsx)",
        data=to_excel_bytes(df),
        file_name=f"{file_stem}_cleaned.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.divider()

    # ── Cleaning log ───────────────────────────────────────────────────────────
    st.subheader("Cleaning Log")
    if log:
        log_text = format_cleaning_log(log)
        st.code(log_text, language="text")
        st.download_button(
            label="Download Cleaning Log",
            data=log_text.encode("utf-8"),
            file_name=f"{file_stem}_cleaning_log.txt",
            mime="text/plain",
        )
    else:
        st.info("No cleaning operations have been applied yet.")

    st.divider()

    # ── Analysis summary ───────────────────────────────────────────────────────
    st.subheader("Analysis Summary")
    summary = build_analysis_summary(df, log)
    st.code(summary, language="text")
    st.download_button(
        label="Download Analysis Summary",
        data=summary.encode("utf-8"),
        file_name=f"{file_stem}_analysis_summary.txt",
        mime="text/plain",
    )
