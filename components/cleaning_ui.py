"""Data Cleaning page — interactive cleaning operations with full undo support.

Structure:
  render() → eight tabs, one per operation category
  Each tab is implemented in a private function.
  All mutations go through utils.session.apply_operation().
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
import streamlit as st

import cleaning.column_ops as col_ops
import cleaning.datetime_ops as dt_ops
import cleaning.duplicates as dups
import cleaning.encoding as enc
import cleaning.missing_values as mv
import cleaning.numeric_ops as num_ops
import cleaning.type_conversion as type_conv
from config.settings import (
    DATETIME_FEATURES,
    DTYPE_TARGETS,
    ENCODING_METHODS,
    MISSING_STRATEGIES,
    OUTLIER_METHODS,
    SCALING_METHODS,
)
from utils.dataframe_utils import get_missing_info, infer_potential_datetimes
from utils.session import apply_operation


# ── Entry point ────────────────────────────────────────────────────────────────

def render() -> None:
    df: pd.DataFrame = st.session_state.df
    st.header("Data Cleaning")
    st.caption(f"{df.shape[0]:,} rows × {df.shape[1]:,} columns")

    tabs = st.tabs([
        "Missing Values",
        "Duplicates",
        "Column Operations",
        "Type Conversion",
        "Numeric Transforms",
        "Encoding",
        "Date Features",
        "Filter & Sort",
    ])

    with tabs[0]:
        _tab_missing(df)
    with tabs[1]:
        _tab_duplicates(df)
    with tabs[2]:
        _tab_column_ops(df)
    with tabs[3]:
        _tab_type_conversion(df)
    with tabs[4]:
        _tab_numeric(df)
    with tabs[5]:
        _tab_encoding(df)
    with tabs[6]:
        _tab_datetime(df)
    with tabs[7]:
        _tab_filter_sort(df)


# ── Tab: Missing Values ────────────────────────────────────────────────────────

def _tab_missing(df: pd.DataFrame) -> None:
    missing_info = get_missing_info(df)

    if missing_info.empty:
        st.success("No missing values detected.")
        return

    st.dataframe(missing_info, use_container_width=True)
    st.write("")

    cols_with_missing = missing_info.index.tolist()
    selected = st.multiselect(
        "Columns to process",
        cols_with_missing,
        default=cols_with_missing[:1],
    )
    strategy = st.selectbox("Strategy", MISSING_STRATEGIES)

    custom_val: Optional[str] = None
    if strategy == "Fill with Custom Value":
        custom_val = st.text_input("Custom fill value", placeholder="e.g. 0 or Unknown")

    if not selected:
        st.caption("Select at least one column.")
        return

    if st.button("Apply", key="btn_missing"):
        _apply_missing(df, selected, strategy, custom_val)


def _apply_missing(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str,
    custom_val: Optional[str],
) -> None:
    dispatch = {
        "Drop Rows":             lambda: mv.drop_rows(df, columns),
        "Drop Column":           lambda: mv.drop_columns(df, columns),
        "Fill with Mean":        lambda: mv.fill_mean(df, columns),
        "Fill with Median":      lambda: mv.fill_median(df, columns),
        "Fill with Mode":        lambda: mv.fill_mode(df, columns),
        "Fill with Custom Value": lambda: mv.fill_constant(df, columns, custom_val),
        "Forward Fill":          lambda: mv.forward_fill(df, columns),
        "Backward Fill":         lambda: mv.backward_fill(df, columns),
    }
    fn = dispatch.get(strategy)
    if fn is None:
        st.error(f"Unknown strategy: {strategy}")
        return
    try:
        new_df = fn()
        apply_operation(new_df, f"Missing values — {strategy} on {_col_list(columns)}")
        st.success("Operation applied.")
        st.rerun()
    except Exception as exc:
        st.error(f"Error: {exc}")


# ── Tab: Duplicates ────────────────────────────────────────────────────────────

def _tab_duplicates(df: pd.DataFrame) -> None:
    n_dups = int(df.duplicated().sum())
    st.metric("Duplicate Rows", f"{n_dups:,}")

    if n_dups == 0:
        st.success("No duplicate rows found.")
        return

    subset = st.multiselect(
        "Deduplicate on columns (leave empty to use all columns)",
        df.columns.tolist(),
    )
    keep = st.radio("Keep", ["first", "last", "none"], horizontal=True)
    keep_val = False if keep == "none" else keep

    with st.expander("Preview duplicate rows"):
        preview = dups.get_duplicate_rows(df, subset or None)
        st.dataframe(preview.head(100), use_container_width=True)
        st.caption(f"{len(preview):,} rows in {n_dups} duplicate groups")

    if st.button("Remove Duplicates", key="btn_dups"):
        try:
            new_df = dups.remove_duplicates(df, subset or None, keep=keep_val)
            removed = df.shape[0] - new_df.shape[0]
            apply_operation(new_df, f"Removed {removed:,} duplicate rows (keep={keep})")
            st.success(f"Removed {removed:,} rows.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


# ── Tab: Column Operations ─────────────────────────────────────────────────────

def _tab_column_ops(df: pd.DataFrame) -> None:
    operation = st.selectbox(
        "Operation",
        [
            "Rename Columns",
            "Delete Columns",
            "Reorder Columns",
            "Merge Columns",
            "Split Column",
            "Copy Column",
            "Strip Whitespace",
            "Change Case",
            "Find & Replace",
            "Regex Replace",
            "Extract Substring",
        ],
    )

    handlers = {
        "Rename Columns":   _op_rename,
        "Delete Columns":   _op_delete,
        "Reorder Columns":  _op_reorder,
        "Merge Columns":    _op_merge,
        "Split Column":     _op_split,
        "Copy Column":      _op_copy,
        "Strip Whitespace": _op_strip,
        "Change Case":      _op_case,
        "Find & Replace":   _op_find_replace,
        "Regex Replace":    _op_regex,
        "Extract Substring": _op_extract,
    }
    handler = handlers.get(operation)
    if handler:
        handler(df)


def _op_rename(df: pd.DataFrame) -> None:
    st.write("Enter a new name for each column you want to rename:")
    mapping: dict = {}
    for col in df.columns:
        new = st.text_input(col, value=col, key=f"ren_{col}")
        if new and new != col:
            mapping[col] = new
    if not mapping:
        st.caption("No changes specified.")
        return
    if st.button("Apply Rename", key="btn_rename"):
        try:
            new_df = col_ops.rename_columns(df, mapping)
            apply_operation(new_df, f"Renamed {len(mapping)} column(s)")
            st.success(f"Renamed {len(mapping)} column(s).")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


def _op_delete(df: pd.DataFrame) -> None:
    cols = st.multiselect("Columns to delete", df.columns.tolist())
    if not cols:
        return
    st.warning(f"This will permanently remove: {', '.join(cols)}")
    if st.button("Delete Columns", key="btn_delete"):
        try:
            new_df = col_ops.delete_columns(df, cols)
            apply_operation(new_df, f"Deleted columns: {_col_list(cols)}")
            st.success(f"Deleted {len(cols)} column(s).")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


def _op_reorder(df: pd.DataFrame) -> None:
    st.write("Select columns in the desired order:")
    ordered = st.multiselect(
        "Column order",
        df.columns.tolist(),
        default=df.columns.tolist(),
    )
    if not ordered:
        return
    if st.button("Apply Order", key="btn_reorder"):
        try:
            new_df = col_ops.reorder_columns(df, ordered)
            apply_operation(new_df, "Reordered columns")
            st.success("Columns reordered.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


def _op_merge(df: pd.DataFrame) -> None:
    cols = st.multiselect("Columns to merge", df.columns.tolist())
    new_name = st.text_input("New column name", value="merged_column")
    separator = st.text_input("Separator", value=" ")
    drop_originals = st.checkbox("Drop original columns", value=True)
    if len(cols) < 2:
        st.caption("Select at least 2 columns.")
        return
    if st.button("Merge Columns", key="btn_merge"):
        try:
            new_df = col_ops.merge_columns(df, cols, new_name, separator, drop_originals)
            apply_operation(new_df, f"Merged {_col_list(cols)} → '{new_name}'")
            st.success("Columns merged.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


def _op_split(df: pd.DataFrame) -> None:
    col = st.selectbox("Column to split", df.columns.tolist())
    delimiter = st.text_input("Delimiter", value=",")
    prefix = st.text_input("New column prefix", value=f"{col}_part")
    max_splits = st.number_input("Max splits (−1 = unlimited)", min_value=-1, value=-1)
    drop_original = st.checkbox("Drop original column", value=True)
    if st.button("Split Column", key="btn_split"):
        try:
            new_df = col_ops.split_column(df, col, delimiter, prefix, int(max_splits), drop_original)
            apply_operation(new_df, f"Split '{col}' on '{delimiter}'")
            st.success("Column split.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


def _op_copy(df: pd.DataFrame) -> None:
    col = st.selectbox("Column to copy", df.columns.tolist())
    new_name = st.text_input("New column name", value=f"{col}_copy")
    if st.button("Copy Column", key="btn_copy"):
        try:
            new_df = col_ops.copy_column(df, col, new_name)
            apply_operation(new_df, f"Copied '{col}' → '{new_name}'")
            st.success("Column copied.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


def _op_strip(df: pd.DataFrame) -> None:
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if not text_cols:
        st.info("No text columns available.")
        return
    cols = st.multiselect("Columns to strip", text_cols, default=text_cols)
    if not cols:
        return
    if st.button("Strip Whitespace", key="btn_strip"):
        try:
            new_df = col_ops.strip_whitespace(df, cols)
            apply_operation(new_df, f"Stripped whitespace from {_col_list(cols)}")
            st.success("Whitespace stripped.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


def _op_case(df: pd.DataFrame) -> None:
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if not text_cols:
        st.info("No text columns available.")
        return
    cols = st.multiselect("Columns", text_cols)
    case = st.radio("Transform to", ["Uppercase", "Lowercase", "Title Case"], horizontal=True)
    if not cols:
        return
    if st.button("Apply Case Transform", key="btn_case"):
        try:
            fn_map = {
                "Uppercase":  col_ops.to_uppercase,
                "Lowercase":  col_ops.to_lowercase,
                "Title Case": col_ops.to_titlecase,
            }
            new_df = fn_map[case](df, cols)
            apply_operation(new_df, f"{case} on {_col_list(cols)}")
            st.success("Case transform applied.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


def _op_find_replace(df: pd.DataFrame) -> None:
    col = st.selectbox("Column", df.columns.tolist(), key="fr_col")
    find = st.text_input("Find")
    replace = st.text_input("Replace with")
    case_sensitive = st.checkbox("Case sensitive", value=True, key="fr_case")
    if not find:
        st.caption("Enter a search string.")
        return
    if st.button("Apply", key="btn_fr"):
        try:
            new_df = col_ops.find_replace(df, col, find, replace, case_sensitive)
            apply_operation(new_df, f"Find & Replace in '{col}': '{find}' → '{replace}'")
            st.success("Replacement applied.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


def _op_regex(df: pd.DataFrame) -> None:
    col = st.selectbox("Column", df.columns.tolist(), key="rx_col")
    pattern = st.text_input("Regex pattern", placeholder=r"e.g. \d+")
    replacement = st.text_input("Replacement", placeholder="e.g. [NUM]")
    if not pattern:
        st.caption("Enter a regex pattern.")
        return
    if st.button("Apply Regex Replace", key="btn_regex"):
        try:
            new_df = col_ops.regex_replace(df, col, pattern, replacement)
            apply_operation(new_df, f"Regex replace in '{col}': r'{pattern}' → '{replacement}'")
            st.success("Regex replacement applied.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


def _op_extract(df: pd.DataFrame) -> None:
    col = st.selectbox("Source column", df.columns.tolist(), key="ex_col")
    new_name = st.text_input("New column name", value=f"{col}_extracted", key="ex_name")
    c1, c2 = st.columns(2)
    start = c1.number_input("Start index", min_value=0, value=0, step=1)
    end_val = c2.number_input("End index (−1 = to end)", min_value=-1, value=-1, step=1)
    end: Optional[int] = None if end_val == -1 else int(end_val)
    if st.button("Extract", key="btn_extract"):
        try:
            new_df = col_ops.extract_substring(df, col, new_name, int(start), end)
            range_str = f"[{int(start)}:{end}]"
            apply_operation(new_df, f"Extracted {range_str} from '{col}' → '{new_name}'")
            st.success("Substring extracted.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


# ── Tab: Type Conversion ───────────────────────────────────────────────────────

def _tab_type_conversion(df: pd.DataFrame) -> None:
    col = st.selectbox("Column to convert", df.columns.tolist())
    current_type = str(df[col].dtype)
    st.caption(f"Current dtype: `{current_type}`")

    target_label = st.selectbox("Convert to", list(DTYPE_TARGETS.keys()))
    target = DTYPE_TARGETS[target_label]

    c1, c2 = st.columns(2)
    if c1.button("Validate", key="btn_validate_type"):
        ok, reason = type_conv.validate_conversion(df[col], target)
        if ok:
            st.success("Conversion is safe to apply.")
        else:
            st.warning(f"Potential issue: {reason}")

    if c2.button("Apply Conversion", key="btn_apply_type"):
        try:
            new_df = type_conv.convert_column(df, col, target)
            apply_operation(new_df, f"Converted '{col}': {current_type} → {target_label}")
            st.success(f"'{col}' converted to {target_label}.")
            st.rerun()
        except Exception as exc:
            st.error(f"Conversion failed: {exc}")


# ── Tab: Numeric Transforms ────────────────────────────────────────────────────

def _tab_numeric(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns in the current dataset.")
        return

    all_ops = SCALING_METHODS + OUTLIER_METHODS
    operation = st.selectbox("Operation", all_ops)

    if operation in SCALING_METHODS:
        _numeric_scaling(df, numeric_cols, operation)
    else:
        _numeric_outliers(df, numeric_cols, operation)


def _numeric_scaling(df: pd.DataFrame, numeric_cols: List[str], operation: str) -> None:
    cols = st.multiselect("Columns", numeric_cols, default=numeric_cols[:1])

    extra: dict = {}
    if operation == "Winsorization":
        c1, c2 = st.columns(2)
        extra["lower"] = c1.slider("Lower clip percentile", 0.0, 0.25, 0.05, step=0.01, format="%.2f")
        extra["upper"] = c2.slider("Upper clip percentile", 0.0, 0.25, 0.05, step=0.01, format="%.2f")
    elif operation == "Log Transform":
        st.info("Applies log₁ₚ(x) = log(1 + x). All values must be ≥ 0.")

    if not cols:
        return

    if st.button("Apply Transform", key="btn_scale"):
        try:
            new_df = _dispatch_scaling(df, operation, cols, extra)
            apply_operation(new_df, f"{operation} on {_col_list(cols)}")
            st.success("Transform applied.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


def _dispatch_scaling(
    df: pd.DataFrame, operation: str, cols: List[str], extra: dict
) -> pd.DataFrame:
    dispatch = {
        "Min-Max Scaling":   lambda: num_ops.min_max_scale(df, cols),
        "Standard Scaling":  lambda: num_ops.standard_scale(df, cols),
        "Robust Scaling":    lambda: num_ops.robust_scale(df, cols),
        "Normalization (L2)": lambda: num_ops.l2_normalize(df, cols),
        "Log Transform":     lambda: num_ops.log_transform(df, cols),
        "Winsorization":     lambda: num_ops.winsorize(df, cols, extra.get("lower", 0.05), extra.get("upper", 0.05)),
    }
    fn = dispatch.get(operation)
    if fn is None:
        raise ValueError(f"Unknown operation: {operation}")
    return fn()


def _numeric_outliers(df: pd.DataFrame, numeric_cols: List[str], operation: str) -> None:
    cols = st.multiselect("Columns", numeric_cols, default=numeric_cols[:1])

    if operation == "IQR Method":
        multiplier = st.slider("IQR multiplier", 1.0, 4.0, 1.5, step=0.1)
    else:
        threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, step=0.1)

    if not cols:
        return

    if st.button("Remove Outliers", key="btn_outliers"):
        try:
            if operation == "IQR Method":
                new_df = num_ops.remove_outliers_iqr(df, cols, multiplier)
            else:
                new_df = num_ops.remove_outliers_zscore(df, cols, threshold)
            removed = df.shape[0] - new_df.shape[0]
            apply_operation(
                new_df,
                f"Outlier removal ({operation}) on {_col_list(cols)} — {removed:,} rows removed",
            )
            st.success(f"Removed {removed:,} rows.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


# ── Tab: Encoding ──────────────────────────────────────────────────────────────

def _tab_encoding(df: pd.DataFrame) -> None:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        st.info("No categorical columns in the current dataset.")
        return

    method = st.selectbox("Encoding method", ENCODING_METHODS)

    if method == "Label Encoding":
        _enc_label(df, cat_cols)
    elif method == "One-Hot Encoding":
        _enc_onehot(df, cat_cols)
    elif method == "Ordinal Encoding":
        _enc_ordinal(df, cat_cols)


def _enc_label(df: pd.DataFrame, cat_cols: List[str]) -> None:
    cols = st.multiselect("Columns to encode", cat_cols)
    if not cols:
        return
    st.caption("Values are mapped to integers (0 … N−1) alphabetically.")
    if st.button("Apply Label Encoding", key="btn_le"):
        try:
            new_df = enc.label_encode(df, cols)
            apply_operation(new_df, f"Label encoded: {_col_list(cols)}")
            st.success("Label encoding applied.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


def _enc_onehot(df: pd.DataFrame, cat_cols: List[str]) -> None:
    cols = st.multiselect("Columns to encode", cat_cols)
    drop_first = st.checkbox(
        "Drop first category (avoids multicollinearity)", value=False
    )
    if not cols:
        return
    n_new = sum(df[c].nunique() - (1 if drop_first else 0) for c in cols)
    st.caption(f"Will add approximately {n_new} new indicator columns.")
    if st.button("Apply One-Hot Encoding", key="btn_ohe"):
        try:
            new_df = enc.one_hot_encode(df, cols, drop_first=drop_first)
            apply_operation(new_df, f"One-hot encoded: {_col_list(cols)}")
            st.success("One-hot encoding applied.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


def _enc_ordinal(df: pd.DataFrame, cat_cols: List[str]) -> None:
    col = st.selectbox("Column to encode", cat_cols)
    uniques = sorted(df[col].dropna().unique().astype(str).tolist())
    st.write("Arrange categories in ascending ordinal order (first = 0):")
    ordered = st.multiselect("Category order", uniques, default=uniques)
    if len(ordered) < len(uniques):
        st.caption(f"{len(uniques) - len(ordered)} categories not mapped — they will become NaN.")
    if ordered and st.button("Apply Ordinal Encoding", key="btn_ord"):
        try:
            new_df = enc.ordinal_encode(df, col, ordered)
            apply_operation(new_df, f"Ordinal encoded '{col}'")
            st.success("Ordinal encoding applied.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


# ── Tab: Date Features ─────────────────────────────────────────────────────────

def _tab_datetime(df: pd.DataFrame) -> None:
    st.subheader("Parse Column as Datetime")
    parse_col = st.selectbox("Column", df.columns.tolist(), key="dt_parse_col")
    fmt = st.text_input(
        "Format string (optional)",
        placeholder="e.g. %Y-%m-%d or %d/%m/%Y",
        key="dt_fmt",
    )
    if st.button("Parse as Datetime", key="btn_dt_parse"):
        try:
            new_df = dt_ops.parse_datetime(df, parse_col, fmt or None)
            null_count = int(new_df[parse_col].isna().sum())
            apply_operation(new_df, f"Parsed '{parse_col}' as datetime")
            suffix = f" ({null_count} values could not be parsed)" if null_count else ""
            st.success(f"'{parse_col}' parsed as datetime{suffix}.")
            st.rerun()
        except Exception as exc:
            st.error(f"Parse failed: {exc}")

    st.divider()
    st.subheader("Extract Date Features")

    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    potential = infer_potential_datetimes(df)
    all_candidates = list(dict.fromkeys(dt_cols + potential))

    if not all_candidates:
        st.info("No datetime columns detected. Parse one above first.")
        return

    feat_col = st.selectbox("Datetime column", all_candidates, key="dt_feat_col")
    features = st.multiselect(
        "Features to extract",
        DATETIME_FEATURES,
        default=["Year", "Month", "Day"],
    )
    if features and st.button("Extract Features", key="btn_dt_extract"):
        try:
            new_df = dt_ops.extract_features(df, feat_col, features)
            apply_operation(new_df, f"Extracted {features} from '{feat_col}'")
            st.success(f"Extracted {len(features)} feature(s) from '{feat_col}'.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


# ── Tab: Filter & Sort ─────────────────────────────────────────────────────────

def _tab_filter_sort(df: pd.DataFrame) -> None:
    st.subheader("Filter Rows")
    st.caption(f"Current dataset: {df.shape[0]:,} rows")

    col = st.selectbox("Column", df.columns.tolist(), key="filt_col")
    col_is_numeric = pd.api.types.is_numeric_dtype(df[col])

    if col_is_numeric:
        operators = ["==", "!=", ">", "<", ">=", "<="]
    else:
        operators = [
            "==", "!=", "contains", "not contains",
            "starts with", "ends with", "is null", "not null",
        ]

    operator = st.selectbox("Operator", operators, key="filt_op")
    value = ""
    if operator not in ("is null", "not null"):
        if col_is_numeric:
            value = str(st.number_input("Value", key="filt_num_val"))
        else:
            value = st.text_input("Value", key="filt_str_val")

    c1, c2 = st.columns(2)
    if c1.button("Preview", key="btn_filt_preview"):
        try:
            preview = col_ops.filter_rows(df, col, operator, value)
            st.caption(f"Result: {preview.shape[0]:,} rows (removed {df.shape[0] - preview.shape[0]:,})")
            st.dataframe(preview.head(20), use_container_width=True)
        except Exception as exc:
            st.error(f"Error: {exc}")

    if c2.button("Apply Filter", key="btn_filt_apply"):
        try:
            new_df = col_ops.filter_rows(df, col, operator, value)
            removed = df.shape[0] - new_df.shape[0]
            apply_operation(new_df, f"Filter: '{col}' {operator} '{value}'")
            st.success(f"Filter applied. {removed:,} rows removed.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")

    st.divider()
    st.subheader("Sort Rows")
    sort_col = st.selectbox("Sort by", df.columns.tolist(), key="sort_col")
    asc = st.radio("Order", ["Ascending", "Descending"], horizontal=True) == "Ascending"

    if st.button("Apply Sort", key="btn_sort"):
        try:
            new_df = col_ops.sort_dataframe(df, sort_col, ascending=asc)
            order_str = "ascending" if asc else "descending"
            apply_operation(new_df, f"Sorted by '{sort_col}' ({order_str})")
            st.success(f"Dataset sorted by '{sort_col}'.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")

    st.divider()
    st.subheader("Select Columns")
    kept = st.multiselect(
        "Columns to keep",
        df.columns.tolist(),
        default=df.columns.tolist(),
        key="col_sel",
    )
    if kept and len(kept) < df.shape[1]:
        dropped = df.shape[1] - len(kept)
        st.caption(f"Will drop {dropped} column(s).")
        if st.button("Apply Column Selection", key="btn_col_sel"):
            try:
                new_df = df[kept].copy()
                apply_operation(new_df, f"Kept {len(kept)} of {df.shape[1]} columns")
                st.success(f"Dropped {dropped} column(s).")
                st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _col_list(cols: List[str]) -> str:
    """Compact column list string for use in log entries."""
    if len(cols) <= 3:
        return ", ".join(f"'{c}'" for c in cols)
    return f"'{cols[0]}', '{cols[1]}', … ({len(cols)} columns)"
