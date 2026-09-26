"""File I/O utilities: reading uploaded files and producing export bytes.

Encoding detection uses chardet.  On failure, a prioritized list of common
encodings is tried in sequence so that most real-world files parse correctly.
"""
from __future__ import annotations

import io
from typing import Optional, Tuple

import chardet
import pandas as pd

# Fallback encoding candidates tried in order after chardet detection
_ENCODING_FALLBACKS = ["utf-8", "latin-1", "cp1252", "utf-16", "iso-8859-1"]


# ── File reading ───────────────────────────────────────────────────────────────

def _detect_encoding(raw: bytes) -> str:
    """Run chardet on a byte sample; fall back to utf-8 if detection is inconclusive."""
    result = chardet.detect(raw[:100_000])
    return result.get("encoding") or "utf-8"


def read_uploaded_file(
    uploaded_file,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parse a Streamlit UploadedFile into a DataFrame.

    Returns:
        (DataFrame, None)  on success
        (None, error_str)  on failure
    """
    name: str = uploaded_file.name.lower()
    try:
        raw: bytes = uploaded_file.read()
        uploaded_file.seek(0)

        if name.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
            return df, None

        sep = "\t" if name.endswith(".tsv") else ","
        detected = _detect_encoding(raw)

        for enc in [detected, *_ENCODING_FALLBACKS]:
            if not enc:
                continue
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    sep=sep,
                    encoding=enc,
                    low_memory=False,
                )
                return df, None
            except (UnicodeDecodeError, LookupError):
                continue

        return None, (
            "Could not decode the file with any supported encoding. "
            "Try saving it as UTF-8 and re-uploading."
        )

    except Exception as exc:
        return None, f"Failed to parse file: {exc}"


# ── Export helpers ─────────────────────────────────────────────────────────────

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to a UTF-8 CSV byte string."""
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to an Excel (.xlsx) byte string."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned Data")
    return buf.getvalue()


# ── Formatting ─────────────────────────────────────────────────────────────────

def format_cleaning_log(log: list[str]) -> str:
    """Render the cleaning log as a plain-text report."""
    if not log:
        return "No cleaning operations have been recorded."
    lines = ["AutoInsight — Cleaning Log", "=" * 44, ""]
    for i, entry in enumerate(log, 1):
        lines.append(f"  {i:>3}.  {entry}")
    return "\n".join(lines)


def format_bytes(size: int) -> str:
    """Return a human-readable string for a byte count (e.g. '3.2 MB')."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def build_analysis_summary(df: pd.DataFrame, log: list[str]) -> str:
    """Generate a plain-text analysis summary report suitable for download."""
    from utils.dataframe_utils import get_column_types, get_missing_info, get_duplicate_count

    col_types = get_column_types(df)
    missing = get_missing_info(df)
    dups = get_duplicate_count(df)

    lines = [
        "AutoInsight — Analysis Summary",
        "=" * 44,
        f"Rows:               {df.shape[0]:,}",
        f"Columns:            {df.shape[1]:,}",
        f"Numeric columns:    {len(col_types['numeric'])}",
        f"Categorical cols:   {len(col_types['categorical'])}",
        f"Datetime cols:      {len(col_types['datetime'])}",
        f"Boolean cols:       {len(col_types['boolean'])}",
        f"Duplicate rows:     {dups:,}",
        f"Total missing:      {df.isnull().sum().sum():,}",
        "",
        "Missing Values by Column:",
        "-" * 44,
    ]
    if missing.empty:
        lines.append("  None detected.")
    else:
        for col, row in missing.iterrows():
            lines.append(f"  {col}: {row['Missing Count']} ({row['Missing %']}%)")

    lines += ["", "Cleaning Operations Applied:", "-" * 44]
    if log:
        for i, entry in enumerate(log, 1):
            lines.append(f"  {i}. {entry}")
    else:
        lines.append("  None applied.")

    return "\n".join(lines)
