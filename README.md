# AutoInsight

**Data Analysis & Cleaning Studio** — a production-quality Streamlit application for end-to-end dataset inspection, cleaning, and visualization.

---

## Features

| Category | Capabilities |
|---|---|
| **Upload** | CSV, TSV, Excel (.xlsx); automatic encoding detection |
| **Inspect** | Row/column counts, dtypes, missing values, duplicates, memory usage |
| **Clean** | Missing value imputation, duplicate removal, column ops, type conversion, scaling, encoding, datetime extraction |
| **Analyze** | Descriptive stats, correlation matrix, automated quality insights |
| **Visualize** | 20+ chart types (Plotly + Seaborn); export as HTML, PNG, SVG |
| **Export** | Cleaned CSV, Excel, cleaning log, analysis summary |
| **Session** | Undo / Redo (20-step stack), reset to original |

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd AutoInsight

# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `kaleido` is required for PNG and SVG chart export. If you skip it, HTML export still works.

---

## Running the App

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501` by default.

---

## Project Structure

```
AutoInsight/
│
├── app.py                      # Entry point — page config, sidebar, routing
├── requirements.txt
├── README.md
│
├── .streamlit/
│   └── config.toml             # Theme and server settings
│
├── config/
│   └── settings.py             # All app-wide constants and labels
│
├── utils/
│   ├── session.py              # Session state init, undo/redo, apply_operation()
│   ├── io_utils.py             # File reading, export bytes, formatting
│   └── dataframe_utils.py      # Column classification, profiling, missing info
│
├── cleaning/
│   ├── missing_values.py       # Drop, fill, forward/backward fill
│   ├── duplicates.py           # Detection and removal
│   ├── column_ops.py           # Rename, merge, split, text ops, filter, sort
│   ├── type_conversion.py      # Safe dtype casting with validation
│   ├── numeric_ops.py          # Scaling, normalization, outlier removal
│   ├── encoding.py             # Label, one-hot, ordinal encoding
│   └── datetime_ops.py         # Datetime parsing and feature extraction
│
├── analysis/
│   ├── statistics.py           # Descriptive stats and correlation (cached)
│   └── insights.py             # Automated quality insights
│
├── visualizations/
│   ├── plotly_charts.py        # All Plotly chart factory functions
│   └── seaborn_charts.py       # Seaborn/Matplotlib chart functions
│
└── components/
    ├── dashboard_ui.py         # Dashboard: metrics + insights + preview
    ├── overview.py             # Dataset Overview: 5-tab inspection view
    ├── cleaning_ui.py          # Data Cleaning: 8-tab interactive operations
    ├── eda_ui.py               # EDA: stats, correlation, insights
    ├── visualization_ui.py     # Visualization Studio: interactive chart builder
    └── export_ui.py            # Export: CSV, Excel, log, summary
```

---

## Architecture Notes

- **Pure cleaning functions** — every function in `cleaning/` accepts a DataFrame and returns a new one. No Streamlit calls, no side effects.
- **Decoupled charts** — chart functions in `visualizations/` return `Figure` objects; the UI component decides how to render them.
- **Centralized state** — all DataFrame mutations go through `utils/session.apply_operation()`, which handles undo snapshots and log entries.
- **Cached analysis** — `analysis/statistics.py` uses `@st.cache_data`, keyed on the DataFrame hash, to avoid recomputing statistics on every render.

---

## Supported Chart Types

**Plotly (interactive):** Histogram, KDE Plot, Distribution Plot, Box Plot, Violin Plot, Strip Plot, Bar Chart, Horizontal Bar, Count Plot, Grouped Bar, Scatter Plot, Bubble Chart, Line Chart, Area Chart, Pie Chart, Treemap, Sunburst, Correlation Heatmap, Heatmap, 3D Scatter Plot

**Seaborn / Matplotlib:** Pair Plot, Joint Plot, Regression Plot, Swarm Plot, Hexbin Plot
