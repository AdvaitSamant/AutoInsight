"""Application-wide constants and configuration values.

All magic strings, numeric limits, and user-facing labels that appear in more
than one module should be defined here to keep things consistent and easy to
update.
"""

# ── App identity ───────────────────────────────────────────────────────────────

APP_NAME = "AutoInsight"
VERSION = "1.0.0"

# ── Runtime limits ─────────────────────────────────────────────────────────────

MAX_UNDO_STACK = 20          # Maximum snapshots kept for undo
MAX_FILE_SIZE_MB = 200       # Uploaded file size ceiling (enforced by Streamlit config)
PREVIEW_DEFAULT_ROWS = 20    # Default number of rows in preview tables

# ── Navigation ─────────────────────────────────────────────────────────────────

PAGES = [
    "Dashboard",
    "Dataset Overview",
    "Data Cleaning",
    "Exploratory Analysis",
    "Visualization Studio",
    "Export",
]

# ── Chart configuration ────────────────────────────────────────────────────────

PLOTLY_THEME = "plotly_white"
CHART_HEIGHT = 520

# Accessible, professional color sequence used across all charts
COLOR_SEQUENCE = [
    "#3B6FD4", "#F5822A", "#E84040", "#2AAF8E",
    "#8F55E0", "#E8B700", "#D44D9F", "#50A8D4",
    "#7B8FA8", "#A8895A",
]

# Chart type categories shown in the Visualization Studio
CHART_CATEGORIES: dict[str, list[str]] = {
    "Distribution": [
        "Histogram",
        "Box Plot",
        "Violin Plot",
        "Strip Plot",
        "KDE Plot",
        "Distribution Plot",
    ],
    "Comparison": [
        "Bar Chart",
        "Horizontal Bar",
        "Count Plot",
        "Grouped Bar",
    ],
    "Relationship": [
        "Scatter Plot",
        "Bubble Chart",
        "Regression Plot",
        "Hexbin Plot",
        "Joint Plot",
    ],
    "Trend": [
        "Line Chart",
        "Area Chart",
    ],
    "Composition": [
        "Pie Chart",
        "Treemap",
        "Sunburst",
    ],
    "Matrix": [
        "Correlation Heatmap",
        "Heatmap",
        "Pair Plot",
    ],
    "3D": [
        "3D Scatter Plot",
    ],
}

# ── Data operation options ─────────────────────────────────────────────────────

# Aggregation labels → pandas method names
AGGREGATION_FUNCTIONS: dict[str, str] = {
    "Count": "count",
    "Sum": "sum",
    "Mean": "mean",
    "Median": "median",
    "Min": "min",
    "Max": "max",
    "Std Dev": "std",
}

# Type conversion target labels → internal dtype strings
DTYPE_TARGETS: dict[str, str] = {
    "Integer (int64)": "int64",
    "Float (float64)": "float64",
    "String (object)": "str",
    "Boolean": "bool",
    "Datetime": "datetime64",
    "Category": "category",
}

# Missing-value handling strategies
MISSING_STRATEGIES: list[str] = [
    "Drop Rows",
    "Drop Column",
    "Fill with Mean",
    "Fill with Median",
    "Fill with Mode",
    "Fill with Custom Value",
    "Forward Fill",
    "Backward Fill",
]

# Numeric transformation methods
SCALING_METHODS: list[str] = [
    "Min-Max Scaling",
    "Standard Scaling",
    "Robust Scaling",
    "Normalization (L2)",
    "Log Transform",
    "Winsorization",
]

# Outlier removal methods
OUTLIER_METHODS: list[str] = [
    "IQR Method",
    "Z-Score Method",
]

# Encoding strategies for categorical columns
ENCODING_METHODS: list[str] = [
    "Label Encoding",
    "One-Hot Encoding",
    "Ordinal Encoding",
]

# Datetime features available for extraction
DATETIME_FEATURES: list[str] = [
    "Year",
    "Month",
    "Week",
    "Day",
    "Weekday",
    "Quarter",
    "Hour",
    "Minute",
]
