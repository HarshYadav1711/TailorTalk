"""Deterministic pandas tools used by the LangChain agent."""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

matplotlib.use("Agg")

try:
    from backend.data_loader import load_titanic_df
except ImportError:
    from data_loader import load_titanic_df


def _df() -> pd.DataFrame:
    return load_titanic_df()


def _validate_column(column: str) -> str:
    normalized = column.strip().lower()
    if normalized not in _df().columns:
        available = ", ".join(sorted(_df().columns))
        raise ValueError(f"Unknown column '{column}'. Available columns: {available}")
    return normalized


class PercentageInput(BaseModel):
    column: str = Field(description="Column name to evaluate")
    value: str = Field(description="Value to match in the column")


def calculate_percentage(column: str, value: str) -> dict[str, Any]:
    """Calculate percentage of rows where column equals value."""
    col = _validate_column(column)
    series = _df()[col]
    if series.empty:
        raise ValueError("Dataset is empty.")

    if pd.api.types.is_numeric_dtype(series):
        try:
            target: Any = float(value)
        except ValueError as exc:
            raise ValueError(f"Value '{value}' is not numeric for column '{column}'.") from exc
        matched = series.fillna(float("inf")) == target
    else:
        matched = series.astype(str).str.strip().str.lower() == value.strip().lower()

    percentage = (matched.sum() / len(series)) * 100
    return {
        "tool": "calculate_percentage",
        "column": col,
        "value": value,
        "matched_count": int(matched.sum()),
        "total_rows": int(len(series)),
        "percentage": round(float(percentage), 2),
    }


class AverageInput(BaseModel):
    column: str = Field(description="Numeric column name")


def average(column: str) -> dict[str, Any]:
    """Compute average for a numeric column."""
    col = _validate_column(column)
    series = _df()[col]
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column '{col}' is not numeric.")
    if series.dropna().empty:
        raise ValueError(f"Column '{col}' has no numeric values.")

    avg_value = series.mean(skipna=True)
    return {"tool": "average", "column": col, "average": round(float(avg_value), 2)}


class CountByInput(BaseModel):
    column: str = Field(description="Column to group by")


def count_by(column: str) -> dict[str, Any]:
    """Count rows by unique values for a column."""
    col = _validate_column(column)
    counts = _df()[col].fillna("missing").value_counts(dropna=False).to_dict()
    normalized_counts = {str(k): int(v) for k, v in counts.items()}
    return {"tool": "count_by", "column": col, "counts": normalized_counts}


class SummaryStatsInput(BaseModel):
    column: str = Field(description="Numeric column name")


def summary_stats(column: str) -> dict[str, Any]:
    """Return summary statistics for a numeric column."""
    col = _validate_column(column)
    series = _df()[col]
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column '{col}' is not numeric.")
    if series.dropna().empty:
        raise ValueError(f"Column '{col}' has no numeric values.")

    desc = series.describe()
    return {
        "tool": "summary_stats",
        "column": col,
        "count": int(desc["count"]),
        "mean": round(float(desc["mean"]), 2),
        "std": round(float(desc["std"]), 2),
        "min": round(float(desc["min"]), 2),
        "q1": round(float(desc["25%"]), 2),
        "median": round(float(desc["50%"]), 2),
        "q3": round(float(desc["75%"]), 2),
        "max": round(float(desc["max"]), 2),
    }


class HistogramInput(BaseModel):
    column: str = Field(description="Numeric column name")
    bins: int = Field(default=20, ge=5, le=100, description="Number of bins")


def histogram(column: str, bins: int = 20) -> dict[str, Any]:
    """Generate histogram PNG (base64) for a numeric column."""
    col = _validate_column(column)
    series = _df()[col].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column '{col}' is not numeric.")
    if series.empty:
        raise ValueError(f"Column '{col}' has no numeric values.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(series, bins=bins, color="#1f77b4", edgecolor="black")
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "tool": "histogram",
        "column": col,
        "bins": bins,
        "image_base64": image_b64,
    }


def get_tools() -> list[StructuredTool]:
    """Create all structured tools exposed to the agent."""
    return [
        StructuredTool.from_function(
            func=calculate_percentage,
            name="calculate_percentage",
            description=(
                "Calculate the percentage of rows where a column matches a value. "
                "Use this for percentage questions."
            ),
            args_schema=PercentageInput,
        ),
        StructuredTool.from_function(
            func=average,
            name="average",
            description="Compute average of a numeric column.",
            args_schema=AverageInput,
        ),
        StructuredTool.from_function(
            func=count_by,
            name="count_by",
            description="Count records grouped by a column.",
            args_schema=CountByInput,
        ),
        StructuredTool.from_function(
            func=summary_stats,
            name="summary_stats",
            description="Return summary statistics for a numeric column.",
            args_schema=SummaryStatsInput,
        ),
        StructuredTool.from_function(
            func=histogram,
            name="histogram",
            description="Create a real histogram for a numeric column and return image_base64.",
            args_schema=HistogramInput,
        ),
    ]

