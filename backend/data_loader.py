"""Dataset loading utilities for the Titanic chat agent."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

TITANIC_CSV_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
TITANIC_CSV_PATH = Path(__file__).resolve().parent / "titanic.csv"


@lru_cache(maxsize=1)
def load_titanic_df() -> pd.DataFrame:
    """Load Titanic dataset into memory and return a cached DataFrame."""
    if TITANIC_CSV_PATH.exists():
        df = pd.read_csv(TITANIC_CSV_PATH)
    else:
        df = pd.read_csv(TITANIC_CSV_URL)
    df.columns = [col.strip().lower() for col in df.columns]
    return df

