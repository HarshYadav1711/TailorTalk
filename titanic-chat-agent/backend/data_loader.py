"""Dataset loading utilities for the Titanic chat agent."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

TITANIC_CSV_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)


@lru_cache(maxsize=1)
def load_titanic_df() -> pd.DataFrame:
    """Load Titanic dataset into memory and return a cached DataFrame."""
    df = pd.read_csv(TITANIC_CSV_URL)
    df.columns = [col.strip().lower() for col in df.columns]
    return df

