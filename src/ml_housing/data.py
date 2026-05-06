"""Data loading and validation helpers for the housing ML project."""

import pandas as pd
from sklearn.datasets import fetch_california_housing

from ml_housing.config import TARGET_COLUMN


def load_housing_data() -> pd.DataFrame:
    """Load the California Housing dataset as a DataFrame."""
    dataset = fetch_california_housing(as_frame=True)
    return dataset.frame


def validate_target_column(df: pd.DataFrame) -> bool:
    """Return True when the target exists and has at least one non-null value."""
    return TARGET_COLUMN in df.columns and df[TARGET_COLUMN].notna().any()
