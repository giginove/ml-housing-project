"""Data preprocessing utilities."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_preprocessing_pipeline() -> Pipeline:
    """Create the preprocessing pipeline for numeric housing features."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
