"""Tests for ml_housing.data."""

from ml_housing.config import TARGET_COLUMN
from ml_housing.data import load_housing_data, validate_target_column


def test_load_housing_data_not_empty():
    df = load_housing_data()
    assert not df.empty


def test_target_column_exists_and_is_not_only_null():
    df = load_housing_data()
    assert TARGET_COLUMN in df.columns
    assert df[TARGET_COLUMN].notna().any()
    assert validate_target_column(df)
