from ml_housing.config import TARGET_COLUMN
from ml_housing.data import load_housing_data


def test_target_column_valid():
    df = load_housing_data()
    assert TARGET_COLUMN in df.columns
    assert df[TARGET_COLUMN].notna().any()
