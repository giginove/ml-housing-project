import pandas as pd
from sklearn.model_selection import train_test_split

from ml_housing.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def split_features_target(df: pd.DataFrame):
    """Separate input features from the regression target."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def split_train_test(
    X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
):
    """Create a reproducible train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
