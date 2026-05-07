from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml_housing.config import RANDOM_FOREST_N_JOBS, RANDOM_STATE


def train_model(X_train, y_train, model_name="random_forest"):
    """Train a regression model based on model_name."""

    if model_name == "linear":
        model = LinearRegression()

    elif model_name == "gbr":
        model = GradientBoostingRegressor(random_state=RANDOM_STATE)

    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=RANDOM_FOREST_N_JOBS,
        )

    else:
        valid_models = "linear, gbr, random_forest"
        raise ValueError(
            f"Unknown model_name '{model_name}'. Choose one of: {valid_models}."
        )

    model.fit(X_train, y_train)
    return model
