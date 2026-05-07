import pytest

from ml_housing.pipeline import run_pipeline


def assert_valid_regression_metrics(metrics):
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert metrics["rmse"] >= metrics["mae"]
    assert 0 <= metrics["r2"] <= 1


def test_pipeline_random_forest(tmp_path):
    metrics = run_pipeline(artifacts_dir=str(tmp_path), model_name="random_forest")

    assert_valid_regression_metrics(metrics)


def test_pipeline_linear(tmp_path):
    metrics = run_pipeline(artifacts_dir=str(tmp_path), model_name="linear")

    assert_valid_regression_metrics(metrics)


def test_pipeline_gbr(tmp_path):
    metrics = run_pipeline(artifacts_dir=str(tmp_path), model_name="gbr")

    assert_valid_regression_metrics(metrics)


def test_pipeline_unknown_model_name_raises_error(tmp_path):
    with pytest.raises(ValueError, match="Unknown model_name"):
        run_pipeline(artifacts_dir=str(tmp_path), model_name="not_a_model")
