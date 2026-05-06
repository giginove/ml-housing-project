import sys
from pathlib import Path

# Make the local src/ package importable when running `python main.py`.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


if __name__ == "__main__":
    from ml_housing.pipeline import run_pipeline

    metrics = run_pipeline()
    print("Pipeline termine avec succes.")
    print(f"MAE  : {metrics['mae']:.4f}")
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"R2   : {metrics['r2']:.4f}")
