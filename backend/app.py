from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


def get_latest_model():
    """Load the active model pipeline from the artifacts folder."""
    project_root = Path(__file__).resolve().parent.parent
    models_path = project_root / "artifacts" / "models"

    latest_model = models_path / "model_latest.joblib"
    if latest_model.exists():
        return joblib.load(latest_model)

    versioned_models = list(models_path.glob("model_v*.joblib"))
    if versioned_models:
        latest_version = sorted(
            versioned_models,
            key=lambda path: int(path.stem.split("_v")[-1]),
        )[-1]
        return joblib.load(latest_version)

    default_model = project_root / "artifacts" / "model.joblib"
    if default_model.exists():
        return joblib.load(default_model)

    raise FileNotFoundError(f"No model found in {models_path}")


model = get_latest_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: HousingFeatures):
    input_df = pd.DataFrame([data.model_dump()], columns=FEATURE_COLUMNS)
    prediction = model.predict(input_df)[0]
    return {"prediction": float(prediction)}
