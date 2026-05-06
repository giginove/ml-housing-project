from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression  # fallback

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


# -------------------------
# LAZY LOADING DU MODELE
# -------------------------
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = get_latest_model()
        except FileNotFoundError:
            # MODE CI : modèle factice
            _model = LinearRegression()
    return _model


@app.get("/health")
def health():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "artifacts" / "models" / "model_latest.joblib"
    return {
        "status": "ok",
        "model_loaded": model_path.name,
        "exists": model_path.exists(),
    }


@app.post("/predict")
def predict(data: HousingFeatures):
    model = get_model()
    input_df = pd.DataFrame([data.model_dump()], columns=FEATURE_COLUMNS)
    prediction = model.predict(input_df)[0]
    return {"prediction": float(prediction)}
