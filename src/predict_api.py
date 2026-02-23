from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression_pipeline.joblib"

app = FastAPI(title="Real-Time Linear Regression Prediction API")


class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms")
    AveBedrms: float = Field(..., description="Average number of bedrooms")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average house occupancy")
    Latitude: float = Field(..., description="Latitude")
    Longitude: float = Field(..., description="Longitude")


@app.on_event("startup")
def load_model() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model not found. Run `python src/train.py` before starting the API."
        )

    app.state.model = joblib.load(MODEL_PATH)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(features: HousingFeatures) -> dict[str, float]:
    try:
        row = pd.DataFrame([features.model_dump()])
        prediction = float(app.state.model.predict(row)[0])
    except Exception as exc:  # pragma: no cover - API safety net
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"prediction": prediction}
