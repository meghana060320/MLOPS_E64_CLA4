"""FastAPI model-serving application."""
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
app = FastAPI(title="Titanic Survival Prediction API", version="1.0.0")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class Passenger(BaseModel):
    Pclass: int = Field(ge=1, le=3, description="Ticket class: 1, 2, or 3")
    Sex: Literal["male", "female"]
    Age: float | None = Field(default=None, ge=0, le=100)
    SibSp: int = Field(default=0, ge=0, le=10)
    Parch: int = Field(default=0, ge=0, le=10)
    Fare: float = Field(ge=0)
    Embarked: Literal["C", "Q", "S"] = "S"

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(passenger: Passenger):
    if model is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Model not found. Run train_model.py first.")
    row = pd.DataFrame([passenger.model_dump()])
    result = int(model.predict(row)[0])
    return {
        "prediction": result,
        "label": "Survived" if result == 1 else "Did not survive",
    }
