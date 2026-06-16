import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Instantiate FastAPI application
app = FastAPI(title="MLOps Workspace Serving API", version="0.1.0")

class PredictRequest(BaseModel):
    model_id: str = "random_forest"
    features: Dict[str, Any]

class InferenceOutput(BaseModel):
    predicted_price_usd: float

def engine_predict(model_id: str, features: dict) -> dict:
    """Shared prediction engine mapping model types to their inference logic."""
    if model_id == "heuristic":
        # Sprint 0 heuristic pricing baseline logic
        area_sqft = float(features.get("area_sqft", 0.0))
        prediction = 150000.0 + area_sqft * 150.0
        return {"predicted_price_usd": prediction}

    model_paths = {
        "random_forest": "data/model.pkl",
        "linear_regression": "data/model_linear.pkl"
    }

    if model_id not in model_paths:
        raise ValueError(f"Unknown model ID '{model_id}'. Supported: random_forest, linear_regression, heuristic")

    local_path = model_paths[model_id]
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Model file '{local_path}' not found. Please run training pipeline first.")

    with open(local_path, "rb") as f:
        model = pickle.load(f)

    # Preprocess features: Models expect scaled area ('area_k_sqft') and bedrooms
    area_sqft = float(features.get("area_sqft", 0.0))
    bedrooms = int(features.get("bedrooms", 0))
    area_k_sqft = area_sqft / 1000.0

    features_df = pd.DataFrame([[area_k_sqft, bedrooms]], columns=["area_k_sqft", "bedrooms"])
    prediction = model.predict(features_df)[0]
    return {"predicted_price_usd": float(prediction)}

@app.post("/predict")
def predict(payload: PredictRequest):
    """POST endpoint forwarding input payload features to the prediction engine."""
    try:
        result = engine_predict(payload.model_id, payload.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    """Service health check endpoint."""
    return {"status": "healthy"}
