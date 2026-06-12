import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
import pandas as pd

from src.module_07_model_serving.serve_api import app

client = TestClient(app)

class MockModel:
    def predict(self, df: pd.DataFrame):
        # Simply return a dummy prediction based on features
        area_k_sqft = df["area_k_sqft"].iloc[0]
        bedrooms = df["bedrooms"].iloc[0]
        return [area_k_sqft * 100000.0 + bedrooms * 15000.0]

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@patch("src.module_07_model_serving.serve_api.load_inference_model")
def test_predict_endpoint(mock_load):
    mock_load.return_value = MockModel()
    
    payload = {
        "area_sqft": 1500.0,
        "bedrooms": 3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price_usd" in data
    # area_k_sqft = 1.5, bedrooms = 3 -> prediction = 1.5 * 100000 + 3 * 15000 = 195000.0
    assert data["predicted_price_usd"] == 195000.0

def test_predict_validation_error():
    # Sending invalid payload (missing bedrooms)
    payload = {
        "area_sqft": 1500.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
