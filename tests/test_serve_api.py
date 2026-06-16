import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
import pandas as pd

from src.model_serving.module_07_model_serving.serve_api import app

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

@patch("src.model_serving.module_07_model_serving.serve_api.load_inference_model")
def test_predict_endpoints(mock_load):
    mock_load.return_value = MockModel()
    
    payload = {
        "area_sqft": 1500.0,
        "bedrooms": 3
    }
    
    # 1. Test default predict endpoint
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_price_usd"] == 195000.0

    # 2. Test explicit random forest endpoint
    response_rf = client.post("/predict/random_forest", json=payload)
    assert response_rf.status_code == 200
    assert response_rf.json()["predicted_price_usd"] == 195000.0

    # 3. Test explicit linear regression endpoint
    response_lr = client.post("/predict/linear_regression", json=payload)
    assert response_lr.status_code == 200
    assert response_lr.json()["predicted_price_usd"] == 195000.0

    # 4. Test heuristic endpoint
    response_h = client.post("/predict/heuristic", json=payload)
    assert response_h.status_code == 200
    # Price = 150000 + 1500 * 150 = 375000.0
    assert response_h.json()["predicted_price_usd"] == 375000.0

    # 5. Test invalid model routing returns 404
    response_invalid = client.post("/predict/nonexistent_model", json=payload)
    assert response_invalid.status_code == 404

def test_predict_validation_error():
    # Sending invalid payload (missing bedrooms)
    payload = {
        "area_sqft": 1500.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_load_inference_model_from_s3():
    import pickle
    import boto3
    from moto import mock_aws
    from src.model_serving.module_07_model_serving.serve_api import load_inference_model, S3_BUCKET, MODEL_CONFIGS
    
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=S3_BUCKET)
        
        dummy_model = MockModel()
        serialized = pickle.dumps(dummy_model)
        s3.put_object(Bucket=S3_BUCKET, Key=MODEL_CONFIGS["random_forest"]["s3_key"], Body=serialized)
        
        loaded = load_inference_model("random_forest")
        # Verify it loaded the pickle correctly
        prediction = loaded.predict(pd.DataFrame([[1.5, 3]], columns=["area_k_sqft", "bedrooms"]))[0]
        assert prediction == 195000.0
