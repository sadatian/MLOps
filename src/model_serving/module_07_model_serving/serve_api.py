# %% [markdown]
# # Model Serving API with FastAPI
#
# Once an ML model is trained, it needs to be served so downstream applications can request predictions.
# In this module, we will explore:
# 1. Defining a web request payload schema using `Pydantic`.
# 2. Writing a `FastAPI` application with a POST `/predict` endpoint.
# 3. Reading our trained Random Forest model from the Integrated MLOps Pipeline.
# 4. Starting a local server and verifying predictions programmatically.

# %%
import os
import pickle
import threading
import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import requests
import boto3
from moto import mock_aws

# Ensure boto3 uses dummy credentials for local simulation
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# %% [markdown]
# ## 1. Defining Input Schema & Creating FastAPI App
# We define our input features (`area_sqft` and `bedrooms`) and instantiate the web framework.

# %%
class InferenceInput(BaseModel):
    area_sqft: float = Field(..., description="Square footage of the house", json_schema_extra={"example": 1500})
    bedrooms: int = Field(..., description="Number of bedrooms", json_schema_extra={"example": 3})

class InferenceOutput(BaseModel):
    predicted_price_usd: float

app = FastAPI(title="Housing Price Inference Service", version="1.0.0")

# Model Registry S3 coordinates (for simulated cloud deployment)
S3_BUCKET = "mlops-model-registry"
S3_KEY = "models/housing_model.pkl"
MODEL_PATH = "data/model.pkl"

def load_inference_model():
    """Attempts to download and load the model from mock S3 registry, with local fallback."""
    try:
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        model_bytes = response["Body"].read()
        print(f"📡 S3 Model Loading: Successfully retrieved model from s3://{S3_BUCKET}/{S3_KEY}")
        return pickle.loads(model_bytes)
    except Exception as e:
        print(f"⚠️ S3 Model Loading: Failed/skipped S3 retrieval ({e}). Falling back to local file: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found locally or on S3.")
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

# Define predictive logic route
@app.post("/predict", response_model=InferenceOutput)
def predict(payload: InferenceInput):
    model = load_inference_model()
    
    # Preprocess: The Integrated MLOps Pipeline expects scaled area ('area_k_sqft')
    area_k_sqft = payload.area_sqft / 1000.0
    bedrooms = payload.bedrooms
    
    # Run prediction
    features = pd.DataFrame([[area_k_sqft, bedrooms]], columns=["area_k_sqft", "bedrooms"])
    prediction = model.predict(features)[0]
    
    return InferenceOutput(predicted_price_usd=float(prediction))

@app.get("/health")
def health():
    return {"status": "healthy"}

# %% [markdown]
# ## 2. Starting and Testing the Server Locally
# To see this server in action without locking our terminal, we will run the `uvicorn` server inside a background thread, query the endpoints using python `requests`, and then close the server thread.

# %%
# Function to run uvicorn server in a separate thread
def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    # Wrap server thread and testing in a mock S3 context
    with mock_aws():
        # Setup mock S3 registry and upload model to simulate cloud serving
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=S3_BUCKET)
        
        # Check if local model exists (run preparation train if not found)
        if not os.path.exists(MODEL_PATH):
            print("⚠️ Model pkl not found. Generating default model for serving...")
            from src.data_experimentation.module_06_integrated_pipeline.run_pipeline import train_model
            train_model("data/housing_train.csv", MODEL_PATH)
        
        with open(MODEL_PATH, "rb") as f:
            s3.put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=f.read())
        print(f"📦 Mock S3 Registry: Model successfully uploaded to 's3://{S3_BUCKET}/{S3_KEY}'")

        # Start server thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        print("📡 FastAPI server starting in background thread...")
        time.sleep(2) # Give the server time to bind and spin up

        try:
            # 1. Health check
            health_url = "http://127.0.0.1:8000/health"
            health_response = requests.get(health_url)
            print(f"Health Status: {health_response.json()}")

            # 2. Prediction request
            predict_url = "http://127.0.0.1:8000/predict"
            payload = {
                "area_sqft": 1850.0,
                "bedrooms": 3
            }
            print(f"Sending input payload: {payload}")
            predict_response = requests.post(predict_url, json=payload)
            
            print("\n✅ Server response:")
            print(predict_response.json())

        except Exception as e:
            print(f"❌ Failed to communicate with FastAPI server: {e}")

# %% [markdown]
# ## 3. Serve in Production
# To launch the server in production (outside of a background thread) so it stays listening, execute this command in your WSL console:
# ```bash
# uv run uvicorn src.model_serving.module_07_model_serving.serve_api:app --host 0.0.0.0 --port 8000 --reload
# ```
#
# Now that we've served the model locally, let's step into the Docker Containerization guide to package this API into a Docker container!
