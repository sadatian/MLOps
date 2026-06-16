# %% [markdown]
# # 🚀 Model Serving API with FastAPI
#
# Once an ML model is trained, it needs to be served so downstream applications can request predictions.
#
# ### The REST Model Serving Architecture
#
# Serving machine learning models requires robust API packaging to handle data serialization, validation, and schema compliance. 
# `FastAPI` combined with `Pydantic` provides:
# 1. **Data Validation:** Automatic validation of incoming JSON request formats against defined type hints.
# 2. **Auto-generated Docs:** Standard OpenAPI specifications and interactive `/docs` Swagger pages.
# 3. **High Performance:** Async routing and fast Python serialization based on Rust backend parsing.
#
# ```mermaid
#  graph TD
#      subgraph http_inference_request ["HTTP Inference Request"]
#          A["Client App"] -->|"HTTP POST /predict JSON"| B["FastAPI Web Server"]
#      end
#
#      subgraph fastapi_request_processing ["FastAPI Request Processing"]
#          B -->|"Pydantic Check"| C{"InferenceInput Schema Valid?"}
#          C -->|"No"| D["HTTP 422 Error Response"]
#          C -->|"Yes"| E["Call predict handler"]
#
#          E -->|"Query model"| F{"Model in Memory?"}
#          F -->|"No: load_inference_model"| G{"S3 Registry Available?"}
#          G -->|"Yes"| H["Download from S3"]
#          G -->|"No"| I["Fallback to local disk pkl"]
#          H -->|"Load object"| J["scikit-learn Model loaded"]
#          I -->|"Load object"| J
#          J -->|"Cache in RAM"| F
#
#          E -->|"Scale area_sqft / 1000"| K[area_k_sqft]
#          K -->|"Build DataFrame"| L["Feature DataFrame"]
#          J -->|"model.predict"| M["Float prediction value"]
#          M -->|"Serialize to InferenceOutput"| N["HTTP 200 OK JSON Response"]
#      end
#
#      style D fill:#f8d7da,stroke:#dc3545,stroke-width:1.5px
#      style N fill:#d4edda,stroke:#28a745,stroke-width:1.5px
#
# ```
#
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
# ## 📝 1. Defining Input Schema & Creating FastAPI App
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
# ## 📡 2. Starting and Testing the Server Locally
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
# ## 🚀 3. Serve in Production via CLI
# Module 7 extends the CLI by introducing `mlops serve`:
# * **Start serving API locally:**
#   ```bash
#   uv run mlops serve --host 0.0.0.0 --port 8000
#   ```
# * **Run via Docker (runs serving by default):**
#   ```bash
#   docker run --rm -it -p 8000:8000 mlops-cli
#   ```
#
# > [!TIP]
# > **ONNX Runtime Serving**: For robust high-throughput serving, production frameworks often load an ONNX model (`model.onnx`) and execute predictions using `onnxruntime` within the FastAPI endpoint. This removes standard framework dependencies like scikit-learn from the serving image.
#
# Let's inspect the serving CLI command structure:

# %%
import subprocess
result = subprocess.run(["mlops", "serve", "--help"], capture_output=True, text=True)
print(result.stdout)

# %% [markdown]
# Now that we've served the model locally, let's step into the Docker Containerization guide to package this API into a Docker container!
