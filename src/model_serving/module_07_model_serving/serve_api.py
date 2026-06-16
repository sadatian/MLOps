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

# Model Registry S3 coordinates and local paths
S3_BUCKET = "mlops-model-registry"
MODEL_CONFIGS = {
    "random_forest": {
        "s3_key": "models/housing_model.pkl",
        "local_path": "data/model.pkl"
    },
    "linear_regression": {
        "s3_key": "models/housing_linear.pkl",
        "local_path": "data/model_linear.pkl"
    }
}

# Cache to store loaded models in memory
model_cache = {}
model_cache_lock = threading.Lock()

def load_inference_model(model_name: str):
    """Attempts to download and load the model from mock S3 registry, with local fallback."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}")

    with model_cache_lock:
        if model_name in model_cache:
            return model_cache[model_name]

    config = MODEL_CONFIGS[model_name]
    s3_key = config["s3_key"]
    local_path = config["local_path"]

    try:
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        model_bytes = response["Body"].read()
        print(f"📡 S3 Model Loading: Successfully retrieved '{model_name}' from s3://{S3_BUCKET}/{s3_key}")
        model = pickle.loads(model_bytes)
    except Exception as e:
        print(f"⚠️ S3 Model Loading: Failed/skipped S3 retrieval for '{model_name}' ({e}). Falling back to local: {local_path}")
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Model file '{local_path}' not found locally or on S3.")
        with open(local_path, "rb") as f:
            model = pickle.load(f)

    with model_cache_lock:
        model_cache[model_name] = model

    return model

# Define predictive logic route supporting multiple models
@app.post("/predict/{model_name}", response_model=InferenceOutput)
def predict_model(model_name: str, payload: InferenceInput):
    if model_name == "heuristic":
        # Sprint 0 heuristic pricing baseline logic
        prediction = 150000.0 + payload.area_sqft * 150.0
        return InferenceOutput(predicted_price_usd=float(prediction))

    if model_name not in MODEL_CONFIGS:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Supported: random_forest, linear_regression, heuristic")

    model = load_inference_model(model_name)
    
    # Preprocess: The Integrated MLOps Pipeline expects scaled area ('area_k_sqft')
    area_k_sqft = payload.area_sqft / 1000.0
    bedrooms = payload.bedrooms
    
    # Run prediction
    features = pd.DataFrame([[area_k_sqft, bedrooms]], columns=["area_k_sqft", "bedrooms"])
    prediction = model.predict(features)[0]
    
    return InferenceOutput(predicted_price_usd=float(prediction))

# Default route delegates to the primary Random Forest model
@app.post("/predict", response_model=InferenceOutput)
def predict(payload: InferenceInput):
    return predict_model("random_forest", payload)

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
        
        # 1. Check and train Random Forest model
        rf_path = MODEL_CONFIGS["random_forest"]["local_path"]
        rf_key = MODEL_CONFIGS["random_forest"]["s3_key"]
        if not os.path.exists(rf_path):
            print("⚠️ Random Forest model pkl not found. Generating default model...")
            from src.data_experimentation.module_06_integrated_pipeline.run_pipeline import train_model
            train_model("data/housing_train.csv", rf_path)
        with open(rf_path, "rb") as f:
            s3.put_object(Bucket=S3_BUCKET, Key=rf_key, Body=f.read())

        # 2. Check and train Linear Regression model
        lr_path = MODEL_CONFIGS["linear_regression"]["local_path"]
        lr_key = MODEL_CONFIGS["linear_regression"]["s3_key"]
        if not os.path.exists(lr_path):
            print("⚠️ Linear Regression model pkl not found. Generating default model...")
            import numpy as np
            from sklearn.linear_model import LinearRegression
            # Generate dummy train file if needed
            train_path = "data/housing_train.csv"
            if not os.path.exists(train_path):
                os.makedirs("data", exist_ok=True)
                df_train = pd.DataFrame({
                    "area_sqft": np.random.randint(800, 3500, size=50),
                    "bedrooms": np.random.randint(1, 6, size=50),
                    "price_usd": np.random.randint(150000, 800000, size=50)
                })
                df_train.to_csv(train_path, index=False)
            df = pd.read_csv(train_path)
            X = pd.DataFrame()
            X["area_k_sqft"] = df["area_sqft"] / 1000.0
            X["bedrooms"] = df["bedrooms"]
            y = df["price_usd"]
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            with open(lr_path, "wb") as f:
                pickle.dump(lr_model, f)
        with open(lr_path, "rb") as f:
            s3.put_object(Bucket=S3_BUCKET, Key=lr_key, Body=f.read())
            
        print("📦 Mock S3 Registry: Models successfully uploaded to simulated registry")

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

            # 2. Prediction requests testing all three models
            payload = {
                "area_sqft": 1850.0,
                "bedrooms": 3
            }
            
            # Query default route (Random Forest)
            default_res = requests.post("http://127.0.0.1:8000/predict", json=payload)
            print(f"Default (Random Forest) Response: {default_res.json()}")

            # Query Random Forest route explicitly
            rf_res = requests.post("http://127.0.0.1:8000/predict/random_forest", json=payload)
            print(f"Explicit Random Forest Response: {rf_res.json()}")

            # Query Linear Regression route
            lr_res = requests.post("http://127.0.0.1:8000/predict/linear_regression", json=payload)
            print(f"Linear Regression Response: {lr_res.json()}")

            # Query Heuristic Baseline route
            h_res = requests.post("http://127.0.0.1:8000/predict/heuristic", json=payload)
            print(f"Heuristic Baseline Response: {h_res.json()}")

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
