# %% [markdown]
# # Module 07: Model Serving API with FastAPI
#
# Once an ML model is trained, it needs to be served so downstream applications can request predictions.
# In this module, we will explore:
# 1. Defining a web request payload schema using `Pydantic`.
# 2. Writing a `FastAPI` application with a POST `/predict` endpoint.
# 3. Reading our trained Random Forest model from Module 06.
# 4. Starting a local server and verifying predictions programmatically.

# %%
import os
import pickle
import threading
import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import requests

# %% [markdown]
# ## 1. Defining Input Schema & Creating FastAPI App
# We define our input features (`area_sqft` and `bedrooms`) and instantiate the web framework.

# %%
class InferenceInput(BaseModel):
    area_sqft: float = Field(..., description="Square footage of the house", example=1500)
    bedrooms: int = Field(..., description="Number of bedrooms", example=3)

class InferenceOutput(BaseModel):
    predicted_price_usd: float

app = FastAPI(title="Housing Price Inference Service", version="1.0.0")

# Load model pickle safely
MODEL_PATH = "data/model.pkl"

def load_inference_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please run Module 06 pipeline first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Define predictive logic route
@app.post("/predict", response_model=InferenceOutput)
def predict(payload: InferenceInput):
    model = load_inference_model()
    
    # Preprocess: Module 06 expects scaled area ('area_k_sqft')
    area_k_sqft = payload.area_sqft / 1000.0
    bedrooms = payload.bedrooms
    
    # Run prediction
    features = [[area_k_sqft, bedrooms]]
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

# Check if model exists (run preparation train if not found)
if not os.path.exists(MODEL_PATH):
    print("⚠️ Model pkl not found. Generating default model for serving...")
    from src.module_06_integrated_pipeline.run_pipeline import train_model
    train_model("data/housing_train.csv", MODEL_PATH)

# Start server thread
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
print("📡 FastAPI server starting in background thread...")
time.sleep(2) # Give the server time to bind and spin up

# %% [markdown]
# ### Querying `/health` and `/predict` endpoints:

# %%
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
# uv run uvicorn src.module_07_model_serving.serve_api:app --host 0.0.0.0 --port 8000 --reload
# ```
#
# Now that we've served the model locally, let's step into **Module 08** to package this API into a Docker container!
