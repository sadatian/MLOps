# %% [markdown]
# # 🔗 Integrated MLOps Pipeline (DVC + MLflow)
#
# A production MLOps pipeline divides the ML workflow into structured, reproducible steps. 
#
# ### The DVC + MLflow Unified Lifecycle
#
# In a robust production environment, DVC and MLflow work together as complements, not competitors:
# *   **DVC** acts as the build orchestrator and dependency manager. It manages pipeline execution caching so that if code and data dependencies do not change, stages are skipped, preventing wasted compute.
#   **MLflow** acts as the logger, auditor, and deployment manager. It tracks evaluation metrics across runs, captures hyperparameters, and holds model versions in the Model Registry.
#
# ```mermaid
# graph TD
#     subgraph DVC Pipeline Orchestration (dvc.yaml DAG)
#         A[data/housing_raw.csv] -->|dvc stage: prepare| B[Data Prep Stage]
#         B -->|Outputs split data| C[data/housing_train.csv]
#         B -->|Outputs split data| D[data/housing_test.csv]
#         
#         C -->|dvc stage: train| E[Model Training Stage]
#         E -->|Outputs weights| F[data/model.pkl]
#         
#         D -->|dvc stage: evaluate| G[Evaluation Stage]
#         F -->|Input weights| G
#         G -->|Outputs local metric file| H[data/metrics.json]
#     end
# 
#     subgraph MLflow Server Tracking (Runtime Logs)
#         E -->|mlflow.log_params| I[MLflow Tracking DB]
#         G -->|mlflow.log_metric| I
#         G -->|mlflow.log_model| J[MLflow Artifact Store (S3)]
#     end
# 
#     style B fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px
#     style E fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px
#     style G fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px
# ```
#
# In this module, we will explore:
# 1. Splitting the ML workflow into distinct stages (Data Prep -> Train -> Evaluate).
# 2. Writing a pipeline manager that executes these stages sequentially.
# 3. Generating a `dvc.yaml` configuration to allow DVC to orchestrate the pipeline.
# 4. Connecting DVC's file caching to MLflow's experiment logs.


# %%
import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import mlflow

# %% [markdown]
# ## 📖 1. Step 1: Data Preparation
# We load the raw dataset, scale the numerical values, split it into train/test, and save them.

# %%
def prepare_data(raw_path, train_path, test_path):
    print(f"📖 Reading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # Simple preprocessing: scale sqft (divide by 1000)
    df["area_k_sqft"] = df["area_sqft"] / 1000.0
    
    # Train test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save outputs
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"✅ Preprocessed and split data. Saved to '{train_path}' and '{test_path}'")

# %% [markdown]
# ## 🏋️ 2. Step 2: Model Training & Tracking
# We train a RandomForestRegressor model, serialize it, and track the process using MLflow.

# %%
def train_model(train_path, model_path, n_estimators=50, max_depth=5):
    print(f"🏋️ Training model on {train_path}...")
    train_df = pd.read_csv(train_path)
    X_train = train_df[["area_k_sqft", "bedrooms"]]
    y_train = train_df["price_usd"]
    
    # Train
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model binary
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"✅ Model serialized and saved to {model_path}")
    return model

# %% [markdown]
# ## 📊 3. Step 3: Evaluation & Metric Exports
# We evaluate the trained model, save a local `metrics.json` file (tracked by DVC), and log to MLflow.

# %%
def evaluate_model(test_path, model_path, metrics_path):
    print(f"📊 Evaluating model using test set {test_path}...")
    test_df = pd.read_csv(test_path)
    X_test = test_df[["area_k_sqft", "bedrooms"]]
    y_test = test_df["price_usd"]
    
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Evaluate
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Write metrics locally (for DVC metric tracking)
    metrics = {"rmse": rmse, "r2_score": r2}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Log to MLflow
    mlflow.set_experiment("Integrated_Pipeline")
    with mlflow.start_run(run_name="dvc_pipeline_run"):
        mlflow.log_params(model.get_params())
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, "random_forest_model", registered_model_name="HousingRandomForestModel")
        
    print(f"✅ Evaluation Complete. Metrics saved to {metrics_path} and logged to MLflow.")
    print(f"RMSE: {rmse:.2f}, R2: {r2:.4f}")

# %% [markdown]
# ## ⚡ 4. Run the Pipeline Sequentially
# Let's run all steps programmatically inside this cell to test the logic.

# %%
# Define file variables
raw_data = "data/housing_raw.csv"
train_data = "data/housing_train.csv"
test_data = "data/housing_test.csv"
model_pkl = "data/model.pkl"
eval_json = "data/metrics.json"

# Run steps
prepare_data(raw_data, train_data, test_data)
train_model(train_data, model_pkl)
evaluate_model(test_data, model_pkl, eval_json)

# %% [markdown]
# ## 🔗 5. Orchestrating with DVC (`dvc.yaml`)
# In production, instead of running these steps in a single python file, we define them in a `dvc.yaml` file so DVC can cache steps and only rerun them if inputs (code or data) change.
#
# To create the pipeline stages natively, execute the following commands in your terminal:
#
# ```bash
# # 1. Add data preparation stage
# uv run dvc stage add -n prepare \
#   -d src/data_experimentation/module_06_integrated_pipeline/run_pipeline.py \
#   -d data/housing_raw.csv \
#   -o data/housing_train.csv \
#   -o data/housing_test.csv \
#   "uv run python -c \"from src.data_experimentation.module_06_integrated_pipeline.run_pipeline import prepare_data; prepare_data('data/housing_raw.csv', 'data/housing_train.csv', 'data/housing_test.csv')\""
#
# # 2. Add model training stage
# uv run dvc stage add -n train \
#   -d src/data_experimentation/module_06_integrated_pipeline/run_pipeline.py \
#   -d data/housing_train.csv \
#   -o data/model.pkl \
#   "uv run python -c \"from src.data_experimentation.module_06_integrated_pipeline.run_pipeline import train_model; train_model('data/housing_train.csv', 'data/model.pkl')\""
#
# # 3. Add model evaluation stage
# uv run dvc stage add -n evaluate \
#   -d src/data_experimentation/module_06_integrated_pipeline/run_pipeline.py \
#   -d data/housing_test.csv \
#   -d data/model.pkl \
#   -M data/metrics.json \
#   "uv run python -c \"from src.data_experimentation.module_06_integrated_pipeline.run_pipeline import evaluate_model; evaluate_model('data/housing_test.csv', 'data/model.pkl', 'data/metrics.json')\""
# ```

# %%
# Let's verify that dvc.yaml is correctly configured
if os.path.exists("dvc.yaml"):
    print("✅ dvc.yaml exists and is configured properly in the root directory:")
    with open("dvc.yaml", "r") as f:
        print(f.read())
else:
    print("❌ dvc.yaml was not found.")

# %% [markdown]
# ## 💾 6. Tracking Pipeline Configuration with Git
#
# Every time you create or modify pipeline stages, DVC updates `dvc.yaml` and `dvc.lock`. To keep Git and DVC in sync, you should track these files in Git:
# ```bash
# git add dvc.yaml
# ```
#
# ### Auto-Staging Changes
# You can instruct DVC to automatically stage these configuration files to Git every time DVC commands update them:
# ```bash
# dvc config core.autostage true
# ```
# This updates `.dvc/config` (which you should also commit to git).
#
# ### Executing the Pipeline
# To run the pipeline stages and compile outputs, execute:
# ```bash
# dvc repro
# ```
# DVC will automatically execute the stages in order, cache outputs, and if you rerun it without modifying inputs or code, it will skip execution and return `Data and pipelines are up to date.`!
#
# Let's proceed to the Model Serving API guide to serve our trained model via FastAPI!
