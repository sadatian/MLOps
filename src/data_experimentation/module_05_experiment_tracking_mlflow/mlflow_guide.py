# %% [markdown]
# # 🧪 Experiment Tracking & Model Registry with MLflow
#
# Track, log, compare, and register machine learning experiments!
#
# ### The MLflow Component Architecture
#
# MLflow is organized into independent components that coordinate via backend interfaces:
# 1. **MLflow Tracking:** Logs parameters (hyperparameters, dataset versions) and metrics (loss, accuracy) using a client/server REST interface.
# 2. **Backend Store:** Saves lightweight run metadata, metrics, and registered model declarations inside a SQL relational database (SQLite, Postgres, RDS).
# 3. **Artifact Store:** Saves heavy binary artifacts (trained model weight files, environment specs, charts, and evaluations) inside an object storage backend (AWS S3, MinIO, or local directories).
# 4. **Model Registry:** Wraps registered models in version control and deployment stages (`Staging`, `Production`, `Archived`).
#
# ```mermaid
#  graph TD
#      subgraph training_script_run ["Training Script Run"]
#          A["Model Training Code"] -->|"SDK Calls"| B["MLflow Client"]
#      end
#
#      subgraph mlflow_server_boundaries ["MLflow Server Boundaries"]
#          B -->|"REST: log_metric / log_param"| C["Backend Tracking DB"]
#          B -->|"REST / Boto3: log_model"| D["Artifact Store Controller"]
#
#          C -->|"Relational Storage"| E["(SQLite / PostgreSQL DB)"]
#          D -->|"Object Storage"| F["AWS S3 / Artifact Bucket"]
#      end
#
#      subgraph downstream_client_fastapi ["Downstream Client (FastAPI)"]
#          E -->|"Register / Query Production version"| G["Model Registry Interface"]
#          G -->|"Retrieve S3 Model URI"| H["API Serving Loader"]
#          F -->|"Download model.pkl"| H
#      end
#
# ```
#
# In this module, we will:
# 1. Initialize MLflow tracking locally.
# 2. Train a simple regression model using `scikit-learn`.
# 3. Log parameters (learning rate, model type).
# 4. Log metrics (Mean Squared Error, R2 Score).
# 5. Log and save the model artifact.
# 6. Register the model to the MLflow Model Registry.


# %%
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load the synthetic housing data (fallback to creating it if not present)
data_path = "data/housing_raw.csv"
if not os.path.exists(data_path):
    print("Warning: data/housing_raw.csv not found, generating temporary data...")
    os.makedirs("data", exist_ok=True)
    pd.DataFrame({
        "area_sqft": np.random.randint(800, 3500, size=100),
        "bedrooms": np.random.randint(1, 6, size=100),
        "price_usd": np.random.randint(150000, 800000, size=100)
    }).to_csv(data_path, index=False)

df = pd.read_csv(data_path)
X = df[["area_sqft", "bedrooms"]]
y = df["price_usd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## 🔬 1. Set Up MLflow Tracking
# By default, MLflow logs metadata and artifacts to a local directory named `mlruns/`.
# Let's set an explicit experiment name.

# %%
experiment_name = "Housing_Price_Prediction"
mlflow.set_experiment(experiment_name)

print(f"🔬 MLflow Tracking active. Experiment: {experiment_name}")
print(f"Local tracking URI: {mlflow.get_tracking_uri()}")

# %% [markdown]
# ## 🏋️ 2. Train and Log the Run
# We will use `mlflow.start_run()` to track this model training phase.

# %%
run_name = "linear_regression_baseline"

with mlflow.start_run(run_name=run_name) as run:
    # 1. Define model hyperparameters
    fit_intercept = True
    
    # 2. Instantiate and train model
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)
    
    # 3. Predict and calculate metrics
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # 4. Log parameters
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("fit_intercept", fit_intercept)
    mlflow.log_param("train_samples", len(X_train))
    
    # 5. Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    
    # 6. Log the scikit-learn model artifact
    # We can also register the model automatically with registered_model_name
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="HousingPriceRegressionModel"
    )
    
    print("\n🎉 MLflow Run Complete!")
    print(f"Run ID:      {run.info.run_id}")
    print(f"MSE:         {mse:.2f}")
    print(f"R2 Score:    {r2:.4f}")
    print(f"Model URI:   {model_info.model_uri}")

# %% [markdown]
# ## 📁 3. Inspecting Saved Artifacts
# Let's check where MLflow stored the registered model files.

# %%
run_id = mlflow.last_active_run().info.run_id
artifact_uri = mlflow.get_run(run_id).info.artifact_uri
print(f"📁 Local Artifact Folder: {artifact_uri}")

# %% [markdown]
# ## 📊 4. Viewing the MLflow UI via CLI
# Module 5 extends the CLI by introducing `mlops mlflow`:
# * **Start the MLflow Tracking Server:**
#   ```bash
#   uv run mlops mlflow server --host 0.0.0.0 --port 5000
#   ```
# * **Run tracking experiments programmatically:**
#   ```bash
#   uv run mlops mlflow run --script src/data_experimentation/module_05_experiment_tracking_mlflow/mlflow_guide.py
#   ```
# * **Run via Docker:**
#   ```bash
#   docker run --rm -it -p 5000:5000 mlops-cli mlflow server --host 0.0.0.0 --port 5000
#   ```
#
# > [!TIP]
# > **ONNX Alternative**: While we log standard scikit-learn model serialization by default, production systems often use the Open Neural Network Exchange (ONNX) format via `mlflow.onnx.log_model` to avoid Python-specific deserialization vulnerabilities (pickle) and to improve inference latency.
#
# Let's inspect the MLflow CLI options:

# %%
import subprocess
result = subprocess.run(["mlops", "mlflow", "--help"], capture_output=True, text=True)
print(result.stdout)

# %% [markdown]
# Now that we know how to track experiments, let's proceed to the Integrated MLOps Pipeline guide to see how DVC Pipelines and MLflow are integrated!
