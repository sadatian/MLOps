# %% [markdown]
# # 🔄 Continuous Training (CT) & HITL Fallbacks
#
# In a production MLOps lifecycle, a model's performance degrades over time. This degradation is typically driven by two forms of dataset shifts:
# 1. **Covariate Shift $P(X)$:** The input feature distribution shifts, while the mapping $P(Y|X)$ remains constant. (e.g., users start searching for larger houses, but the price per square foot remains the same).
# 2. **Concept Drift $P(Y|X)$:** The relationship between input features and target labels changes, while the input distribution might remain stable. (e.g., a housing market crash occurs, meaning a house of the same size is now valued much lower).
#
# ### The Continuous Training (CT) & Safety Loop
#
# Real-world MLOps pipelines require automated feedback loops to identify model degradation, retrain models on fresh profiles, and promote models safely:
# *   **Automated Diagnostics:** Running statistical checks (like Kolmogorov-Smirnov for Covariate Shift and independent t-tests on prediction residuals for Concept Drift).
# *   **Retraining & Registration:** Retraining the model and uploading the candidate version to the MLflow Model Registry.
# *   **Human-in-the-Loop (HITL) Gate:** Reviewing and signing off on the retrained model prior to promotion.
# *   **Out-of-Distribution (OOD) & Runtime Fallbacks:** Protecting the serving layer by intercepting anomalies and falling back to a safe `HeuristicBaselineModel` if inputs are out-of-distribution (OOD) or if the active model suffers failures.
#
# ```mermaid
#  graph TD
#      subgraph 1_production_serving_logging ["1. Production serving & Logging"]
#          A["FastAPI Server"] -->|"Inference logs"| B["(Inference Database)"]
#          A -->|"Delayed ground-truth labels"| C["(Labels Database)"]
#      end
#
#      subgraph 2_statistical_drift_detection ["2. Statistical Drift Detection"]
#          B -->|"Query features current_df"| D[detect_covariate_shift]
#          C -->|"Query prediction errors"| E[detect_concept_drift]
#          D -->|"KS Test: p < 0.05"| F{"Retrain Triggered?"}
#          E -->|"t-test on residuals: p < 0.05"| F
#      end
#
#      subgraph 3_continuous_training_ct ["3. Continuous Training (CT)"]
#          G["Start Retraining Job"] -->|"Fetch recent training sets"| H["Fit New LinearRegression"]
#          H -->|"Log run parameters & MAE"| I["Register to MLflow Registry"]
#      end
#
#      subgraph 4_release_promotion_gate_hitl ["4. Release Promotion Gate (HITL)"]
#          J["Evaluate MAE: Challenger vs Baseline"] --> K["Human-in-the-Loop Approval Gate"]
#          K -->|"Approved"| L["Promote Model in Registry & Load"]
#          K -->|"Rejected / Timeout"| M["SafeServingWrapper: Fallback Active"]
#      end
#
#      subgraph 5_runtime_serving_guards ["5. Runtime Serving Guards"]
#          N{"OOD Input Check?"} -->|"Yes: Out of Bounds"| O["HeuristicBaselineModel Fallback"]
#          N -->|"No"| P{"Fallback Active?"}
#          P -->|"Yes"| O
#          P -->|"No"| Q["Run active ML Model"]
#      end
#
#      F -->|"Yes"| G
#      I --> J
#      A --> N
#
#      L ~~~ N
#      M ~~~ N
#
#      style F fill:#fff3e0,stroke:#ffb74d,stroke-width:1.5px
#      style K fill:#fff3e0,stroke:#ffb74d,stroke-width:2px
#      style L fill:#d4edda,stroke:#28a745,stroke-width:1.5px
#      style O fill:#f8d7da,stroke:#dc3545,stroke-width:1.5px
#
# ```
#
# In this module, we will implement:
# 1. Programmatic identification of mathematical drift (Covariate Shift and Concept Drift).
# 2. An automated retraining pipeline logged and registered to the MLflow Model Registry.
# 3. A Human-in-the-Loop (HITL) approval gate and fallback serving mechanism to protect production.


# %%
import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict, Any, Tuple, List

# Define file paths
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_NAME = "HousingPriceCTModel"

# Set up MLflow local tracking
mlflow.set_experiment("Continuous_Training_Module_15")

# %% [markdown]
# ## 📊 1. Synthetic Data Generation
# Let's generate:
# - **Reference Data:** Historical baseline.
# - **Covariate Shift Data:** Inputs shift (houses are larger), but pricing rules remain identical.
# - **Concept Drift Data:** Input sizes are similar, but housing prices drop significantly (shift in $P(Y|X)$).

# %%
np.random.seed(42)
num_samples = 200

# 1. Reference Data (Baseline)
ref_area = np.random.normal(1800, 300, size=num_samples)
ref_bedrooms = np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2], size=num_samples)
# Target function: price = 150 * area + 20000 * bedrooms + 50000 + noise
ref_price = 150.0 * ref_area + 20000.0 * ref_bedrooms + 50000.0 + np.random.normal(0, 10000, size=num_samples)

reference_df = pd.DataFrame({
    "area_sqft": ref_area,
    "bedrooms": ref_bedrooms,
    "price_usd": ref_price
})

# 2. Covariate Shift Data (Inputs shift upward, relation remains the same)
cov_area = np.random.normal(2400, 400, size=num_samples) # Mean shifts from 1800 to 2400
cov_bedrooms = np.random.choice([2, 3, 4], p=[0.1, 0.4, 0.5], size=num_samples)
cov_price = 150.0 * cov_area + 20000.0 * cov_bedrooms + 50000.0 + np.random.normal(0, 10000, size=num_samples)

covariate_shift_df = pd.DataFrame({
    "area_sqft": cov_area,
    "bedrooms": cov_bedrooms,
    "price_usd": cov_price
})

# 3. Concept Drift Data (Inputs same as reference, relation shifts)
con_area = np.random.normal(1800, 300, size=num_samples)
con_bedrooms = np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2], size=num_samples)
# New target function: price = 100 * area + 10000 * bedrooms + 30000 + noise (prices drop!)
con_price = 100.0 * con_area + 10000.0 * con_bedrooms + 30000.0 + np.random.normal(0, 10000, size=num_samples)

concept_drift_df = pd.DataFrame({
    "area_sqft": con_area,
    "bedrooms": con_bedrooms,
    "price_usd": con_price
})

print(f"📊 Reference Price Mean:      ${reference_df['price_usd'].mean():,.2f}")
print(f"📊 Covariate Shift Price Mean: ${covariate_shift_df['price_usd'].mean():,.2f}")
print(f"📊 Concept Drift Price Mean:   ${concept_drift_df['price_usd'].mean():,.2f}")

# %% [markdown]
# ## 🔍 2. Mathematical Drift Identification
#
# - **Covariate Shift $P(X)$ Detection:** We run a Kolmogorov-Smirnov (KS) test comparing features from the reference dataset against the production dataset. If the test statistic's p-value is below a threshold (e.g. 0.05), we reject the null hypothesis, confirming a distribution shift.
# - **Concept Drift $P(Y|X)$ Detection:** If ground-truth labels arrive (even with a delay), we run a statistical test (independent t-test) to verify if the prediction errors (residuals) are significantly larger in production compared to reference validation errors.

# %%
def detect_covariate_shift(
    ref_df: pd.DataFrame, 
    cur_df: pd.DataFrame, 
    features: List[str], 
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Identifies covariate shift by running a Kolmogorov-Smirnov test on continuous features.
    """
    drift_results = {}
    any_drift = False

    for feat in features:
        ks_stat, p_value = stats.ks_2samp(ref_df[feat], cur_df[feat])
        drifted = p_value < alpha
        drift_results[feat] = {
            "ks_stat": ks_stat,
            "p_value": p_value,
            "drifted": drifted
        }
        if drifted:
            any_drift = True

    return {
        "drift_detected": any_drift,
        "feature_metrics": drift_results
    }


def detect_concept_drift(
    ref_errors: np.ndarray, 
    cur_errors: np.ndarray, 
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Identifies concept drift by comparing model prediction error distributions (residuals/absolute errors).
    Uses a one-sided independent t-test to check if current errors are significantly greater.
    """
    t_stat, p_value = stats.ttest_ind(cur_errors, ref_errors, alternative="greater")
    drift_detected = p_value < alpha

    return {
        "drift_detected": bool(drift_detected),
        "t_stat": t_stat,
        "p_value": p_value
    }

# %% [markdown]
# ### Running Drift Tests on Our Datasets
# Let's train a baseline model and run these detectors.

# %%
# Train baseline model on reference data
baseline_model = LinearRegression()
X_ref = reference_df[["area_sqft", "bedrooms"]]
y_ref = reference_df["price_usd"]
baseline_model.fit(X_ref, y_ref)

ref_preds = baseline_model.predict(X_ref)
ref_errors = np.abs(y_ref - ref_preds)

# Evaluate Covariate Shift Data
cov_shift_res = detect_covariate_shift(reference_df, covariate_shift_df, ["area_sqft"])
cov_preds = baseline_model.predict(covariate_shift_df[["area_sqft", "bedrooms"]])
cov_errors = np.abs(covariate_shift_df["price_usd"] - cov_preds)
concept_shift_res_on_cov = detect_concept_drift(ref_errors, cov_errors)

# Evaluate Concept Drift Data
con_shift_res = detect_covariate_shift(reference_df, concept_drift_df, ["area_sqft"])
con_preds = baseline_model.predict(concept_drift_df[["area_sqft", "bedrooms"]])
con_errors = np.abs(concept_drift_df["price_usd"] - con_preds)
concept_drift_res = detect_concept_drift(ref_errors, con_errors)

print("🔍 Covariate Shift Scenario Evaluation:")
print(f"  Covariate Shift Detected: {cov_shift_res['drift_detected']} (p-val: {cov_shift_res['feature_metrics']['area_sqft']['p_value']:.4e})")
print(f"  Concept Drift Detected:   {concept_shift_res_on_cov['drift_detected']} (p-val: {concept_shift_res_on_cov['p_value']:.4f})")

print("\n🔍 Concept Drift Scenario Evaluation:")
print(f"  Covariate Shift Detected: {con_shift_res['drift_detected']} (p-val: {con_shift_res['feature_metrics']['area_sqft']['p_value']:.4f})")
print(f"  Concept Drift Detected:   {concept_drift_res['drift_detected']} (p-val: {concept_drift_res['p_value']:.4e})")

# %% [markdown]
# ## ⚡ 3. Automated Retraining Trigger
#
# If concept drift is detected, we programmatically trigger retraining and log the newly trained model to MLflow Model Registry.

# %%
def retrain_and_register_model(
    train_df: pd.DataFrame, 
    model_name: str
) -> Dict[str, Any]:
    """
    Fits a new linear regression model on new training data and registers it in MLflow.
    """
    X_train = train_df[["area_sqft", "bedrooms"]]
    y_train = train_df["price_usd"]
    
    with mlflow.start_run(run_name="automated_retraining") as run:
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        preds = model.predict(X_train)
        mae = mean_absolute_error(y_train, preds)
        r2 = r2_score(y_train, preds)
        
        # Log metrics to MLflow
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("train_mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # Log and Register model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        return {
            "run_id": run.info.run_id,
            "mae": mae,
            "r2": r2,
            "model_uri": model_info.model_uri,
            "model": model
        }

# %% [markdown]
# ## 👥 4. Human-in-the-Loop (HITL) Gate & Fallback Wrapping
#
# Even with automated retraining, we should prevent automatic deployment without oversight.
# We implement a simulated `hitl_approval_gate` that requires operator consent, and a `SafeServingWrapper` that falls back to a heuristic baseline model (`HeuristicBaselineModel`) if input parameters are out-of-distribution or if the active model has degraded.

# %%
def hitl_approval_gate(
    model_name: str, 
    old_mae: float, 
    new_mae: float, 
    force_approve: bool = False
) -> bool:
    """
    Simulated Human-in-the-Loop gating decision.
    """
    print(f"\n📢 [HITL GATING] Promotion request for model '{model_name}':")
    print(f"  Old Production MAE: ${old_mae:,.2f}")
    print(f"  Retrained Model MAE: ${new_mae:,.2f}")
    
    if force_approve:
        print("✅ HITL Decision: Automatically Approved (force_approve = True)")
        return True
        
    # Check if run in an interactive terminal (stdin is a tty)
    if sys.stdin.isatty():
        try:
            choice = input("👉 Do you approve promoting the retrained model to production? (y/N): ").strip().lower()
            approved = choice in ['y', 'yes']
        except Exception:
            approved = False
    else:
        # Fallback for non-interactive runner/tests
        print("⚠️ Non-interactive shell detected. Approving promotion based on performance improvements.")
        approved = new_mae < old_mae
        
    if approved:
        print("✅ HITL Decision: Approved! Promoting model to production serving.")
    else:
        print("❌ HITL Decision: Rejected. Fallback or baseline heuristic active.")
    return approved


class HeuristicBaselineModel:
    """
    Extremely simple, stable fallback model predicting price based on a fixed rate per sqft.
    """
    def __init__(self, price_per_sqft: float = 160.0):
        self.price_per_sqft = price_per_sqft
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["area_sqft"] * self.price_per_sqft).values


class SafeServingWrapper:
    """
    Wraps the active ML model, monitors input ranges (OOD detection), and delegates
    to a heuristic baseline model if anomalies are detected or if the model state is degraded.
    """
    def __init__(self, ml_model: Any, heuristic_model: HeuristicBaselineModel):
        self.ml_model = ml_model
        self.heuristic_model = heuristic_model
        self.fallback_active = False

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        for idx, row in X.iterrows():
            # Out of distribution check: Area must be positive and within reasonable limits
            area = row["area_sqft"]
            is_anomaly = (area < 400 or area > 10000 or pd.isna(area))
            
            if self.fallback_active or is_anomaly:
                # Fallback to heuristic
                fallback_pred = self.heuristic_model.predict(pd.DataFrame([row]))[0]
                predictions.append(fallback_pred)
            else:
                try:
                    ml_pred = self.ml_model.predict(pd.DataFrame([row]))[0]
                    predictions.append(ml_pred)
                except Exception:
                    # In case of runtime prediction error, gracefully fallback
                    fallback_pred = self.heuristic_model.predict(pd.DataFrame([row]))[0]
                    predictions.append(fallback_pred)
                    
        return np.array(predictions)

# %% [markdown]
# ### Putting it all together: The Continuous Training Loop
# Let's simulate a live scenario where concept drift occurs, is identified, and triggers the CT pipeline.

# %%
print("\n🎬 Simulating Production Concept Drift CT Loop:")

# 1. Check for concept drift on the concept drift dataset
drift_results = detect_concept_drift(ref_errors, con_errors)
if drift_results["drift_detected"]:
    print(f"🚨 ALERT: Concept Drift detected! p-value: {drift_results['p_value']:.4e}")
    print("⚡ Triggering automated retraining pipeline...")
    
    # 2. Retrain model on fresh production data
    retrain_res = retrain_and_register_model(concept_drift_df, MODEL_NAME)
    new_model = retrain_res["model"]
    new_mae = retrain_res["mae"]
    
    # 3. Request HITL Approval
    old_mae = mean_absolute_error(concept_drift_df["price_usd"], con_preds)
    approved = hitl_approval_gate(MODEL_NAME, old_mae, new_mae, force_approve=False)
    
    # 4. Wrap with serving guards
    heuristic = HeuristicBaselineModel(price_per_sqft=140.0)
    
    if approved:
        serving_model = SafeServingWrapper(new_model, heuristic)
        print("🚀 Serving Wrapper updated with retrained ML model.")
    else:
        # Fallback to baseline
        serving_model = SafeServingWrapper(baseline_model, heuristic)
        serving_model.fallback_active = True
        print("⚠️ Fallback serving wrapper active with Heuristic Baseline model.")
        
    # 5. Serve predictions on typical and anomalous requests
    sample_queries = pd.DataFrame([
        {"area_sqft": 1500.0, "bedrooms": 3},
        {"area_sqft": 15000.0, "bedrooms": 4},  # Out-of-bounds anomaly! Should trigger fallback
    ])
    
    preds = serving_model.predict(sample_queries)
    print(f"\n🔮 Served Predictions:")
    print(f"  Valid Request (1500 sqft):  ${preds[0]:,.2f}")
    print(f"  OOD Request (15000 sqft):    ${preds[1]:,.2f} (Heuristic Baseline Fallback)")

else:
    print("🟢 No Concept Drift detected. Serving continues normally.")

# %% [markdown]
# ## 🖥️ 6. Unified CLI Continuous Training Integration
# Module 15 extends our CLI with `mlops ct` to manage continuous retraining pipelines and operator approvals:
# * **Trigger retraining check locally via CLI:**
#   ```bash
#   uv run mlops ct trigger
#   ```
# * **Approve retrained model promotion via CLI:**
#   ```bash
#   uv run mlops ct approve
#   ```
# * **Run via Docker:**
#   ```bash
#   docker run --rm mlops-cli ct trigger
#   ```
#
# Let's verify the help information for continuous training operations on our unified CLI:

# %%
import subprocess
result = subprocess.run(["mlops", "ct", "--help"], capture_output=True, text=True)
print(result.stdout)

# %% [markdown]
# Now that we've set up continuous training, let's step into the LLMOps guide to explore generative AI pipelines!

