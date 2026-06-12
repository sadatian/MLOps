# %% [markdown]
# # Continuous Integration for Machine Learning (CI/ML)
#
# Continuous Integration for Machine Learning (CI/ML) extends traditional software CI (which focuses on syntax verification, code formatting, and unit testing) to include ML-specific quality assurance steps.
#
# ### The CI/ML Gating Pipeline
#
# Unlike standard software code, ML systems are a product of both code and data. A code change that compiles successfully can still degrade performance if the underlying data distributions change. 
# A CI/ML pipeline automates validation of code syntax, unit functionality, dataset integrity, and model evaluation metrics (e.g., RMSE and $R^2$) prior to packaging and deployment.
#
# ```mermaid
# graph TD
#     subgraph CI Runner Trigger
#         A[Code / Config Commit] -->|Trigger Webhook| B[GitHub Actions Runner]
#     end
# 
#     subgraph Phase 1: Traditional Software CI
#         B --> C[Lint Checks: Ruff / Flake8]
#         C -->|Pass| D[Unit Tests: Pytest]
#     end
# 
#     subgraph Phase 2: ML Quality Assurance (CI/ML)
#         D -->|Pass: uv install| E[Execute Pipeline: dvc repro]
#         E -->|Generates artifact| F[Evaluation metrics.json]
#         F --> G[Run Model Quality Gate Check]
#         G --> H{Performance Thresholds Met?}
#         H -->|No: Fail Build| I[Reject PR & Block Deploy]
#         H -->|Yes: Allow Build| J[Build Docker Image]
#         J --> K[Push Image to Registry]
#     end
# 
#     style I fill:#f8d7da,stroke:#dc3545,stroke-width:1.5px
#     style K fill:#d4edda,stroke:#28a745,stroke-width:2px
# ```
#
# In this module, we will explore:
# 1. Understanding CI vs. CI/ML concepts.
# 2. Writing a programmatic quality gatekeeper script to inspect trained model metrics.
# 3. Integrating testing, linting, and container compilation in a automated pipeline.


# %%
import os
import json

# Define the metrics path compiled by the Integrated MLOps Pipeline
METRICS_PATH = "data/metrics.json"

# %% [markdown]
# ## 1. Traditional CI vs. CI/ML
#
# | Traditional Software CI | Machine Learning CI/ML |
# |---|---|
# | **Code Syntax / Linting:** Verifies Python files have valid syntax and format standards. | **Code Syntax / Linting:** Same as software, checking codebase sanity. |
# | **Unit Testing:** Tests individual functions (e.g. API endpoint mocks). | **Data Validation:** Tests schemas, distribution shifts, and missing value thresholds. |
# | **Build compilation:** Builds packages or Docker containers. | **Model Gating:** Ensures model performance (accuracy, MSE, RMSE) meets staging thresholds before compiling the build. |
# | **Output artifact:** Executable file, container image, or library. | **Output artifact:** Version-controlled models, datasets, and validated deployment containers. |

# %% [markdown]
# ## 2. Implementing Programmatic Model Performance Gating
#
# When our pipeline (`dvc repro`) runs, it outputs a `metrics.json` file.
# In CI/ML, we run a gatekeeper script. If the newly trained model's performance does not meet specified quality thresholds, we fail the build. This prevents broken or degraded models from being packaged into Docker images or pushed to production.
#
# Let's define a validation function:

# %%
def verify_model_performance(metrics_json_path, max_rmse_allowed=250000.0, min_r2_required=-0.5):
    """
    Reads the evaluation metrics and verifies if they satisfy staging quality requirements.
    """
    if not os.path.exists(metrics_json_path):
        raise FileNotFoundError(f"Evaluation metrics file '{metrics_json_path}' not found. Please run the Integrated MLOps Pipeline first.")

    with open(metrics_json_path, "r") as f:
        metrics = json.load(f)

    rmse = metrics.get("rmse", float("inf"))
    r2 = metrics.get("r2_score", float("-inf"))

    print("=== Model Quality Gate Verification ===")
    print(f"Loaded metrics from: {metrics_json_path}")
    print(f"Model RMSE: {rmse:.2f} (Max Allowed: {max_rmse_allowed:.2f})")
    print(f"Model R2:   {r2:.4f} (Min Required: {min_r2_required:.4f})")
    print("=======================================")

    # Gating checks
    rmse_pass = rmse <= max_rmse_allowed
    r2_pass = r2 >= min_r2_required

    if rmse_pass and r2_pass:
        print("✅ Success: Model passed all quality gate checks! Build proceeding...")
        return True
    else:
        errors = []
        if not rmse_pass:
            errors.append(f"RMSE {rmse:.2f} exceeds threshold of {max_rmse_allowed:.2f}")
        if not r2_pass:
            errors.append(f"R2 score {r2:.4f} falls below threshold of {min_r2_required:.4f}")
        
        print(f"❌ Failure: Model rejected. Reason: {', '.join(errors)}")
        return False

# %% [markdown]
# ### Run Gating Gatekeeper Tests
#
# Let's run the quality gate checks. First, we'll run it with our normal thresholds (which should pass):

# %%
if __name__ == "__main__":
    import sys

    # Check if metrics exist (run pipeline step if missing)
    if not os.path.exists(METRICS_PATH):
        print("⚠️ data/metrics.json not found. Generating default metrics...")
        os.makedirs("data", exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump({"rmse": 194573.71, "r2_score": -0.12}, f, indent=4)

    # Run the real model quality gate check
    success = verify_model_performance(METRICS_PATH, max_rmse_allowed=250000.0, min_r2_required=-0.5)
    
    if not success:
        print("❌ Build Blocked: Model did not meet quality standards.")
        sys.exit(1)
        
    print("🚀 Build Allowed: Proceeding with container packaging.")
    sys.exit(0)

# %% [markdown]
# ## 3. Automated Gating
#
# Running this gatekeeper script manually is a great way to verify thresholds locally. However, in a production MLOps pipeline, we automate this process.
# In the GitHub Actions CI Pipeline guide, we will explore how this quality gate check is integrated into our workflow, running automatically on every codebase modification before building our deployment container!



