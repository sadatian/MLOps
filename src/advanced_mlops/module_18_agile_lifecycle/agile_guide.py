# %% [markdown]
# # 📊 Agile MLOps Lifecycle & Heuristic Baselines
#
# In modern MLOps development, starting with a complex machine learning model (like Random Forests or Deep Neural Networks) in "Sprint 1" is often a anti-pattern.
#
# Instead, the Agile MLOps Lifecycle recommends starting with a **Sprint 0** heuristic baseline.
#
# ### Heuristic baseline design patterns
#
# A heuristic baseline is a hard-coded or simple statistic-based rule (e.g., pricing a house at a flat \$150 per sqft + \$50k base). 
# This serves three primary purposes:
# 1. **End-to-End Pipeline Validation:** Ensuring that database ingestion, serialization, API servers, and monitoring are fully connected before introducing training noise.
# 2. **Lower Bound Benchmark:** Setting a strict minimum quality target. If a complex model cannot outperform a basic rule, it should not be built.
# 3. **Instant Business Value:** Providing an immediate, fallback predictor that API clients can invoke on day one.
#
# ```mermaid
#  graph TD
#      A["Sprint 0: Define Heuristics"] --> B["Build & Deploy API with Baseline"]
#      B --> C["Establish Telemetry & Monitoring"]
#      C --> D["Sprint 1: Train Complex ML Model"]
#      D --> E{"Outperforms Baseline?"}
#      E -->|"Yes"| F["Promote to Active Serving"]
#      E -->|"No"| G["Keep Baseline / Investigate"]
# ```
#
# In this module, we will explore:
# 1. Structuring "Sprint 0" and managing Agile-ML scope creep.
# 2. Implementing a simple heuristic baseline model.
# 3. Utilizing our unified CLI to run heuristic predictions.
#
# ---
#
# ## 🖥️ 1. Heuristic Baseline Execution via CLI
#
# Module 18 extends our unified CLI with the `mlops baseline` command:
# * **Execute heuristic baseline locally:**
#   ```bash
#   uv run mlops baseline run --input data/housing_raw.csv --output data/baseline_predictions.csv
#   ```
# * **Run via Docker:**
#   ```bash
#   docker run --rm -v $(pwd)/data:/app/data mlops-cli baseline run
#   ```
#
# Let's run this baseline check programmatically:

# %%
import subprocess
import os
import pandas as pd

# Generate default dataset if missing
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/housing_raw.csv"):
    pd.DataFrame({
        "area_sqft": [1500, 2000, 2500],
        "bedrooms": [3, 4, 4],
        "price_usd": [275000, 350000, 425000]
    }).to_csv("data/housing_raw.csv", index=False)

# Run baseline command
result = subprocess.run(
    ["mlops", "baseline", "run", "--input", "data/housing_raw.csv", "--output", "data/baseline_predictions.csv"],
    capture_output=True,
    text=True
)
print(result.stdout)

# Verify outputs
if os.path.exists("data/baseline_predictions.csv"):
    print("📋 Baseline Predictions Sample:")
    df = pd.read_csv("data/baseline_predictions.csv")
    print(df)
else:
    print("❌ Failed to output baseline predictions.")

# %% [markdown]
# Let's check the help menu for baseline operations:

# %%
result = subprocess.run(["mlops", "baseline", "--help"], capture_output=True, text=True)
print(result.stdout)

# %% [markdown]
# ---
#
# 🎉 **Module 18 Completed!** You have successfully implemented heuristic baselines to benchmark MLOps architectures and ensure reliable Sprint 0 operations!
