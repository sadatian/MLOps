# %% [markdown]
# # 📈 Model Monitoring & Data Drift Detection with `evidently`
#
# Once a model is running in production, its performance can degrade over time due to shifts in input data distributions (Data Drift) or shifts in target variables (Concept Drift).
# To identify these changes before they impact users, we run monitoring pipelines using **Evidently AI**.
#
# ### The Model Monitoring & Drift Feedback Loop
#
# Data drift occurs when the input distribution shifts ($P(X) \neq P'(X)$), making training baseline assumptions obsolete. 
# Evidently AI processes current production batches against historical baseline/reference datasets to run statistical hypothesis tests (e.g., Kolmogorov-Smirnov, Wasserstein distance, Chi-Square). If critical features cross significance thresholds, it triggers alerts or retraining.
#
# ```mermaid
#  graph TD
#      subgraph production_environment ["Production Environment"]
#          A["Incoming Inference Data"] -->|"Log features"| B["(Live Database)"]
#      end
#
#      subgraph baseline_registry ["Baseline Registry"]
#          C["(Reference / Training Dataset)"]
#      end
#
#      subgraph evidently_ai_analyzer ["Evidently AI Analyzer"]
#          B -->|"Query Current Batch"| D["DataDriftPreset Reporter"]
#          C -->|"Query Reference Batch"| D
#          D -->|"Run Statistical Tests"| E{"Drift Score > Threshold?"}
#          E -->|"Yes: Statistical Shift"| F["Trigger Retraining Trigger"]
#          E -->|"No: Stable"| G["Continue API serving"]
#          D -->|"Compile Dashboard"| H["data_drift_report.html"]
#      end
#
#      style F fill:#f8d7da,stroke:#dc3545,stroke-width:1.5px
#      style G fill:#d4edda,stroke:#28a745,stroke-width:1.5px
#
# ```
#
# In this module, we will explore:
# 1. Generating Reference data (training baseline) and Current data (production inputs).
# 2. Simulating a statistical shift (Data Drift) in features.
# 3. Running an Evidently `Report` using the `DataDriftPreset`.
# 4. Exporting the metrics into a shareable HTML file.


# %%
# Ensure we run from the project root directory
import os
import sys

# Locate project root (searching upwards for mkdocs.yml)
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "mkdocs.yml")):
        break
    current_dir = os.path.dirname(current_dir)

if os.path.exists(os.path.join(current_dir, "mkdocs.yml")):
    os.chdir(current_dir)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
else:
    print("⚠️ Could not find project root containing mkdocs.yml.")

import pandas as pd
import numpy as np

# Import evidently components
from evidently import Report
from evidently.presets import DataDriftPreset

# %% [markdown]
# ## 📉 1. Simulating Production Data Drift
# Let's generate a Reference dataset (historical baseline) and a Current dataset representing production inputs where houses are suddenly much larger (e.g. sqft values shift upward).

# %%
np.random.seed(42)
num_samples = 200

# 1. Reference Data (Normal Distribution)
reference_df = pd.DataFrame({
    "area_sqft": np.random.normal(1800, 300, size=num_samples),
    "bedrooms": np.random.choice([2, 3, 4, 5], p=[0.4, 0.4, 0.15, 0.05], size=num_samples),
    "price_usd": np.random.normal(300000, 50000, size=num_samples)
})

# 2. Current Data (Simulated Data Drift: Mean sqft increased, bedrooms count shifted)
current_df = pd.DataFrame({
    "area_sqft": np.random.normal(2300, 400, size=num_samples), # Significant shift in mean
    "bedrooms": np.random.choice([2, 3, 4, 5], p=[0.05, 0.15, 0.4, 0.4], size=num_samples), # Shifted distribution
    "price_usd": np.random.normal(420000, 70000, size=num_samples)
})

print("📊 Data generation complete.")
print(f"Reference Area Mean: {reference_df['area_sqft'].mean():.2f} sqft")
print(f"Current Area Mean:   {current_df['area_sqft'].mean():.2f} sqft (Simulating Shift)")

# %% [markdown]
# ## 📊 2. Running the Drift Detection Report
# We instantiate a `Report` loaded with the `DataDriftPreset`. This runs statistical tests (such as Kolmogorov-Smirnov, chi-square, etc.) to determine if the differences between baseline and current distributions are statistically significant.

# %%
# Instantiate Report
report = Report(metrics=[
    DataDriftPreset()
])

print("\n🚀 Running Evidently statistical drift analysis...")
snapshot = report.run(reference_data=reference_df, current_data=current_df)

# %% [markdown]
# ## 📁 3. Inspecting Results and Exporting Report
# We save the report output as an interactive HTML page.

# %%
output_html_path = "data/data_drift_report.html"
os.makedirs("data", exist_ok=True)

# Save Report as HTML
snapshot.save_html(output_html_path)

print(f"\n✅ Evidently Report successfully compiled and saved to: {output_html_path}")

# %% [markdown]
# ## 🖥️ 4. Unified CLI Monitoring
# Module 9 extends the CLI by introducing `mlops monitor`:
# * **Run drift analysis locally via CLI:**
#   ```bash
#   uv run mlops monitor drift -o data/data_drift_report.html
#   ```
# * **Run via Docker:**
#   ```bash
#   docker run --rm -v $(pwd)/data:/app/data mlops-cli monitor drift -o data/data_drift_report.html
#   ```
#
# Let's verify the help information for monitoring operations on our unified CLI:

# %%
import subprocess
result = subprocess.run(["mlops", "monitor", "--help"], capture_output=True, text=True)
print(result.stdout)

# %% [markdown]
# Now that we've set up monitoring, let's step into the CI/ML Quality Gates guide to learn about gating checks!

