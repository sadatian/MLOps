# %% [markdown]
# # Model Monitoring & Data Drift Detection with `evidently`
#
# Once a model is running in production, its performance can degrade over time due to shifts in input data distributions (Data Drift) or shifts in target variables (Concept Drift).
# To identify these changes before they impact users, we run monitoring pipelines using **Evidently AI**.
#
# In this module, we will explore:
# 1. Generating Reference data (training baseline) and Current data (production inputs).
# 2. Simulating a statistical shift (Data Drift) in features.
# 3. Running an Evidently `Report` using the `DataDriftPreset`.
# 4. Exporting the metrics into a shareable HTML file.

# %%
import os
import pandas as pd
import numpy as np

# Import evidently components
from evidently import Report
from evidently.presets import DataDriftPreset

# %% [markdown]
# ## 1. Simulating Production Data Drift
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
# ## 2. Running the Drift Detection Report
# We instantiate a `Report` loaded with the `DataDriftPreset`. This runs statistical tests (such as Kolmogorov-Smirnov, chi-square, etc.) to determine if the differences between baseline and current distributions are statistically significant.

# %%
# Instantiate Report
report = Report(metrics=[
    DataDriftPreset()
])

print("\n🚀 Running Evidently statistical drift analysis...")
snapshot = report.run(reference_data=reference_df, current_data=current_df)

# %% [markdown]
# ## 3. Inspecting Results and Exporting Report
# We save the report output as an interactive HTML page.

# %%
output_html_path = "data/data_drift_report.html"
os.makedirs("data", exist_ok=True)

# Save Report as HTML
snapshot.save_html(output_html_path)

print(f"\n✅ Evidently Report successfully compiled and saved to: {output_html_path}")

# %% [markdown]
# ## 4. Viewing the Report
#
# To view your generated data drift dashboard:
#
# Open the HTML report in your browser:
# ```bash
# # On WSL/Linux, you can view the absolute file location or copy it:
# echo "File Path: file://$(pwd)/data/data_drift_report.html"
# ```
# This will display a gorgeous dashboard highlighting which features have drifted, the drift score, and visual comparisons of the distributions.
#
# ---
#
# 🎉 **Congratulations!** You have completed all 9 modules of the MLOps Depth curriculum.
# You now have hands-on, executable guides for:
# - Package synchronization (`uv`)
# - Interactive Python notebook compilation (`mkdocs-jupyter`)
# - Data versioning (`dvc`)
# - Cloud service simulation (`moto`)
# - Experiment tracking and registry (`mlflow`)
# - Machine learning pipelines (`dvc.yaml`)
# - REST API deployment (`fastapi`)
# - Containerization (`docker`)
# - Statistical model monitoring (`evidently`)
