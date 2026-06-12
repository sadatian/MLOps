# %% [markdown]
# # Modern Python Environment & Dependency Control with `uv`
#
# This tutorial demonstrates how to use `uv`, an extremely fast Python package installer and resolver written in Rust.
# In this module, we will explore:
# 1. Inspecting the project environment configuration (`pyproject.toml`).
# 2. Programmatically verifying installed dependencies.
# 3. Synchronizing virtual environments.

# %%
import os
import sys
import platform

print("=== Python Environment Diagnostics ===")
print(f"Python Executable: {sys.executable}")
print(f"Python Version:    {sys.version}")
print(f"OS Platform:       {platform.system()} ({platform.release()})")
print("======================================")

# %% [markdown]
# ## 1. Package Verification
# Let's inspect some of the core dependencies we installed using `uv` (like DVC, MLflow, FastAPI, and Moto) to confirm they are accessible in our virtual environment.

# %%
try:
    import dvc
    print(f"✅ DVC is installed, version: {dvc.__version__}")
except ImportError:
    print("❌ DVC is NOT installed.")

try:
    import mlflow
    print(f"✅ MLflow is installed, version: {mlflow.__version__}")
except ImportError:
    print("❌ MLflow is NOT installed.")

try:
    import fastapi
    print(f"✅ FastAPI is installed, version: {fastapi.__version__}")
except ImportError:
    print("❌ FastAPI is NOT installed.")

try:
    import moto
    print(f"✅ Moto (AWS mock service) is installed, version: {moto.__version__}")
except ImportError:
    print("❌ Moto is NOT installed.")

# %% [markdown]
# ## 2. How to Work with `uv`
#
# Here is a quick reference guide of commands to run in your terminal:
#
# *   **Initialize a new project:**
#     ```bash
#     uv init
#     ```
# *   **Add dependencies:**
#     ```bash
#     uv add pandas scikit-learn
#     ```
# *   **Remove dependencies:**
#     ```bash
#     uv remove pandas
#     ```
# *   **Sync project virtual environment:**
#     ```bash
#     uv sync
#     ```
# *   **Run scripts inside the virtual environment:**
#     ```bash
#     uv run python main.py
#     ```
#
# Now that we've verified our python environment, let's move to the Notebook Documentation guide to understand how percent-style cells (`#%%`) are converted into rich docs!
