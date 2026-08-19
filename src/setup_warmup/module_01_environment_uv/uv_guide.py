# %% [markdown]
# # ⚙️ Modern Python Environment & Dependency Control with `uv`
#
# This tutorial demonstrates how to use `uv`, an extremely fast Python package installer and resolver written in Rust.
# `uv` replaces `pip`, `pip-tools`, `virtualenv`, and `poetry` in unified projects, improving installation speed and dependency resolution times by up to 10-100x.
#
# ### How `uv` Works
#
# Modern development teams face challenges with slow virtual environment creation and duplicate storage consumption. Traditional `pip` downloads and installs duplicate packages separately for every virtual environment.
# `uv` changes this by utilizing a **global content-addressable cache** and leveraging **hard links or symlinks** where supported by the file system. 
#
# ```mermaid
#  graph TD
#      subgraph traditional_workflow_slow_redundant ["Traditional Workflow (slow, redundant)"]
#          A["Project 1 venv"] -->|"Download & Build"| B["PyPI Package A"]
#          C["Project 2 venv"] -->|"Download & Build"| D["PyPI Package A (Duplicate)"]
#          B -->|"Write full copy"| E["Disk Space Used"]
#          D -->|"Write full copy"| E
#      end
#
#      subgraph rust_optimized_workflow_uv ["Rust-Optimized Workflow (uv)"]
#          F["PyPI Registry"] -->|"Rust Resolver"| G["Centralized uv Cache"]
#          G -->|"Ref-linked / Hard-linked"| H["Project 1 venv"]
#          G -->|"Ref-linked / Hard-linked"| I["Project 2 venv"]
#          H -->|"Zero Disk Overhead"| J["Optimized Disk Space"]
#          I -->|"Zero Disk Overhead"| J
#      end
#
#      A ~~~ G
# ```
#
# In this module, we will explore:
# 1. Inspecting the project environment configuration (`pyproject.toml`).
# 2. Programmatically verifying installed dependencies.
# 3. Synchronizing virtual environments.


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

import platform

print("=== Python Environment Diagnostics ===")
print(f"Python Executable: {sys.executable}")
print(f"Python Version:    {sys.version}")
print(f"OS Platform:       {platform.system()} ({platform.release()})")
print("======================================")

# %% [markdown]
# ## 📦 1. Package Verification
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
# ## 🛠️ 2. How to Work with `uv`
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
# ### 🖥️ 3. Introducing the `mlops` CLI
# To unify our operations, we have introduced a project-wide `mlops` CLI.
# The first command we initiate is `mlops status` (or `mlops diagnose`), which verifies environment versions and crucial dependencies.
#
# * **Run CLI status locally:**
#   ```bash
#   uv run mlops status
#   ```
# * **Run CLI status via Docker:**
#   ```bash
#   docker run --rm -it mlops-cli status
#   ```
#
# Let's run this diagnostics check programmatically:

# %%
import subprocess
result = subprocess.run(["mlops", "status"], capture_output=True, text=True)
print(result.stdout)

# %% [markdown]
# Now that we've verified our python environment and unified CLI entry point, let's move to the Notebook Documentation guide to understand how percent-style cells (`#%%`) are converted into rich docs!
