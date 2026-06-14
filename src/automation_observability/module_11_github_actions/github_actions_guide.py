# %% [markdown]
# # 🤖 GitHub Actions CI Pipeline for MLOps
#
# Continuous Integration (CI) requires a central server to run checks on every codebase modification. In this module, we will explore the structure of our GitHub Actions workflow file (`.github/workflows/ci.yml`) and understand how the pipeline orchestrates linting, testing, and containerization.
#
# ### The CI/CD Runner Orchestration
#
# CI/CD systems automate quality control by scheduling independent task agents (Runners) that spawn on isolated machines (Docker containers or VMs). 
# On each trigger (such as a commit push or PR open), the runner spins up, installs dependencies, executes pipeline steps (like code checks, model quality evaluations, and test cases), and compiles the deployment assets.
#
# ```mermaid
#  graph TD
#      subgraph trigger_events ["Trigger Events"]
#          A["Code Push / PR to main"] -->|"GitHub Webhook"| B["Queue Job"]
#      end
#
#      subgraph github_actions_runner_vm ["GitHub Actions Runner (VM)"]
#          B -->|"Allocate Runner"| C["Checkout Code"]
#          C -->|"astral-sh/setup-uv"| D["Setup uv Dependency Cache"]
#          D -->|"Compile Python files"| E["Syntax Verification"]
#
#          E -->|"Pass"| F["DVC Repro: Run prep/train/eval DAG"]
#          F -->|"Generates metrics.json"| G["Run Gate Check: ci_ml_guide.py"]
#
#          G -->|"Pass"| H["Run pytest test suite"]
#          H -->|"Pass"| I["docker build -t app:latest ."]
#
#          I -->|"Pass"| J["CI Success Checkmark"]
#      end
#
#      style E fill:#fff3e0,stroke:#ffb74d,stroke-width:1px
#      style G fill:#fff3e0,stroke:#ffb74d,stroke-width:1px
#      style J fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
#
# ```
#

# ## 📋 1. Anatomy of a GitHub Actions Workflow
#
# A GitHub Actions workflow is defined in a YAML configuration file inside the `.github/workflows/` directory. Here is the structure of our workflow:
#
# *   **Name:** Defines the title of the workflow shown in the GitHub actions UI.
#     ```yaml
#     name: Continuous Integration
#     ```
# *   **Triggers (`on`):** Defines what events trigger the pipeline. We configure it to run on pushes and pull requests targeting the `main` branch:
#     ```yaml
#     on:
#       push:
#         branches: [ main ]
#       pull_request:
#         branches: [ main ]
#     ```
# *   **Jobs & Runners (`runs-on`):** Defines the jobs to execute and the virtual operating system they run on. We use the latest Ubuntu runner:
#     ```yaml
#     jobs:
#       lint-and-test:
#         runs-on: ubuntu-latest
#     ```

# %% [markdown]
# ## 🪜 2. Core CI Stages and Steps
#
# Within our job, we execute a sequence of actions and commands to build and verify our MLOps system:
#
# ### Step 1: Checkout & Environment Setup
# We check out the source repository code, set up the lightning-fast Python package resolver `uv`, and synchronize project dependencies:
# ```yaml
# - name: Checkout Code
#   uses: actions/checkout@v4
#
# - name: Setup uv
#   uses: astral-sh/setup-uv@v5
#   with:
#     enable-cache: true
#     version: "latest"
#
# - name: Setup Python
#   uses: actions/setup-python@v5
#   with:
#     python-version-file: ".python-version"
#
# - name: Install Dependencies
#   run: uv sync --frozen
# ```
#
# ### Step 2: Syntax and Lint Verification
# Before running tests, we ensure that the codebase has no syntax issues:
# ```yaml
# - name: Syntax and Lint Verification
#   run: uv run python -m compileall src/ tests/
# ```
#
# ### Step 3: Reproduce Pipeline (DVC)
# We generate synthetic training data and run `dvc repro`. This builds the end-to-end pipeline (data preparation -> training -> evaluation) and outputs our model binary (`data/model.pkl`) and performance metrics (`data/metrics.json`):
# ```yaml
# - name: Generate Raw Data and Initialize DVC
#   run: |
#     git config --global user.email "ci@example.com"
#     git config --global user.name "CI Runner"
#     uv run python src/data_experimentation/module_04_data_versioning_dvc/dvc_guide.py
#
# - name: Execute End-to-End Pipeline
#   run: uv run dvc repro
# ```
#
# ### Step 4: Model Quality Gating
# We execute the gatekeeper check from the CI/ML Quality Gates guide to ensure the newly trained model satisfies performance staging requirements:
# ```yaml
# - name: Verify Model Performance Quality Gate
#   run: uv run python src/automation_observability/module_10_ci_ml_automation/ci_ml_guide.py
# ```
#
# ### Step 5: Test Execution & Container Compilation
# We run our Pytest suite (using mocked dependencies for serving APIs and decoupled MLflow file stores) and build the Docker deployment image to ensure it compiles without failures:
# ```yaml
# - name: Run Test Suite
#   env:
#     MLFLOW_ALLOW_FILE_STORE: "true"
#   run: uv run pytest -v
#
# - name: Build Docker Container
#   run: docker build -t mlops-housing-service:latest .
# ```

# %%
# Let's verify that the CI workflow config file exists and print its header structure
import os

workflow_path = ".github/workflows/ci.yml"
if os.path.exists(workflow_path):
    print(f"✅ GitHub Actions workflow configuration found at: {workflow_path}\n")
    with open(workflow_path, "r") as f:
        # Read first 15 lines of the YAML file
        for _ in range(15):
            line = f.readline()
            if not line:
                break
            print(line.rstrip())
else:
    print(f"❌ Workflow configuration was not found at: {workflow_path}")

# %% [markdown]
# ## 🐳 3. Testing Workflows Locally (Tip)
#
# Instead of pushing to GitHub to trigger Actions every time you edit your workflow, you can test workflows locally on your WSL/Linux workstation using **`act`** (which runs GitHub Actions inside local Docker containers):
#
# ```bash
# # Install act on Ubuntu/WSL:
# curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
#
# # Run the CI workflow locally:
# act
# ```
#
# 🎉 **Congratulations!** You have completed the entire MLOps lifecycle curriculum!
