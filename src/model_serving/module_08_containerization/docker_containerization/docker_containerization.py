# %% [markdown]
# # 🐳 Model Deployment & Containerization with Docker
#
# Deploying APIs directly onto bare VMs leads to "works on my machine" failures. 
# Containerization using **Docker** ensures that our python dependencies, OS-level binaries, environment variables, and model weights are packaged into a single, identical, and immutable image that runs identically anywhere.
#
# ### The Multi-Stage Docker Compilation Flow
#
# To maintain security and minimize deployment costs, production Docker containers must be as small as possible. 
# We achieve this using a **multi-stage build**:
# 1. **Builder Stage:** Uses a heavy base image containing compiler tools and the fast `uv` installer to compile dependencies into a `.venv`.
# 2. **Runner Stage:** Uses a stripped-down Alpine Linux runner, importing *only* the compiled virtual environment from the builder stage without any dev dependencies or compiler tools.
#
# ```mermaid
#  graph TD
#      subgraph builder_stage_heavy_toolchain ["Builder Stage (Heavy Toolchain)"]
#          A["Base: astral-sh/uv:python3.12-alpine"] -->|"COPY pyproject.toml & uv.lock"| B["Copy Package List"]
#          B -->|"uv sync --frozen --no-dev"| C["Compile Binary Wheels"]
#          C -->|"Output"| D["Isolated .venv"]
#      end
#
#      subgraph runtime_stage_slim_image ["Runtime Stage (Slim Image)"]
#          E["Base: python:3.12-alpine"] -->|"COPY --from=builder /app/.venv"| F["Import Clean Runtime .venv"]
#          G["Host: code & data/model.pkl"] -->|"COPY src/ & COPY model.pkl"| H["Inject Application Assets"]
#          F --> H
#          H -->|"Define ENTRYPOINT"| I["Exposed Port 8000 & CMD Uvicorn"]
#          I -->|"Package Output"| J["Production Container Image"]
#      end
#
#      style D fill:#fff9c4,stroke:#fbc02d,stroke-width:1.5px
#      style J fill:#d4edda,stroke:#28a745,stroke-width:2px
#
# ```
#
# In this module, we will explore:
# 1. Understanding a multi-stage Docker build for python applications.
# 2. Writing a production-grade `Dockerfile`.
# 3. Building, running, and inspecting the docker container.


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

# %% [markdown]
# ## 🐳 1. Production Dockerfile Specifications
# Below is the structure of a professional, lightweight `Dockerfile` configured to run our FastAPI service.
#
# The `Dockerfile` has been pre-created at the root of the project. Let's review its configuration:
#
# ```dockerfile
# # Build Stage: Prepare the environment and install dependencies
# FROM ghcr.io/astral-sh/uv:python3.12-alpine AS builder
# WORKDIR /app
# ENV UV_COMPILE_BYTECODE=1
# COPY pyproject.toml uv.lock ./
# RUN uv sync --frozen --no-install-project --no-dev
#
# # Run Stage: Minimal runtime environment with no build tools
# FROM python:3.12-alpine
# WORKDIR /app
# COPY --from=builder /app/.venv /app/.venv
# ENV PATH="/app/.venv/bin:$PATH"
# COPY data/model.pkl ./data/model.pkl
# COPY src/model_serving/module_07_model_serving/serve_api.py ./src/model_serving/module_07_model_serving/serve_api.py
# EXPOSE 8000
# ENV HOST=0.0.0.0
# ENV PORT=8000
# CMD ["uvicorn", "src.model_serving.module_07_model_serving.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
# ```
#
# Let's verify that this file is configured properly in the root directory.

# %%
if os.path.exists("Dockerfile"):
    print("✅ Dockerfile exists in the root directory.")
else:
    print("❌ Dockerfile was not found.")

# %% [markdown]
# ## 🛠️ 2. How to Compile and Run Your CLI Container
# Module 8 extends the CLI by introducing `mlops container` to manage container compilation and execution:
#
# ### Step 1: Build the Image via CLI
# Build the Docker container packaging the CLI:
# ```bash
# uv run mlops container build --tag mlops-cli
# ```
#
# ### Step 2: Start the Container via CLI
# Spin up the server:
# ```bash
# uv run mlops container run --tag mlops-cli --port 8000
# ```
#
# ### Step 3: Test Container Endpoints
# Check health and predict status:
# ```bash
# curl http://localhost:8000/health
#
# curl -X POST http://localhost:8000/predict \
#      -H "Content-Type: application/json" \
#      -d '{"area_sqft": 2000.0, "bedrooms": 4}'
# ```
#
# ### Step 4: Run any CLI command directly inside Docker!
# Since the Docker entrypoint is configured to call `mlops`, you can execute any sub-command in isolation:
# ```bash
# docker run --rm -it mlops-cli status
# docker run --rm -it mlops-cli moto s3 -p 5001
# ```
#
# Let's verify the help information for container operations on our unified CLI:

# %%
import subprocess
result = subprocess.run(["mlops", "container", "--help"], capture_output=True, text=True)
print(result.stdout)

# %% [markdown]
# ## 🌐 3. Multi-Model Serving & CLI Operations
#
# Our unified CLI allows us to train different candidate models and serve them concurrently from the same Docker container using custom endpoint routing.
#
# ### Step 1: Train the Random Forest Model
# We use the pipeline command to split the dataset and train the default Random Forest model:
# ```bash
# uv run mlops pipeline run --model random_forest
# ```
# This outputs the model parameters and serializes weights to `data/model.pkl`.
#
# ### Step 2: Train the Linear Regression Model
# Next, we run the pipeline specifically specifying the Linear Regression candidate:
# ```bash
# uv run mlops pipeline run --model linear_regression
# ```
# This executes data prep, fits a linear regression model, and serializes weights to `data/model_linear.pkl`.
#
# ### Step 3: Run the Serving Container on Port 8080
# Boot up the containerized CLI to expose the FastAPI server:
# ```bash
# docker run -d --name mlops-multi-model -p 8080:8000 mlops-cli serve
# ```
#
# ### Step 4: Query the Models via FastAPI Endpoints
#
# You can now request predictions from the Random Forest, Linear Regression, or Heuristic Baseline model using different URI flags:
#
# *   **Random Forest Model (Default / random_forest):**
#     ```bash
#     curl -X POST http://localhost:8080/predict/random_forest \
#          -H "Content-Type: application/json" \
#          -d '{"area_sqft": 1850.0, "bedrooms": 3}'
#     ```
#
# *   **Linear Regression Model (linear_regression):**
#     ```bash
#     curl -X POST http://localhost:8080/predict/linear_regression \
#          -H "Content-Type: application/json" \
#          -d '{"area_sqft": 1850.0, "bedrooms": 3}'
#     ```
#
# *   **Heuristic Baseline Model (heuristic):**
#     ```bash
#     curl -X POST http://localhost:8080/predict/heuristic \
#          -H "Content-Type: application/json" \
#          -d '{"area_sqft": 1850.0, "bedrooms": 3}'
#     ```
#
# Now that we know how to serve multiple models from our containerized service, let's step into the Model Monitoring guide to set up monitoring and drift detection!
