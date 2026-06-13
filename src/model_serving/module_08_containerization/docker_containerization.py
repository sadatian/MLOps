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
# graph TD
#     subgraph Builder Stage (Heavy Toolchain)
#         A[Base: astral-sh/uv:python3.12-alpine] -->|COPY pyproject.toml & uv.lock| B[Copy Package List]
#         B -->|uv sync --frozen --no-dev| C[Compile Binary Wheels]
#         C -->|Output| D[Isolated .venv]
#     end
# 
#     subgraph Runtime Stage (Slim Image)
#         E[Base: python:3.12-alpine] -->|COPY --from=builder /app/.venv| F[Import Clean Runtime .venv]
#         G[Host: code & data/model.pkl] -->|COPY src/ & COPY model.pkl| H[Inject Application Assets]
#         F --> H
#         H -->|Define ENTRYPOINT| I[Exposed Port 8000 & CMD Uvicorn]
#         I -->|Package Output| J[Production Container Image]
#     end
# 
#     style D fill:#fff9c4,stroke:#fbc02d,stroke-width:1.5px
#     style J fill:#d4edda,stroke:#28a745,stroke-width:2px
# ```
#
# In this module, we will explore:
# 1. Understanding a multi-stage Docker build for python applications.
# 2. Writing a production-grade `Dockerfile`.
# 3. Building, running, and inspecting the docker container.


# %%
import os

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
# ## 🛠️ 2. How to Compile and Run Your Container
# Once the Dockerfile is ready, compile and test it using standard Docker commands in your terminal:
#
# ### Step 1: Build the Image
# Run this command in the project root folder (ensure you run the Integrated MLOps Pipeline first to create the `model.pkl` binary):
# ```bash
# docker build -t mlops-housing-service:v1 .
# ```
#
# ### Step 2: Start the Container
# Spin up the server mapping host port 8000 to container port 8000:
# ```bash
# docker run -d -p 8000:8000 --name housing-api mlops-housing-service:v1
# ```
#
# ### Step 3: Test Container Endpoints
# Check health and predict status:
# ```bash
# curl http://localhost:8000/health
#
# curl -X POST http://localhost:8000/predict \\
#      -H "Content-Type: application/json" \\
#      -d '{"area_sqft": 2000.0, "bedrooms": 4}'
# ```
#
# ### Step 4: Cleanup
# Stop and delete the container:
# ```bash
# docker stop housing-api
# docker rm housing-api
# ```
#
# Now that we know how to containerize the service, let's step into the Model Monitoring guide to set up monitoring and drift detection!
