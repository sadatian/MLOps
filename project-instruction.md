# MLOps Depth Tutorials: Project Instructions & Roadmap

Welcome to the MLOps tutorial workspace. This document outlines the roadmap, coding standards, mocking protocols, and agent rules for this learning sandbox.

---

## 1. Project Goal & Design Philosophy

The purpose of this project is to provide a step-by-step, highly hands-on environment for learning modern MLOps. 
Rather than reading static text or clicking through Jupyter notebooks (`.ipynb`), we write **interactive, celled Python files (`.py`)** utilizing the `# %%` syntax (standard percent format). 

These scripts serve two purposes:
1. **Interactive Execution:** Users and IDEs can run them cell-by-cell in an interactive window.
2. **Auto-compiled Documentation:** `mkdocs` compiles these files into HTML notebooks at build time using the `mkdocs-jupyter` plugin.

---

## 2. Core Technical Guidelines

### 2.1 Dependency Management
- All Python packages must be managed using `uv`.
- Use `uv add <package>` to introduce new tools. Do not modify dependencies manually unless syncing lockfiles immediately after.

### 2.2 Python Script cell formatting (`# %%`)
- Always use standard Jupytext percent format for scripts.
- Text blocks must reside in markdown cells:
  ```python
  # %% [markdown]
  # # Title of the Cell
  # Description text goes here...
  ```
- Code blocks reside in standard code cells:
  ```python
  # %%
  import numpy as np
  print("Hello MLOps!")
  ```

### 2.3 Local LLM API Redirection
- **Endpoint:** Any LLM usage must use `http://localhost:5055/v1` as the base API URL (e.g., using `openai.OpenAI(base_url="http://localhost:5055/v1", api_key="dummy")`).
- **No Remote Calls:** Never hardcode OpenAI or Anthropic real keys or use external cloud APIs.

### 2.4 AWS Service Simulation
- **Storage Simulator:** Use S3 as the primary storage layout. Mock all calls using `moto`.
- **Implementation:** In code cells, apply `moto.mock_aws()` context managers or decorators to establish local mocked S3 environments.
- **DVC remote configuration:** DVC can store data locally or connect to a simulated local S3 server run via `moto_server s3`.

### 2.5 Containerization & Docker
- Each serving tutorial must include instructions on compiling a lightweight `Dockerfile`.
- Run health checks locally against the built docker container if Docker is available.

---

## 3. Agent Rules & Token Efficiency

- **Read Before Action:** Every agent step must review this instruction guide.
- **Minimize Builds:** Never build/serve the documentation with `mkdocs` automatically. The compile phase triggers expensive code executions and renders many static HTML files, wasting context windows and compute.
- **Python-first Validation:** Validate scripts by executing them directly: `uv run python src/<module>/<script>.py`.
- **Targeted Reading:** When reading workspace files, limit the view output (using line ranges) to the affected blocks.

---

## 4. Module Roadmap and TODOs

### Module 1: Modern Python Environment & Dependency Control (`uv`)
- [x] Understand `pyproject.toml` dependencies and locks.
- [x] Demonstrate virtual environment instantiation and sync with `uv sync`.
- [x] Execute scripts natively in the environment with `uv run`.

### Module 2: Documentation Setup (`mkdocs` + `mkdocs-jupyter`)
- [x] Render markdown headers and python code.
- [x] Review `mkdocs.yml` configurations (markdown extensions, plugins).
- [x] Establish standard layout theme configurations.

### Module 3: Data Versioning & Pipelines with DVC (`dvc`)
- [x] Initialize DVC inside the repository.
- [x] Track synthetic raw datasets in `data/raw/` and exclude them from git.
- [x] Configure local remote directory mimicking production cloud storage.

### Module 4: AWS Simulation with `boto3` & `moto`
- [x] Create mock S3 buckets using `moto`.
- [x] Upload and download model and data artifacts using `boto3` inside interactive python cells.

### Module 5: Experiment Tracking & Model Registry with MLflow (`mlflow`)
- [x] Run mock training scripts logging metrics (loss, accuracy) and parameters (epochs, learning rate).
- [x] Configure MLflow to write artifacts into a mock S3 bucket simulation.
- [x] Register the trained model inside MLflow Model Registry.

### Module 6: End-to-End MLOps Pipeline (DVC + MLflow)
- [x] Connect DVC pipeline steps (`dvc.yaml`) to run data preparation, model training, and evaluation.
- [x] Ensure the training script queries DVC tracked files and logs outputs directly to the MLflow server.

### Module 7: Model Serving API (FastAPI)
- [x] Load the trained model artifact from the registered path.
- [x] Wrap the inference script in a FastAPI app with a POST `/predict` endpoint.
- [x] Create a testing client to submit sample JSON input.

### Module 8: Docker Containerization
- [x] Write a production-grade `Dockerfile` for the serving API.
- [x] Detail commands to build, containerize, and inspect the served endpoints locally.

### Module 9: Model Monitoring & Drift Detection (`evidently`)
- [x] Create Reference and Current data profiles representing shifts in feature distributions.
- [x] Generate Evidently AI monitoring dashboards and export HTML drift reports.

### Module 10: CI/ML Quality Gates
- [x] Implement a programmatic model performance gatekeeper that flags metrics regression.
- [x] Integrate gating checks as part of the test suite and execution pipeline.

### Module 11: GitHub Actions CI Pipeline
- [x] Understand triggers, jobs, and steps in GitHub Actions configuration (`ci.yml`).
- [x] Automate syntax verification, data generation, pipeline execution, pytest runs, and container builds.


