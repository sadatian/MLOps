# =====================================================================
# Build Stage: Prepare the environment and install dependencies
# =====================================================================
FROM ghcr.io/astral-sh/uv:python3.12-alpine AS builder

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project specification files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (exclude code to speed up image layers cache)
RUN uv sync --frozen --no-install-project --no-dev

# =====================================================================
# Run Stage: Minimal runtime environment with no build tools
# =====================================================================
FROM python:3.12-alpine

WORKDIR /app

# Copy built virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Add virtualenv to path so we can run python directly
ENV PATH="/app/.venv/bin:$PATH"

# Copy model artifacts, application code
COPY data/model.pkl ./data/model.pkl
COPY src/module_07_model_serving/serve_api.py ./src/module_07_model_serving/serve_api.py

# Expose API port
EXPOSE 8000

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000

# Run the inference service using uvicorn
CMD ["uvicorn", "src.module_07_model_serving.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
