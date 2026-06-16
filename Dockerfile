# =====================================================================
# Build Stage: Prepare the environment and install dependencies
# =====================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

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
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy built virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Add virtualenv to path so we can run python/mlops directly
ENV PATH="/app/.venv/bin:$PATH"

# Copy package metadata and source files
COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/

# Install the package in editable mode within the virtual env
RUN pip install --no-deps -e .

# Expose API port
EXPOSE 8000

# Set default host and port environment variables
ENV HOST=0.0.0.0
ENV PORT=8000

# Entrypoint runs the unified mlops CLI, defaulting to model serving
ENTRYPOINT ["mlops"]
CMD ["serve"]
